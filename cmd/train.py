import random
import string
from functools import partial
from typing import Optional, Literal

import click
from datasets import load_dataset, concatenate_datasets, ClassLabel
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers.utils import logging

from oleace.datasets.hans import HANSDataset
from oleace.utils.callbacks import ConceptEraserCallback
from oleace.utils.concept import get_bert_concept_eraser, no_heuristic, build_heuristic_loader, build_entailment_loader
from oleace.utils.eval import compute_metrics, compute_metrics_auc
from oleace.utils.tokenization import tokenize_mnli

TrainDataset = Literal["mnli", "hansmnli"]

@click.command()
@click.option(
    "--concept_erasure", default=None, help="Concept erasure method to use (if any)."
)
@click.option(
    "--include_sublayers",
    is_flag=True,
    help="Include sublayers of BERT in the concept erasure method.",
)
@click.option(
    "--ema_beta",
    default=None,
    help="Exponential moving average beta for concept erasure.",
)
@click.option(
    "--local-rank",
    default=0,
    help="torch.distributed GPU rank",
)
@click.option(
    "--update_frequency",
    default=50,
    help="update every _ steps",
)
@click.option(
    "--gpus",
    default=1,
    help="divide batch size for data parallelism",
)
@click.option(
    "--dataset",
    default="hansmnli",
    help="training dataset",
)
@click.option(
    "--name",
    default=None,
    help="descriptive name for the run",
)
@click.option(
    "--imbalance",
    default=100,
    help="imbalance factor for HANS dataset",
)
@click.option(
    "--acc",
    is_flag=True,
    help="use accuracy-by-label instead of AUC for HANS evaluation",
)
@click.option(
    "--erase_labels",
    is_flag=True,
    help="LEACE the entailment labels",
)
def main(
    concept_erasure: Optional[str] = None, 
    include_sublayers: bool = False,
    ema_beta: Optional[float] = None,
    local_rank: int = 0,
    update_frequency: Optional[int] = 50,
    gpus: int = 1,
    dataset: TrainDataset = "hansmnli",
    name: Optional[str] = None,
    imbalance: int = 100,
    acc: bool = False,
    erase_labels: bool = False,
) -> None:
    
    auc = not acc

    # Initialize Weights and Biases
    try:
        import wandb

        run = wandb.init(project="oleace", name=f"{name}#{local_rank}" if name is not None else None)
        assert run is not None
        run_id = run.id

    # If Weights and Biases is not installed, set run_id to random string of size 7
    except ImportError:
        run_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=7))

    logging.set_verbosity_info()
    logger = logging.get_logger("transformers")

    logger.info(
        "Welcome in the finetuning script for BERT on MNLI with evaluation on HANS!"
    )

    # Prepare MNLI dataset
    logger.info("Loading MNLI dataset.")
    mnli_train_dataset = load_dataset("multi_nli", split="train")
    mnli_val_dataset = load_dataset("multi_nli", split="validation_matched")

    # Tokenize MNLI dataset
    logger.info("Tokenizing MNLI dataset.")
    mnli_train_dataset = tokenize_mnli(mnli_train_dataset)
    mnli_val_dataset = tokenize_mnli(mnli_val_dataset)

    match dataset:
        case "mnli":
            train_dataset = mnli_train_dataset
            val_dataset = mnli_val_dataset
        case "hansmnli":
            # Prepare HANS dataset
            hans_train_dataset = load_dataset("hans", split="train")
            hans_val_dataset = load_dataset("hans", split="validation")

            # Tokenize HANS dataset
            hans_train_dataset = tokenize_mnli(hans_train_dataset)
            hans_val_dataset = tokenize_mnli(hans_val_dataset)

            # shuffle HANS
            hans_train_dataset = hans_train_dataset.shuffle(seed=42)
            hans_val_dataset = hans_val_dataset.shuffle(seed=42)
            # get first third of HANS examples with label 1
            hans_train_noentail = hans_train_dataset.filter(lambda example: example["label"] == 1)
            hans_val_noentail = hans_val_dataset.filter(lambda example: example["label"] == 1)
            hans_train_noentail = hans_train_noentail.select(range(len(hans_train_noentail) // imbalance))
            hans_val_noentail = hans_val_noentail.select(range(len(hans_val_noentail) // imbalance))
            # get all HANS examples with label 0
            hans_train_entail = hans_train_dataset.filter(lambda example: example["label"] == 0)
            hans_val_entail = hans_val_dataset.filter(lambda example: example["label"] == 0)
            # concatenate HANS examples
            hans_train_dataset = concatenate_datasets([hans_train_noentail, hans_train_entail])
            hans_val_dataset = concatenate_datasets([hans_val_noentail, hans_val_entail])

            # train: cut down MNLI to match HANS
            mnli_train_dataset = mnli_train_dataset.shuffle(seed=42)
            mnli_train_dataset = mnli_train_dataset.select(range(len(hans_train_dataset)))
            # take MNLI examples with no applicable heuristic
            mnli_train_dataset = mnli_train_dataset.filter(no_heuristic)
            mnli_val_dataset = mnli_val_dataset.filter(no_heuristic)
            # val: cut down HANS to match MNLI
            hans_val_dataset = hans_val_dataset.shuffle(seed=42)
            hans_val_dataset = hans_val_dataset.select(range(len(mnli_val_dataset)))

            # rename HANS 'binary_parse_premise' to 'premise_binary_parse'
            # and 'binary_parse_hypothesis' to 'hypothesis_binary_parse'
            hans_train_dataset = hans_train_dataset.rename_column("binary_parse_premise", "premise_binary_parse")
            hans_train_dataset = hans_train_dataset.rename_column("binary_parse_hypothesis", "hypothesis_binary_parse")
            hans_val_dataset = hans_val_dataset.rename_column("binary_parse_premise", "premise_binary_parse")
            hans_val_dataset = hans_val_dataset.rename_column("binary_parse_hypothesis", "hypothesis_binary_parse")

            # relabel MNLI 'label' 1 and 2 both to 1
            mnli_train_dataset = mnli_train_dataset.map(lambda example: {"label": 1 if example["label"] > 0 else 0})
            mnli_val_dataset = mnli_val_dataset.map(lambda example: {"label": 1 if example["label"] > 0 else 0})
            # change MNLI 'label' from ClassLabel(names=['entailment', 'neutral', 'contradiction']) 
            # to ClassLabel(names=['entailment', 'non-entailment'])
            mnli_features = mnli_train_dataset.features.copy()
            mnli_features["label"] = ClassLabel(names=["entailment", "non-entailment"])
            mnli_train_dataset = mnli_train_dataset.cast(mnli_features)
            mnli_val_dataset = mnli_val_dataset.cast(mnli_features)

            # concatenate HANS and MNLI examples
            hans_columns = set(hans_train_dataset.column_names)
            mnli_columns = set(mnli_train_dataset.column_names)
            hans_only_columns = hans_columns - mnli_columns
            mnli_only_columns = mnli_columns - hans_columns
            hans_train_dataset = hans_train_dataset.remove_columns(list(hans_only_columns))
            mnli_train_dataset = mnli_train_dataset.remove_columns(list(mnli_only_columns))
            hans_val_dataset = hans_val_dataset.remove_columns(list(hans_only_columns))
            mnli_val_dataset = mnli_val_dataset.remove_columns(list(mnli_only_columns))
            train_dataset = concatenate_datasets([hans_train_dataset, mnli_train_dataset])
            val_dataset = concatenate_datasets([hans_val_dataset, mnli_val_dataset])

    num_labels_dict = {
        "mnli": 3,
        "hansmnli": 2,
    }

    # Load BERT model and tokenizer
    logger.info("Loading BERT model.")
    bert = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels_dict[dataset]
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/{run_id}",
        num_train_epochs=3,
        per_device_train_batch_size=32 // gpus,
        per_device_eval_batch_size=64 // gpus,
        learning_rate=2e-5,
        logging_dir=f"./logs/{run_id}",
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        save_total_limit=5,
        # fp16=True,
        tf32=True,
    )

    # If concept erasure is specified, create the concept eraser callback
    if concept_erasure is not None:
        logger.info(f"Creating concept erasure callback using {concept_erasure}.")
        build_loader = build_entailment_loader if erase_labels else build_heuristic_loader
        concept_data_loader = build_loader(train_dataset)
        concept_eraser = get_bert_concept_eraser(
            bert=bert,
            concept_erasure=concept_erasure,
            include_sublayers=include_sublayers,
            ema_beta=ema_beta,
            num_concepts=1 if erase_labels else 3,
        )
        concept_eraser_callback = ConceptEraserCallback(
            concept_eraser=concept_eraser, 
            concept_data_loader=concept_data_loader,
            update_frequency=update_frequency,
        )

    # Define trainer
    trainer = Trainer(
        model=bert,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=partial(compute_metrics, dataset_name=dataset),
        callbacks=[concept_eraser_callback] if concept_erasure is not None else None,
    )

    # Finetune model
    logger.info("Finetuning model.")
    trainer.train()

    # Prepare HANS dataset w/ legacy code
    logger.info("Loading HANS dataset for evals.")
    hans_train = HANSDataset(split="train")
    hans_eval = HANSDataset(split="val")

    # Evaluate model on HANS dataset
    logger.info("Evaluating model on HANS dataset.")
    trainer.compute_metrics = compute_metrics_auc if auc else partial(compute_metrics, dataset_name="hans")
    eval_results = trainer.evaluate(
        eval_dataset=hans_train.get_splits(split_entail=not auc) | hans_eval.get_splits(split_entail=not auc)
    )
    logger.info(eval_results)


if __name__ == "__main__":
    main()
