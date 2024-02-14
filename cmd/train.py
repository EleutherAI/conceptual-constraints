import random
import string
from functools import partial
from typing import Optional

import click
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers.utils import logging

from oleace.datasets.hans import HANSDataset
from oleace.utils.callbacks import ConceptEraserCallback
from oleace.utils.concept import build_mnli_heuristic_loader, get_bert_concept_eraser
from oleace.utils.eval import compute_metrics
from oleace.utils.tokenization import tokenize_mnli


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
    "--local-rank",
    default=None,
    help="distributed GPU rank",
)
@click.option(
    "--update_frequency",
    default=50,
    help="update every _ steps",
)
def main(
    concept_erasure: Optional[str] = None, 
    include_sublayers: bool = False,
    local_rank: Optional[int] = None,
    update_frequency: Optional[int] = 50,
) -> None:

    # Initialize Weights and Biases
    try:
        import wandb

        run = wandb.init(project="oleace")
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
    train_dataset = load_dataset("multi_nli", split="train")
    val_dataset = load_dataset("multi_nli", split="validation_matched")

    # Tokenize MNLI dataset
    logger.info("Tokenizing MNLI dataset.")
    train_dataset = tokenize_mnli(train_dataset)
    val_dataset = tokenize_mnli(val_dataset)

    # Prepare HANS dataset
    logger.info("Loading HANS dataset.")
    hans_train = HANSDataset(split="train")
    hans_eval = HANSDataset(split="val")

    # Load BERT model and tokenizer
    logger.info("Loading BERT model.")
    bert = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=3
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/{run_id}",
        num_train_epochs=3,
        per_device_train_batch_size=32//8,
        per_device_eval_batch_size=64//8,
        learning_rate=2e-5,
        logging_dir=f"./logs/{run_id}",
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        # fp16=True,
        tf32=True,
    )

    # If concept erasure is specified, create the concept eraser callback
    if concept_erasure is not None:
        logger.info(f"Creating concept erasure callback using {concept_erasure}.")
        concept_data_loader = build_mnli_heuristic_loader()
        concept_eraser = get_bert_concept_eraser(
            bert=bert,
            concept_erasure=concept_erasure,
            include_sublayers=include_sublayers,
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
        compute_metrics=partial(compute_metrics, dataset_name="mnli"),
        callbacks=[concept_eraser_callback] if concept_erasure is not None else None,
    )

    # Finetune model
    logger.info("Finetuning model.")
    trainer.train()

    # Evaluate model on HANS dataset
    logger.info("Evaluating model on HANS dataset.")
    trainer.compute_metrics = partial(compute_metrics, dataset_name="hans")
    eval_results = trainer.evaluate(
        eval_dataset=hans_train.get_splits() | hans_eval.get_splits()
    )
    logger.info(eval_results)


if __name__ == "__main__":
    main()
