import random
import string
from functools import partial

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.utils import logging

from oleace.datasets.hans import HANSDataset
from oleace.utils.eval import compute_metrics
from oleace.utils.tokenization import mnli_tokenize_function


def main() -> None:

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
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenize_fn = partial(mnli_tokenize_function, tokenizer=tokenizer, max_length=128)
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    val_dataset = val_dataset.map(tokenize_fn, batched=True)

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
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        logging_dir=f"./logs/{run_id}",
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
    )

    # Define trainer
    trainer = Trainer(
        model=bert,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, dataset_name="mnli"),
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
