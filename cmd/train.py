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
from oleace.utils.tokenize import mnli_tokenize_function


def main() -> None:

    logging.set_verbosity_info()
    logger = logging.get_logger("transformers")

    logger.info(
        "Welcome in the finetuning script for BERT on MNLI with evaluation on HANS!"
    )

    # Prepare MNLI dataset
    logger.info("Loading MNLI dataset.")
    train_dataset = load_dataset("glue", "mnli", split="train")
    val_dataset = load_dataset("glue", "mnli", split="validation_matched")

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
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        logging_dir="./logs",
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
        eval_dataset={
            "hans_eval": hans_eval,
            "hans_train": hans_train,
        }
    )
    logger.info(eval_results)


if __name__ == "__main__":
    main()
