import json
from functools import partial
from pathlib import Path

import click
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers.utils import logging

from oleace.datasets.hans import HANSDataset
from oleace.utils.eval import compute_metrics, get_latest_checkpoint


@click.command()
@click.argument("run_id")
def main(run_id: str) -> None:
    logging.set_verbosity_info()
    logger = logging.get_logger("transformers")

    logger.info("Welcome in the evaluation script on HANS!")

    # Load the model
    logger.info(f"Loading BERT model with {run_id=}.")
    latest_checkpoint_dir = get_latest_checkpoint(f"./results/{run_id}")
    model = AutoModelForSequenceClassification.from_pretrained(latest_checkpoint_dir)

    # Load the HANS dataset
    logger.info("Loading HANS dataset.")
    hans_eval = HANSDataset(split="val")
    hans_train = HANSDataset(split="train")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/{run_id}",
        per_device_eval_batch_size=64,
        logging_dir=f"./logs/{run_id}",
        do_eval=True,
        evaluation_strategy="epoch",
        report_to=None,
    )

    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=partial(compute_metrics, dataset_name="hans"),
    )

    # Evaluate model on HANS dataset
    logger.info("Evaluating model on HANS dataset.")
    eval_results = trainer.evaluate(
        eval_dataset=hans_train.get_splits() | hans_eval.get_splits()
    )

    # Just keep the useful entries in the dictionary
    eval_results = {
        k: v for k, v in eval_results.items() if ("loss" in k or "accuracy" in k)
    }
    logger.info(eval_results)

    # Save evaluation results in the checkpoint directory in a JSON file
    with open(Path(latest_checkpoint_dir) / "hans_results.json", "w") as f:
        json.dump(eval_results, f)


if __name__ == "__main__":
    main()
