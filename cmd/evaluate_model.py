import json
from functools import partial
from pathlib import Path
from typing import Optional

import click
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers.utils import logging

from oleace.datasets.hans import HANSDataset
from oleace.utils.concept import bert_erase_heuristic
from oleace.utils.eval import compute_metrics, get_latest_checkpoint


@click.command()
@click.argument("run_id")
@click.option(
    "--concept-erasure", default=None, help="Concept erasure method to use (if any)."
)
def main(run_id: str, concept_erasure: Optional[str] = None) -> None:
    logging.set_verbosity_info()
    logger = logging.get_logger("transformers")

    logger.info("Welcome in the evaluation script on HANS!")

    # Load the model
    logger.info(f"Loading BERT model with {run_id=}.")
    latest_checkpoint_dir = get_latest_checkpoint(f"./results/{run_id}")
    model = AutoModelForSequenceClassification.from_pretrained(latest_checkpoint_dir)
    model.cuda()

    # Erase heuristic concepts if necessary
    if concept_erasure is not None:
        logger.info(f"Erasing heuristic concepts using {concept_erasure}.")
        bert_erase_heuristic(bert=model, concept_erasure=concept_erasure)

    # Load the HANS dataset
    logger.info("Loading HANS dataset.")
    hans_eval = HANSDataset(split="val")
    hans_train = HANSDataset(split="train")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/{run_id}",
        per_device_eval_batch_size=1 if concept_erasure == "leace-flatten" else 64,
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
    with open(
        Path(latest_checkpoint_dir)
        / f"hans_results_{concept_erasure if concept_erasure is not None else 'default'}.json",
        "w",
    ) as f:
        json.dump(eval_results, f)


if __name__ == "__main__":
    main()
