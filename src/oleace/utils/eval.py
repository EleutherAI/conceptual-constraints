from pathlib import Path

import evaluate
import numpy as np


def compute_metrics(
    eval_pred: tuple[np.ndarray, np.ndarray], dataset_name: str = "mnli"
) -> dict[str, float]:
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    match dataset_name:
        case "mnli":
            predictions = np.argmax(logits, axis=-1)
        case "hans":
            predictions = np.argmax(logits, axis=-1)
            predictions = np.where(predictions == 0, 0, 1)

    return metric.compute(predictions=predictions, references=labels)  # type: ignore


def get_latest_checkpoint(checkpoints_dir: str) -> str:

    # Check if the directory exists and is not empty
    assert Path(
        checkpoints_dir
    ).exists(), f"Directory {checkpoints_dir} does not exist."
    assert (
        len(list(Path(checkpoints_dir).iterdir())) > 0
    ), f"Directory {checkpoints_dir} is empty."

    # Find the latest checkpoint
    current_last_step = 0
    for path in sorted(Path(checkpoints_dir).iterdir()):
        if "checkpoint-" in path.name:
            checkpoint_step = int(path.name.split("checkpoint-")[-1])
            if checkpoint_step > current_last_step:
                current_last_step = checkpoint_step
                latest_checkpoint = path
    return str(latest_checkpoint)
