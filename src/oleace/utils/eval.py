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
