from typing import Any

import torch
from transformers import BertTokenizer


def mnli_tokenize_function(
    data: dict[str, Any], tokenizer: BertTokenizer, max_length: int = 128
) -> dict[str, torch.Tensor]:
    return {
        k: torch.tensor(v)
        for k, v in tokenizer(
            data["premise"],
            data["hypothesis"],
            padding="max_length",
            max_length=max_length,
            truncation=True,
        ).items()
    }
