from functools import partial
from typing import Any

import datasets
import torch
from transformers import AutoTokenizer, BertTokenizer


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


def tokenize_mnli(dataset: datasets.Dataset) -> datasets.Dataset:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenize_fn = partial(mnli_tokenize_function, tokenizer=tokenizer, max_length=128)
    return dataset.map(tokenize_fn, batched=True)
