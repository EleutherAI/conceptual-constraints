from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from tqdm import tqdm
from transformers import AutoTokenizer


class HANSDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path = Path.cwd() / "hans",
        batch_size: int = 32,
        max_seq_len: int = 128,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        self.preprocess_data()

    def preprocess_data(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.train_set: list[dict[str, Any]] = []
        val_set: list[dict[str, Any]] = []

        # Preparing training set
        with open(self.data_dir / "heuristics_train_set.txt", "r") as f:
            train_lines = f.readlines()
            for idx, line in enumerate(
                tqdm(train_lines[1:], desc="Processing train set", leave=False)
            ):
                data = self.parse_line(line)
                data["idx"] = idx
                data.update(
                    tokenizer(
                        data["premise"],
                        data["hypothesis"],
                        return_tensors="pt",
                        padding="max_length",
                        max_length=self.max_seq_len,
                        truncation=True,
                    )
                )
                self.train_set.append(data)

        # Preparing validation set
        with open(self.data_dir / "heuristics_evaluation_set.txt", "r") as f:
            val_lines = f.readlines()
            for idx, line in enumerate(
                tqdm(val_lines[1:], desc="Processing val set", leave=False)
            ):
                data = self.parse_line(line)
                data["idx"] = idx
                data.update(
                    tokenizer(
                        data["premise"],
                        data["hypothesis"],
                        return_tensors="pt",
                        padding="max_length",
                        max_length=self.max_seq_len,
                        truncation=True,
                    )
                )
                val_set.append(data)

    @staticmethod
    def parse_line(line: str) -> dict[str, Any]:
        data: dict[str, Any] = {}
        row = line.split("\t")
        data["label"] = 1 if row[0] == "entailment" else 0
        data["premise"] = row[5]
        data["hypothesis"] = row[6]
        data["heuristic"] = row[8]
        data["subcase"] = row[9]
        return data


if __name__ == "__main__":
    dm = HANSDatamodule()
    dm.preprocess_data()
