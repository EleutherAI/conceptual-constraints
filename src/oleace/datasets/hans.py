from pathlib import Path
from typing import Any

from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from oleace.utils.tokenize import mnli_tokenize_function


class HANSDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        data_dir: Path = Path.cwd() / "data/hans",
        batch_size: int = 32,
        max_length: int = 128,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self.split: str = split
        self.preprocess_data()

    def preprocess_data(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.data_list: list[dict[str, Any]] = []

        match self.split:
            case "train":
                data_path = self.data_dir / "heuristics_train_set.txt"
            case "val":
                data_path = self.data_dir / "heuristics_evaluation_set.txt"
            case _:
                raise ValueError("Invalid split")

        # Preparing training set
        with open(data_path, "r") as f:
            lines = f.readlines()
            for idx, line in enumerate(
                tqdm(
                    lines[1:],
                    desc="Processing HANS dataset",
                    leave=False,
                    units="examples",
                )
            ):
                data = self.parse_line(line)
                data["idx"] = idx
                data.update(
                    mnli_tokenize_function(
                        data=data, tokenizer=tokenizer, max_length=self.max_length
                    )
                )
                self.data_list.append(data)

    @staticmethod
    def parse_line(line: str) -> dict[str, Any]:
        data: dict[str, Any] = {}
        row = line.split("\t")
        data["labels"] = 0 if row[0] == "entailment" else 1
        data["premise"] = row[5]
        data["hypothesis"] = row[6]
        data["heuristic"] = row[8]
        data["subcase"] = row[9]
        return data

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.data_list[idx]
