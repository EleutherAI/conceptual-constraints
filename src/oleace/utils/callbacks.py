from typing import Any

import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from transformers.utils import logging

from .hooks import ConceptEraser


class ConceptEraserCallback(TrainerCallback):
    def __init__(
        self,
        concept_eraser: ConceptEraser,
        concept_data_loader: DataLoader,
        update_frequency: int = 50,
    ):
        super().__init__()
        self.logger = logging.get_logger("transformers")
        self.concept_eraser = concept_eraser
        self.concept_data_loader = concept_data_loader
        self.update_frequency = update_frequency

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: nn.Module,
        **kwargs: dict[str, Any],
    ) -> None:
        if state.global_step % self.update_frequency == 0:
            self.update_concept_eraser(model)

    def on_evaluate_start(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: nn.Module,
        **kwargs: dict[str, Any],
    ) -> None:
        self.update_concept_eraser(model)

    def update_concept_eraser(self, model: nn.Module) -> None:
        self.concept_eraser.reset_eraser()
        self.concept_eraser.deactivate_eraser()
        self.concept_eraser.fit(model, self.concept_data_loader)
        self.concept_eraser.activate_eraser()
