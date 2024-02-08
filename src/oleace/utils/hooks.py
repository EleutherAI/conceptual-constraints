from typing import Callable

import torch
import torch.nn as nn
from concept_erasure import LeaceFitter
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle


class HookManager:
    def __init__(self, module_list: list[nn.Module]):
        self.module_list = module_list
        self.handles: list[RemovableHandle] = []

    def register_hooks(self, hook_fn: Callable) -> None:
        for module in self.module_list:
            handle = module.register_forward_hook(hook_fn)
            self.handles.append(handle)

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


class LeaceCLS(HookManager):
    def __init__(self, module_list: list[nn.Module], num_concepts: int = 2):
        super().__init__(module_list)
        self.num_concepts = num_concepts
        self.leace_erasers: dict[nn.Module, LeaceFitter | None] = {
            module: None for module in module_list
        }

    def fit(self, model: nn.Module, data_loader: DataLoader) -> None:
        self.register_hooks(self.leace_fit_hook)
        for batch in data_loader:
            model(batch)
        self.remove_hooks()

    def activate_eraser(self) -> None:
        self.register_hooks(self.leace_erase_hook)

    def deactivate_eraser(self) -> None:
        self.remove_hooks()

    def leace_fit_hook(
        self,
        module: nn.Module,
        input: torch.Tensor,
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:

        # Extract the representation from the CLS token
        if isinstance(output, tuple):
            cls_rep = output[0][:, 0, :]
        else:
            cls_rep = output[:, 0, :]

        # Assign concept labels by assuming that the order is [concept_0, concept_1, ..., concept_n, concept_0, ...]
        assert (
            cls_rep.shape[0] % self.num_concepts == 0
        ), f"The batch size {cls_rep.shape[0]} be divisible by the number of concepts {self.num_concepts}."
        n_per_concept = cls_rep.shape[0] // self.num_concepts
        labels = torch.arange(
            self.num_concepts, dtype=torch.long, device=cls_rep.device
        ).repeat(n_per_concept)

        if self.leace_erasers[module] is None:
            self.leace_erasers[module] = LeaceFitter(
                cls_rep.shape[-1], self.num_concepts - 1
            )

        leace_eraser = self.leace_erasers[module]
        assert leace_eraser is not None
        leace_eraser.update(cls_rep, labels)

    def leace_erase_hook(
        self,
        module: nn.Module,
        input: torch.Tensor,
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        # Extract the representation from the CLS token
        if isinstance(output, tuple):
            cls_rep = output[0][:, 0, :]
        else:
            cls_rep = output[:, 0, :]

        # Erase the concept from the representation of the CLS token
        leace_eraser = self.leace_erasers[module]
        assert leace_eraser is not None
        new_cls_rep = leace_eraser.eraser(cls_rep)

        # Replace the representation of the CLS token with the erased representation
        output[0][:, 0, :] = new_cls_rep
        return output
