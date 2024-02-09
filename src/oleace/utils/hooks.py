from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from concept_erasure import LeaceEraser, LeaceFitter
from torch.nn.modules import Module
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle
from tqdm import tqdm


class HookManager(ABC):
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


class ConceptEraser(HookManager, ABC):
    def __init__(self, module_list: list[Module], num_concepts: int = 2):
        super().__init__(module_list)
        self.num_concepts = num_concepts
        self.erasers: dict[nn.Module, LeaceFitter | None | LeaceEraser] = {
            module: None for module in module_list
        }

    def fit(self, model: nn.Module, data_loader: DataLoader) -> None:
        self.register_hooks(self.fit_hook)
        batch: dict[str, torch.Tensor]
        for batch in data_loader:

            # Remove labels and promptID from batch
            if "labels" in batch:
                batch.pop("labels")
            if "promptID" in batch:
                batch.pop("promptID")

            # Move batch to the same device as the model
            batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}

            # Forward pass
            model(**batch)

        self.remove_hooks()

    def activate_eraser(self) -> None:
        self.register_hooks(self.erase_hook)

    def deactivate_eraser(self) -> None:
        self.remove_hooks()

    @abstractmethod
    def fit_hook(
        self,
        module: nn.Module,
        input: torch.Tensor,
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
        pass

    @abstractmethod
    def erase_hook(
        self,
        module: nn.Module,
        input: torch.Tensor,
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        pass

    def concept_label_assignment(self, representation: torch.Tensor) -> torch.Tensor:
        # Assign concept labels by assuming that the order is [concept_0, concept_1, ..., concept_n, concept_0, ...]
        assert (
            representation.shape[0] % (self.num_concepts + 1) == 0
        ), f"The batch size {representation.shape[0]} be divisible by  {self.num_concepts + 1}."
        n_per_concept = representation.shape[0] // (self.num_concepts + 1)
        labels = torch.arange(
            self.num_concepts + 1, dtype=torch.long, device=representation.device
        ).repeat(n_per_concept)
        labels = F.one_hot(labels, num_classes=self.num_concepts + 1)
        return labels


class LeaceCLS(ConceptEraser):
    def fit_hook(
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

        # Assign concept labels
        labels = self.concept_label_assignment(cls_rep)

        # Instantiate LEACE eraser if necessary
        if self.erasers[module] is None:
            self.erasers[module] = LeaceFitter(
                cls_rep.shape[-1], self.num_concepts + 1, device=cls_rep.device
            )

        # Update LEACE eraser with the right statistics
        leace_eraser = self.erasers[module]
        assert isinstance(leace_eraser, LeaceFitter)
        leace_eraser.update(cls_rep, labels)

    def erase_hook(
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
        leace_eraser = self.erasers[module]
        assert isinstance(leace_eraser, LeaceFitter)
        cls_rep = leace_eraser.eraser(cls_rep)

        # Replace the representation of the CLS token with the erased representation
        if isinstance(output, tuple):
            output[0][:, 0, :] = cls_rep
        else:
            output[:, 0, :] = cls_rep
        return output


class LeaceFlatten(ConceptEraser):
    def fit_hook(
        self,
        module: nn.Module,
        input: torch.Tensor,
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:

        # Extract the sequence representation token
        if isinstance(output, tuple):
            sequence_rep = output[0]
        else:
            sequence_rep = output

        # Flatten the sequence representation
        sequence_rep = sequence_rep.flatten(start_dim=1)

        # Assign concept labels by assuming that the order is [concept_0, concept_1, ..., concept_n, concept_0, ...]
        labels = self.concept_label_assignment(sequence_rep)

        # Instantiate LEACE eraser if necessary (with orth flag to save vRAM)
        if self.erasers[module] is None:
            self.erasers[module] = LeaceFitter(
                sequence_rep.shape[-1],
                self.num_concepts + 1,
                device=sequence_rep.device,
                method="orth",
            )

        leace_eraser = self.erasers[module]
        assert isinstance(leace_eraser, LeaceFitter)
        leace_eraser.update(sequence_rep, labels)

    def erase_hook(
        self,
        module: nn.Module,
        input: torch.Tensor,
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:

        # Extract the sequence representation
        if isinstance(output, tuple):
            sequence_rep = output[0]
        else:
            sequence_rep = output

        # Flatten the sequence representation
        sequence_rep_shape = sequence_rep.shape
        sequence_rep = sequence_rep.flatten(start_dim=1)

        # Erase the concept from the representation of the CLS token
        leace_eraser = self.erasers[module]
        assert isinstance(leace_eraser, LeaceFitter)
        sequence_rep = leace_eraser.eraser(sequence_rep)
        assert isinstance(sequence_rep, torch.Tensor)

        # Unflatten the sequence representation
        sequence_rep = sequence_rep.unflatten(dim=1, sizes=sequence_rep_shape[1:])

        # Replace the sequence representation with the erased representation
        if isinstance(output, tuple):
            return tuple([sequence_rep] + list(output[1:]))
        else:
            return sequence_rep
