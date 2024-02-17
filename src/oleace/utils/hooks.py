from abc import ABC, abstractmethod
from typing import Callable, Optional

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
    def __init__(
        self, module_list: list[Module], 
        num_concepts: int = 2,
        ema_beta: Optional[float] = None,
        only_fit: bool = False,
    ):
        super().__init__(module_list)
        self.num_concepts = num_concepts
        self.ema_beta = ema_beta
        self.only_fit = only_fit
        self.erasers: dict[nn.Module, LeaceFitter | None | LeaceEraser] = {
            module: None for module in module_list
        }
        self.current_labels = None

    def fit(self, model: nn.Module, data_loader: DataLoader) -> None:
        self.register_hooks(self.fit_hook)
        batch: dict[str, torch.Tensor]
        for batch in data_loader:

            # Remove labels and promptID from batch
            if "labels" in batch:
                batch.pop("labels")
            if "promptID" in batch:
                batch.pop("promptID")

            # Stash the concept labels
            self.current_labels = batch.pop("concept_labels")

            # Move batch to the same device as the model
            batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}

            # Forward pass
            model(**batch)

        if self.only_fit:
            for module in self.module_list:
                eraser = self.erasers[module]
                print("Computing cosine similarity (col 1 is entailment)")
                print("================================")
                sigma_xz = eraser.sigma_xz

                # Compute the cosine similarity between columns of sigma_xz
                for i in list(range(2, sigma_xz.shape[1]))+[0, 1]:
                    for j in range(i + 1, sigma_xz.shape[1]):
                        sim = F.cosine_similarity(sigma_xz[:, i], sigma_xz[:, j], dim=0)
                        print(f"Similarity between concept {i} and {j}: {sim.item()}")
                
                # Project column 1 onto subspace spanned by columns [2:]
                entail = sigma_xz[:, 1]
                u = sigma_xz[:, 2:].svd().U
                p = u @ u.T
                cosine = (p @ entail).norm() / entail.norm()
                print(f"Total cosine similarity between entailment and the rest: {cosine.item()}")

        self.remove_hooks()

    def activate_eraser(self) -> None:
        self.register_hooks(self.erase_hook)

    def deactivate_eraser(self) -> None:
        self.remove_hooks()

    def reset_eraser(self) -> None:
        self.erasers = {module: None for module in self.module_list}

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


class LeaceCLS(ConceptEraser):
    @torch.autocast("cuda", enabled=False)
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

        # Retrieve concept labels
        labels = self.current_labels
        assert labels is not None

        # Instantiate LEACE eraser if necessary
        if self.erasers[module] is None:
            self.erasers[module] = LeaceFitter(
                cls_rep.shape[-1], self.num_concepts + 1, 
                device=cls_rep.device,
                ema_beta=self.ema_beta,
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
        
        if self.only_fit:
            return output

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

        # Retrieve concept labels
        labels = self.current_labels
        assert labels is not None

        # Instantiate LEACE eraser if necessary (with orth flag to save vRAM)
        if self.erasers[module] is None:
            self.erasers[module] = LeaceFitter(
                sequence_rep.shape[-1],
                self.num_concepts + 1,
                device=sequence_rep.device,
                method="orth",
                ema_beta=self.ema_beta,
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
        
        if self.only_fit:
            return output

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
