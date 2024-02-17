# The concept assignment for the HANS heuristics has been adapted from
# https://github.com/tommccoy1/hans/blob/master/heuristic_finder_scripts/const_finder.py

from collections import defaultdict
from typing import Literal, Optional, Callable

from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification, DefaultDataCollator

from .hooks import ConceptEraser, LeaceCLS, LeaceFlatten
from .tokenization import tokenize_mnli

Concepts = Literal["labels", "heuristics", "both"]

def parse_phrase_list(parse: str, phrases: list[str]) -> list[str]:
    if parse == "":
        return phrases

    phrase_list = phrases

    words = parse.split()
    this_phrase: list[str] = []
    next_level_parse = []
    for word in words:
        if word == "(":
            next_level_parse += this_phrase
            this_phrase = ["("]

        elif word == ")" and len(this_phrase) > 0 and this_phrase[0] == "(":
            phrase_list.append(" ".join(this_phrase[1:]))
            next_level_parse += this_phrase[1:]
            this_phrase = []
        elif word == ")":
            next_level_parse += this_phrase
            next_level_parse.append(")")
            this_phrase = []
        else:
            this_phrase.append(word)

    return parse_phrase_list(" ".join(next_level_parse), phrase_list)


def is_constituent(data: dict[str, str]) -> bool:
    hypothesis = data["hypothesis"]
    parse = data["premise_binary_parse"]

    parse_new = []
    for word in parse.split():
        if word not in [".", "?", "!"]:
            parse_new.append(word.lower())

    all_phrases = parse_phrase_list(" ".join(parse_new), [])

    hyp_words = []

    for word in hypothesis.split():
        if word not in [".", "?", "!"]:
            hyp_words.append(
                word.lower().replace(".", "").replace("?", "").replace("!", "")
            )

    hyp_filtered = " ".join(hyp_words)

    if hyp_filtered in all_phrases:
        assert is_subsequence(data)
        return True
    return False


def is_lexical_overlap(data: dict[str, str]) -> bool:
    premise = data["premise"]
    hypothesis = data["hypothesis"]

    prem_words = []
    hyp_words = []

    for word in premise.split():
        if word not in [".", "?", "!"]:
            prem_words.append(word.lower())

    for word in hypothesis.split():
        if word not in [".", "?", "!"]:
            hyp_words.append(word.lower())

    all_in = True

    for word in hyp_words:
        if word not in prem_words:
            all_in = False
            break

    return all_in


def is_subsequence(data: dict[str, str]) -> bool:
    premise = data["premise"]
    hypothesis = data["hypothesis"]

    prem_words = []
    hyp_words = []

    for word in premise.split():
        if word not in [".", "?", "!"]:
            prem_words.append(word.lower())

    for word in hypothesis.split():
        if word not in [".", "?", "!"]:
            hyp_words.append(word.lower())

    prem_filtered = " ".join(prem_words)
    hyp_filtered = " ".join(hyp_words)

    if hyp_filtered in prem_filtered:
        assert is_lexical_overlap(data)
        return True
    return False


def no_heuristic(data: dict[str, str]) -> bool:
    return not (is_constituent(data) or is_lexical_overlap(data) or is_subsequence(data))


def build_mnli_heuristic_loader(batch_size: int = 32) -> DataLoader:
    # Get the MNLI dataset and tokenize it
    mnli_dataset = load_dataset("multi_nli", split="train")
    mnli_dataset = tokenize_mnli(mnli_dataset)

    return build_heuristic_loader(mnli_dataset, batch_size)


def build_heuristic_loader(dataset, batch_size: int = 32) -> DataLoader:
    concept_detectors = {
        "lexical_overlap": lambda data: is_lexical_overlap(data) and not is_subsequence(data),
        "subsequence": lambda data: is_subsequence(data) and not is_constituent(data),
        "constituent": is_constituent,
    }

    return build_concept_loader(dataset, concept_detectors, batch_size=batch_size)


def build_entailment_loader(dataset, batch_size: int = 32) -> DataLoader:
    concept_detectors = {
        "positive": lambda data: data["label"] == 0,
    }

    return build_concept_loader(dataset, concept_detectors, batch_size=batch_size)


def build_concept_loader(dataset, 
                         concept_detectors: dict[str, Callable],
                         batch_size: int = 32, 
                         ) -> DataLoader:
    num_concepts = len(concept_detectors)

    assert batch_size > 0, "The batch size must be positive."
    assert (
        batch_size % (num_concepts + 1) == 0
    ), f"The batch size must be divisible by {num_concepts+1} as it needs to be balanced across {num_concepts} concepts and 1 negative."

    # Make a list of indices for each concept
    concept_indices: dict[str, list[int]] = defaultdict(list)
    concepts = ["negative"] + list(concept_detectors.keys())
    new_column = []
    for idx, example_data in enumerate(
        tqdm(
            dataset,
            desc="Assigning concepts to dataset examples",
            unit=" examples",
            leave=False,
        )
    ):
        no_concept_assigned = True
        concept_labels = [0] * (num_concepts + 1)
        for concept, detector in concept_detectors.items():
            if detector(example_data):
                concept_indices[concept].append(idx)
                concept_labels[concepts.index(concept)] = 1
                no_concept_assigned = False
        if no_concept_assigned:
            concept_indices["negative"].append(idx)
            concept_labels[0] = 1
        
        new_column.append(concept_labels)
    
    dataset = dataset.add_column("concept_labels", new_column)

    # Measure the minimum concept size
    min_size = min(len(indices) for indices in concept_indices.values())

    # Create an alternative sequence of concept indices
    concept_sequence = []
    for i in range(min_size):
        for concept, indices in concept_indices.items():
            concept_sequence.append(indices[i])

    # Create a subset of the dataset with the concept sequence
    dataset = dataset.select(concept_sequence)

    # Return a DataLoader with the concept set
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DefaultDataCollator(),
    )


def get_bert_concept_eraser(
    bert: BertForSequenceClassification,
    concept_erasure: str,
    include_sublayers: bool = False,
    layers: Optional[list[int]] = None,
    ema_beta: Optional[float] = None,
    num_concepts: int = 3,
) -> ConceptEraser:
    
    if layers is not None:
        assert all(isinstance(layer, int) for layer in layers), \
            f"The layers must be a list of integers. {repr(layers), type(layers), [type(layer) for layer in layers]}"

    # Get a list of BERT layers (i.e. all transformer blocks)
    bert_layers = list(bert.bert.encoder.layer.children())

    if layers is not None:
        bert_layers = [bert_layers[i] for i in layers]

    if include_sublayers:
        bert_layers = [module for layer in bert_layers for module in layer.children()]

    # Apply the right concept erasure method to these layers
    match concept_erasure:
        case "leace-cls":
            return LeaceCLS(bert_layers, num_concepts=num_concepts, ema_beta=ema_beta)
        case "leace-flatten":
            return LeaceFlatten(bert_layers, num_concepts=num_concepts, ema_beta=ema_beta)
        case _:
            raise ValueError(f"Invalid concept erasure method: {concept_erasure}.")


def bert_erase_heuristic(
    bert: BertForSequenceClassification,
    concept_erasure: str = "leace-cls",
    include_sublayers: bool = False,
) -> None:

    concept_eraser = get_bert_concept_eraser(
        bert, concept_erasure, include_sublayers=include_sublayers
    )
    concept_eraser.fit(model=bert, data_loader=build_mnli_heuristic_loader())
    concept_eraser.activate_eraser()
