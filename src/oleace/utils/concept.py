# The concept assignment for the HANS heuristics has been adapted from
# https://github.com/tommccoy1/hans/blob/master/heuristic_finder_scripts/const_finder.py

from datasets import load_dataset
from tqdm import tqdm


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

    return hyp_filtered in all_phrases


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

    return hyp_filtered in prem_filtered


if __name__ == "__main__":

    train_dataset = load_dataset("multi_nli", split="train")
    consituent_count = 0
    lexical_overlap_count = 0
    subsequence_count = 0
    for example in tqdm(train_dataset):
        if is_constituent(example):
            consituent_count += 1
        if is_lexical_overlap(example):
            lexical_overlap_count += 1
        if is_subsequence(example):
            subsequence_count += 1
    print(consituent_count / len(train_dataset))
    print(lexical_overlap_count / len(train_dataset))
    print(subsequence_count / len(train_dataset))
