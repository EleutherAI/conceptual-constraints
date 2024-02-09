# Online LEACE
This repo tests an online version of [LEACE](https://github.com/EleutherAI/concept-erasure/tree/main) to erase spurious heuristic concepts from [BERT](https://huggingface.co/docs/transformers/model_doc/bert) representations *during finetuning* on the [MNLI dataset](https://huggingface.co/datasets/multi_nli). All the models are evaluated on the [HANS dataset](https://github.com/tommccoy1/hans), which evaluates the robustness of a model with respect to these spurious heuristics.

## Install requirements
Create a Python >= 3.10 environment and install the requirements:
```bash
pip install -e .
```

## Usage
One can finetune a BERT model on the MNLI dataset and evaluate the resulting model on the HANS dataset with the command:
```bash
python cmd/train.py --concept-erasure ERASURE_METHOD --include_sublayers
```
where `--concept_erasure` and `--include_sublayers` are optional flags controlling the concept erasure technique used, ERASURE_METHOD` is an optional argument assuming one of the following values:
| ERASURE_METHOD | Description |
| --- | --- |
| `leace-cls` | LEACE applied to the representation of each CLS token. |
| `leace-flatten` | LEACE with orth flag applied to sequence representation. |

This model can be evaluated (with or without concept erasure) via the command:
```bash
python cmd/evaluate_model.py XYZ --concept-erasure ERASURE_METHOD --include_sublayers
```

where `XYZ` is the ID of the model to be evaluated (assign in the `./results` directory) , where `--concept_erasure` and `--include_sublayers` are optional flags and `ERASURE_METHOD` is an optional argument assuming one of the above values.




## Contribute
Development dependencies can be installed with:
```bash
pip install -e .[dev]
```
