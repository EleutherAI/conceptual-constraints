# Online LEACE
 LEACE during model training

## Install requirements
Create a Python >= 3.10 environment and install the requirements:
```bash
pip install -e .
```

## Usage
One can finetune a BERT model on the MNLI dataset and evaluate the resulting model on the HANS dataset with the command:
```bash
python cmd/train.py
```
This model can be evaluated (with or without concept erasure) via the command:
```bash
python cmd/evaluate_model.py XYZ --concept-erasure ERASURE_METHOD
```

where `XYZ` is the ID of the model to be evaluated (assign in the `./results` directory) and `ERASURE_METHOD` is an optional argument assuming one of the following values:

| ERASURE_METHOD | Description |
| --- | --- |
| `leace-cls` | LEACE applied to the representation of each CLS token. |
| `leace-flatten` | LEACE with orth flag applied to sequence representation. |


## Contribute
Development dependencies can be installed with:
```bash
pip install -e .[dev]
```
   