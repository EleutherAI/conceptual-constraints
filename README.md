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

## Contribute
Development dependencies can be installed with:
```bash
pip install -e .[dev]
```
   