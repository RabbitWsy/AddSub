# Description
A simple python implementation of transformer model from scratch using pytorch.

This project complete a task of addsub problem with constant input and output.

We provide a pretrained model in `ckpts` folder and the data used in train and test in `data/raw` folder.

# Installation
clone the repo:
```bash
git clone https://github.com/RabbitWsy/AddSub.git
```
create a virtual environment:
```bash
conda create -n addsub python=3.12
conda activate addsub
```
install the dependencies:
```bash
pip install -r requirements.txt
```
# Usage
## Data Preparation
The data is saved in json format. You can find the data in `data/raw` folder. If you want to add your own data, you shuold follow the format of `data/raw/train.json`, which is like:
```json
{
    "inp_expression": [
        "BOS POSITIVE 47 add NEGATIVE 27 EOS",
        // ...
    ],
    "out_expression": [
        "BOS POSITIVE 20 EOS",
        // ...
    ]
}
```
where `POSITIVE` and `NEGATIVE` are the positive and negative numbers, and `BOS` and `EOS` are the beginning of sentence and end of sentence tokens.

The answers in `"out_expression"` should be corresponding to the questions in `"inp_expression"`.

Note that the length of tokens of `"inp_expression"` and `"out_expression"` should be 9 and 5, where each digit is a token.

## Train and Test
```bash
python train.py \
    --config config/train.toml \
    --test \
    --output_dir ckpts
```
- `--test`: If you want to test the model, you should add this argument.
- `--output_dir`: Optional, ckpts is the default output directory.

Please refer to `config/train.toml` for more details. You can also change the config file to train the model with different hyperparameters.
## Inference
```bash
python inference.py \
    --config config/inference.toml \
    --question "BOS POSITIVE 47 add NEGATIVE 28 EOS"
```
- `--question`: Nessary, the question to be answered

## Visualize
to be updated
```bash
python visualize.py
```