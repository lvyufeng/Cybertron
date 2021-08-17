# bert4ms
mindspore implementation of transformers

## Installation

```bash
pip install bert4ms
```

## Quick Start

```python
from bert4ms import BertTokenizer, BertModel
from bert4ms import compile_model

tokenizer = BertTokenizer.load('bert-base-uncased')
model = BertModel.load('bert-base-uncased')

# get tokenized inputs
inputs = tokenizer("hello world")

# compile model
compile_model(model, inputs)

# run model inference
outputs = model(inputs)
```

## Why bert4ms?

