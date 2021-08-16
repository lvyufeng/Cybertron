# bert4ms
mindspore implementation of transformers

## Installation

```bash
pip install bert4ms
```

## Quick Start

```python
>>> from bert4ms import BertTokenizer, BertModel
>>> tokenizer = BertTokenizer.load('bert-base-uncased')
>>> model = BertModel.load('bert-base-uncased')
>>> inputs = tokenizer("hello world")
>>> outputs = model(**inputs)
```

## Why bert4ms?

