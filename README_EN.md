# bert4ms
mindspore implementation of transformers

## Installation

### Install from source
```bash
git clone https://github.com/lvyufeng/bert4ms
python setup.py install
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

MindSpore has already provide the implementation of SOTA models in [ModelZoo](http://gitee.com/mindspore/models), but all checkpoints are trained from scratch which is not faithful. Since [Transformers](https://github.com/huggingface/transformers) has become a convenient toolkit to finish research or industry tasks, I develop this tool to transfer the checkpoint with code from [huggingface](https://huggingface.co/) to MindSpore. You can use it as same as Transformers to develop your own pretrained or finetuned models.
