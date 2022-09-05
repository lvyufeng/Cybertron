# Cybertron

[![Run tests](https://github.com/lvyufeng/cybertron/actions/workflows/run_tests.yml/badge.svg)](https://github.com/lvyufeng/cybertron/actions/workflows/run_tests.yml)


使用[MindSpore](https://www.mindspore.cn/)实现的预训练模型框架，支持各类Transformers，兼容Huggingface的Transformers实现。针对MindSpore提供的自动并行特性，逐步提供相关能力，给使用MindSpore的同学们做科研提供便利。

## 安装

由于还在不断更新迭代，暂时不上传whl包。

### 源码安装

```bash
# From Github(outside GFW)
pip install git+https://github.com/lvyufeng/cybertron@ms1.7
# From OpenI(inside GFW)
pip install git+https://git.openi.org.cn/lvyufeng/cybertron@ms1.7
```

## 快速入门

Cybertron提供类似于Transformers的编码体验，可以直接使用模型名一键加载，具体使用方式如下：

```python
import mindspore
from cybertron import BertTokenizer, BertModel

tokenizer = BertTokenizer.load('bert-base-uncased')
model = BertModel.load('bert-base-uncased')
model.set_train(False)

# get tokenized inputs
inputs = mindspore.Tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)], mindspore.int32)
# run model inference
outputs = model(inputs)
```

此外，模型直接兼容Huggingface提供的Checkpoint，故而可以选择直接下载Pytorch权重，一键转换并加载：

```python
from cybertron import BertModel

model = BertModel.load('bert-base-uncased', from_torch=True)
```

## 什么是Cybertron？

塞伯坦(Cybertron)是变形金刚种族的母星，是一个和地球近邻土星体积近似的巨大金属行星。Cybertron框架致力于提供各类Transformers模型及其变体，以及各类预训练模型的创新算法应用，成为Transformers模型的“母星”。


## Notice

所有的Checkpoint将会放到[Huggingface](https://huggingface.co/lvyufeng)，解决下载问题。