# bert4ms
使用[MindSpore](https://www.mindspore.cn/)实现的另一版Transformers，兼容Huggingface的Transformers实现。针对MindSpore提供的自动并行特性，逐步提供相关能力，给使用MindSpore的同学们做科研提供便利。

## 安装

由于还在不断更新迭代，暂时不上传whl包。

### 源码安装

```bash
git clone https://github.com/lvyufeng/bert4ms
python setup.py install
```

## 快速入门

bert4ms提供类似于Transformers的编码体验，可以直接使用模型名一键加载，具体使用方式如下：

```python
from bert4ms import BertTokenizer, BertModel

tokenizer = BertTokenizer.load('bert-base-uncased')
model = BertModel.load('bert-base-uncased'， force_download=True)
model.set_train(False)

# get tokenized inputs
inputs = tokenizer("hello world")

# run model inference
outputs = model(inputs)
```

此外，模型直接兼容Huggingface提供的Checkpoint，故而可以选择直接下载Pytorch权重，一键转换并加载：

```python
from bert4ms import BertModel

model = BertModel.load('bert-base-uncased', from_torch=True)
```

## 为什么要做bert4ms?

- MindSpore在[ModelZoo](http://gitee.com/mindspore/models)提供了大量众智完成的SOTA模型，但是一方面实现上非常冗余（难看至极），且预训练模型支持的很少(而且ckpt都是自己训出来的，相信大家都会选择各家公司的原版)，在现阶段大量实验都需要预训练模型的情况下，很难找到直接的Baseline或者仿照着ModelZoo实现。

- MindSpore本身经过近两年的版本迭代，逐渐趋于完善，且自动并行特性相较于Megatron更简单易用。但是由于没有充分的大模型迁移和实现，仅有盘古一个模型可以作为案例支持，难以体现其优势。

- 个人有一篇压了好久的论文，需要Roberta、GPT等模型当做Baseline，既然入了MindSpore的坑还是想用MindSpore实现一下。

## Notice

本仓库中使用Heroku搭建了一个临时的checkpoint存储，提供公网下载MindSpore版的Checkpoint的能力。由于Heroku的限制，30分钟没有访问则会自动下线，有新访问时会重新启动，因此可能会有数分钟的启动时间，如果下载失败请重试，或将`force_download`设置为`True`。

此外，由于临时服务的不稳定性，**下载地址随时可能发生失效或更换**。若发生此类情况，请默认将`from_torch`设置为`True`。
