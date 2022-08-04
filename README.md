# bert4ms
使用[MindSpore](https://www.mindspore.cn/)实现的另一版Transformers，兼容Huggingface的Transformers实现。针对MindSpore提供的自动并行特性，逐步提供相关能力，给使用MindSpore的同学们做科研提供便利。

## 安装

由于还在不断更新迭代，暂时不上传whl包。

### 源码安装

```bash
# From Github(outside GFW)
pip install git+https://github.com/lvyufeng/bert4ms
# From OpenI(inside GFW)
pip install git+https://git.openi.org.cn/lvyufeng/bert4ms
```

## 快速入门

bert4ms提供类似于Transformers的编码体验，可以直接使用模型名一键加载，具体使用方式如下：

```python
import mindspore
from bert4ms import BertTokenizer, BertModel

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
from bert4ms import BertModel

model = BertModel.load('bert-base-uncased', from_torch=True)
```

## 为什么要做bert4ms?

- MindSpore在[ModelZoo](http://gitee.com/mindspore/models)提供了大量众智完成的SOTA模型，但是一方面实现上非常冗余（难看至极），且预训练模型支持的很少(而且ckpt都是自己训出来的，相信大家都会选择各家公司的原版)，在现阶段大量实验都需要预训练模型的情况下，很难找到直接的Baseline或者仿照着ModelZoo实现。

- MindSpore本身经过近两年的版本迭代，逐渐趋于完善，且自动并行特性相较于Megatron更简单易用。但是由于没有充分的大模型迁移和实现，仅有盘古一个模型可以作为案例支持，难以体现其优势。

- 个人有一篇压了好久的论文，需要Roberta、GPT等模型当做Baseline，既然入了MindSpore的坑还是想用MindSpore实现一下。

## Notice

所有的Checkpoint将会放到[Huggingface](https://huggingface.co/lvyufeng)，解决下载问题。