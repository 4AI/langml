<p align='center'><img src='docs/langml-logo.png' width=480 /></p>

LangML (**Lang**uage **M**ode**L**) is a Keras-based and TensorFlow-backend language model toolkit, which provides mainstream pre-trained language models, e.g., BERT/RoBERTa/ALBERT, and their downstream application models.


[![pypi](https://img.shields.io/pypi/v/langml?style=for-the-badge)](https://pypi.org/project/langml/) [![](https://img.shields.io/badge/tensorflow-1.14+,2.x-orange.svg?style=for-the-badge#from=url&id=tVzOp&margin=%5Bobject%20Object%5D&originHeight=28&originWidth=197&originalType=binary&ratio=1&status=done&style=none)](https://code.alipay.com/riskstorm/langml/blob/master/) [![](https://img.shields.io/badge/keras-2.3.1+-blue.svg?style=for-the-badge#from=url&id=AIJ4T&margin=%5Bobject%20Object%5D&originHeight=28&originWidth=132&originalType=binary&ratio=1&status=done&style=none)](https://code.alipay.com/riskstorm/langml/blob/master/)

# Outline
- [Outline](#outline)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Specify the Keras variant](#specify-the-keras-variant)
  - [Load pretrained language models](#load-pretrained-language-models)
  - [Finetune a model](#finetune-a-model)
  - [Use langml-cli to train baseline models](#use-langml-cli-to-train-baseline-models)
- [Documentation](#documentation)
- [Reference](#reference)


# Features
<a href='#features'></a>

- Common and widely-used Keras layers: CRF, Transformer, Attentions: Additive, ScaledDot, MultiHead, GatedAttentionUnit, and so on.
- Pretrained Language Models: BERT, RoBERTa, ALBERT. Providing friendly designed interfaces and easy to implement downstream singleton, shared/unshared two-tower or multi-tower models.
- Tokenizers: WPTokenizer (wordpiece), SPTokenizer (sentencepiece)
- Baseline models: Text Classification, Named Entity Recognition, Contrastive Learning. It's no need to write any code, and just need to preprocess the data into a specific format and use the "langml-cli" to train various baseline models.
- Prompt-Based Tuning: PTuning


# Installation
<a href='#installation'></a>

You can install or upgrade langml/langml-cli via the following command:

```bash
pip install -U langml
```

# Quick Start
<a href='#quick-start'></a>

## Specify the Keras variant

1) Use pure Keras (default setting)
   
```bash
export TF_KERAS=0
```

2) Use TensorFlow Keras

```bash
export TF_KERAS=1
```


## Load pretrained language models

```python
from langml import WPTokenizer, SPTokenizer
from langml import load_bert, load_albert

# load bert / roberta plm
bert_model, bert = load_bert(config_path, checkpoint_path)
# load albert plm
albert_model, albert = load_albert(config_path, checkpoint_path)
# load wordpiece tokenizer
wp_tokenizer = WPTokenizer(vocab_path, lowercase)
# load sentencepiece tokenizer
sp_tokenizer = SPTokenizer(vocab_path, lowercase)
```

## Finetune a model

```python
from langml import keras, L
from langml import load_bert

config_path = '/path/to/bert_config.json'
ckpt_path = '/path/to/bert_model.ckpt'
vocab_path = '/path/to/vocab.txt'

bert_model, bert_instance = load_bert(config_path, ckpt_path)
# get CLS representation
cls_output = L.Lambda(lambda x: x[:, 0])(bert_model.output)
output = L.Dense(2, activation='softmax',
                 kernel_intializer=bert_instance.initializer)(cls_output)
train_model = keras.Model(bert_model.input, cls_output)
train_model.summary()
train_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizer.Adam(1e-5))
```

## Use langml-cli to train baseline models

1) Text Classification

```bash
$ langml-cli baseline clf --help
Usage: langml baseline clf [OPTIONS] COMMAND [ARGS]...

  classification command line tools

Options:
  --help  Show this message and exit.

Commands:
  bert
  bilstm
  textcnn
```

2) Named Entity Recognition

```bash
$ langml-cli baseline ner --help
Usage: langml baseline ner [OPTIONS] COMMAND [ARGS]...

  ner command line tools

Options:
  --help  Show this message and exit.

Commands:
  bert-crf
  lstm-crf
```

3) Contrastive Learning

```bash
$ langml-cli baseline contrastive --help
Usage: langml baseline contrastive [OPTIONS] COMMAND [ARGS]...

  contrastive learning command line tools

Options:
  --help  Show this message and exit.

Commands:
  simcse
```

4) Text Matching

```bash
$ langml-cli baseline matching --help
Usage: langml baseline matching [OPTIONS] COMMAND [ARGS]...

  text matching command line tools

Options:
  --help  Show this message and exit.

Commands:
  sbert
```


# Documentation
<a href='#documentation'></a>

Please visit the [langml.readthedocs.io](https://langml.readthedocs.io/en/latest/index.html) to check the latest documentation.


# Reference
<a href='#reference'></a>

The implementation of pretrained language model is inspired by [CyberZHG/keras-bert](https://github.com/CyberZHG/keras-bert#Download-Pretrained-Checkpoints) and [bojone/bert4keras](https://github.com/bojone/bert4keras).
