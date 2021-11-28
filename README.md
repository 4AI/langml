LangML (**Lang**uage **M**ode**L**) is a Keras-based and TensorFlow-backend language model toolkit, which provides mainstream pre-trained language models, e.g., BERT/RoBERTa/ALBERT, and their downstream application models.


[![pypi](https://img.shields.io/pypi/v/langml?style=for-the-badge)](https://pypi.org/project/langml/) [![](https://img.shields.io/badge/tensorflow-1.14+,2.x-orange.svg?style=for-the-badge#from=url&id=tVzOp&margin=%5Bobject%20Object%5D&originHeight=28&originWidth=197&originalType=binary&ratio=1&status=done&style=none)](https://code.alipay.com/riskstorm/langml/blob/master/) [![](https://img.shields.io/badge/keras-2.3.1+-blue.svg?style=for-the-badge#from=url&id=AIJ4T&margin=%5Bobject%20Object%5D&originHeight=28&originWidth=132&originalType=binary&ratio=1&status=done&style=none)](https://code.alipay.com/riskstorm/langml/blob/master/)

# Outline
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Reference](#reference)


# Features
<a href='#features'></a>

- Common and widely-used Keras layers: CRF, Attentions, Transformer
- Pretrained Language Models: Bert, RoBERTa, ALBERT. Friendly designed interfaces and easy to implement downstream singleton, shared/unshared two-tower or multi-tower models.
- Tokenizers: WPTokenizer (wordpiece), SPTokenizer (sentencepiece)
- Baseline models: Text Classification, Named Entity Recognition. It's no need to write any code, and just need to preprocess the data into a specific format and use the "langml-cli" to train various baseline models.
- Prompt-Based Tuning: PTuning


# Installation
<a href='#installation'></a>

You can install or upgrade langml/langml-cli via the following command:
```bash
pip install -U langml
```

# Quick Start
<a href='#quick-start'></a>

## Finetune a model

```python
from langml import keras, L
from langml.plm import load_bert

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

To train a bert classifier, just one line:

```bash
$ langml-cli baseline clf bert --backbone bert --config_path /path/to/bert_config.json --ckpt_path /path/to/bert_model.ckpt --vocab_path /path/to/vocab.txt --train_path /path/to/train.jsonl --dev_path /path/to/dev.jsonl --save_dir model --verbose 2
```

## Prompt-Based Tuning

Use Ptuning for text classification:

```python
from langml.prompt import Template,  PTuniningPrompt, PTuningForClassification
from langml.tokenizer import WPTokenizer

vocab_path = '/path/to/vocab.txt'

tokenizer = WPTokenizer(vocab_path, lowercase=True)

# 1. Define a template
template = Template(
    #  must specify tokens that are defined in the vocabulary, and the mask token is required
    template=['it', 'was', '[MASK]', '.'],
    # must specify tokens that are defined in the vocabulary.
    label_tokens_map={
        'positive': ['good'],
        'negative': ['bad', 'terrible']
    },
    tokenizer=tokenizer
)

# 2. Define Prompt Model

bert_config_path = '/path/to/bert_config.json'
bert_ckpt_path = '/path/to/bert_model.ckpt'

prompt_model = PTuniningPrompt('bert', bert_config_path, bert_ckpt_path,
                               template, freeze_plm=False, learning_rate=5e-5, encoder='lstm')
prompt_classifier = PTuningForClassification(prompt_model, tokenizer)

# 3. Train and Infer

data = [('I do not like this food', 'negative'),
        ('I hate you', 'negative'),
        ('I like you', 'positive'),
        ('I like this food', 'positive')]

X = [d for d, _ in data]
y = [l for _, l in data]

prompt_classifier.fit(X, y, X, y, batch_size=2, epoch=50, model_path='best_model.weight')
# load pretrained model
# prompt_classifier.load('best_model.weight')
print("pred", prompt_classifier.predict('I hate you'))
```


# Documentation
<a href='#documentation'></a>

Please visit the [langml.readthedocs.io](https://langml.readthedocs.io/en/latest/index.html) to check the latest documentation.


# Reference
<a href='#reference'></a>

The implementation of pretrained language model is inspired by [CyberZHG/keras-bert](https://github.com/CyberZHG/keras-bert#Download-Pretrained-Checkpoints) and [bojone/bert4keras](https://github.com/bojone/bert4keras).
