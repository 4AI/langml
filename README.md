LangML (**Lang**uage **M**ode**L**) is a Keras-based and TensorFlow-backend language model toolkit, which provides mainstream pre-trained language models, e.g., BERT/RoBERTa/ALBERT, and their downstream application models.


[![](https://img.shields.io/badge/tensorflow-1.14+,2.x-orange.svg?style=for-the-badge#from=url&id=tVzOp&margin=%5Bobject%20Object%5D&originHeight=28&originWidth=197&originalType=binary&ratio=1&status=done&style=none)](https://code.alipay.com/riskstorm/langml/blob/master/) [![](https://img.shields.io/badge/keras-2.3.1+-blue.svg?style=for-the-badge#from=url&id=AIJ4T&margin=%5Bobject%20Object%5D&originHeight=28&originWidth=132&originalType=binary&ratio=1&status=done&style=none)](https://code.alipay.com/riskstorm/langml/blob/master/)

# Outline
- [Features](#features)
- [Installation](#installation)
- [Documents](#documents)
  - [Keras Variants](#keras-variants)
  - [NLP Baseline Models](#nlp-baseline-models)
    - [Text Classification](#text-classification)
    - [Named Entity Recognition](#named-entity-recognition)
  - [Pretrained Language Models](#pretrained-language-models)
  - [Tokenizers](#tokenizers)
  - [Keras Layers](#keras-layers)
  - [Save Model](#save-model)
- [Reference](#reference)


# Features
<a href='#features'></a>

- Common and widely-used Keras layers: CRF, Attentions, Transformer
- Pretrained Language Models: Bert, RoBERTa, ALBERT. Friendly designed interfaces and easy to implement downstream singleton, shared/unshared two-tower or multi-tower models.
- Tokenizers: WPTokenizer (wordpiece), SPTokenizer (sentencepiece)
- Baseline models: Text Classification, Named Entity Recognition. It's no need to write any code to train the baselines. You just need to preprocess the data into a specific format and use the "langml-cli" to train the model.



# Installation
<a href='#installation'></a>

You can install or upgrade langml/langml-cli via the following command:
```bash
pip install -U langml
```


# Documents
<a href='#documents'></a>

## Keras Variants
<a href='#keras-variants'></a>


LangML supports keras and tf.keras. You can configure environment variables to set specific Keras variant.


`export TF_KERAS=0`  # use keras

`export TF_KERAS=1`  # use tf.keras


## NLP Baseline Models
<a href='#nlp-baseline-models'></a>

You can train various baseline models using "langml-cli".


Usage:
```bash
$ langml-cli --help
Usage: langml [OPTIONS] COMMAND [ARGS]...

  LangML client

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  baseline  LangML Baseline client
```


### Text Classification
<a href='#text-classification'></a>

Please prepare your data into JSONLines format, and provide `text` and `label` field in each line, for example:

```json
{"text": "this is sentence1", "label": "label1"}
{"text": "this is sentence2", "label": "label2"}
```

#### Bert

```bash
$ langml-cli baseline clf bert --help
Usage: langml baseline clf bert [OPTIONS]

Options:
  --backbone TEXT              specify backbone: bert | roberta | albert
  --epoch INTEGER              epochs
  --batch_size INTEGER         batch size
  --learning_rate FLOAT        learning rate
  --max_len INTEGER            max len
  --lowercase                  do lowercase
  --tokenizer_type TEXT        specify tokenizer type from [`wordpiece`,
                               `sentencepiece`]

  --monitor TEXT               monitor for keras callback
  --early_stop INTEGER         patience to early stop
  --use_micro                  whether to use micro metrics
  --config_path TEXT           bert config path  [required]
  --ckpt_path TEXT             bert checkpoint path  [required]
  --vocab_path TEXT            bert vocabulary path  [required]
  --train_path TEXT            train path  [required]
  --dev_path TEXT              dev path  [required]
  --test_path TEXT             test path
  --save_dir TEXT              dir to save model  [required]
  --verbose INTEGER            0 = silent, 1 = progress bar, 2 = one line per
                               epoch

  --distributed_training       distributed training
  --distributed_strategy TEXT  distributed training strategy
  --help                       Show this message and exit.
```

#### BiLSTM

```bash
$ langml-cli baseline clf bilstm --help
Usage: langml baseline clf bilstm [OPTIONS]

Options:
  --epoch INTEGER              epochs
  --batch_size INTEGER         batch size
  --learning_rate FLOAT        learning rate
  --embedding_size INTEGER     embedding size
  --hidden_size INTEGER        hidden size of lstm
  --max_len INTEGER            max len
  --lowercase                  do lowercase
  --tokenizer_type TEXT        specify tokenizer type from [`wordpiece`,
                               `sentencepiece`]

  --monitor TEXT               monitor for keras callback
  --early_stop INTEGER         patience to early stop
  --use_micro                  whether to use micro metrics
  --vocab_path TEXT            vocabulary path  [required]
  --train_path TEXT            train path  [required]
  --dev_path TEXT              dev path  [required]
  --test_path TEXT             test path
  --save_dir TEXT              dir to save model  [required]
  --verbose INTEGER            0 = silent, 1 = progress bar, 2 = one line per
                               epoch

  --with_attention             apply attention mechanism
  --distributed_training       distributed training
  --distributed_strategy TEXT  distributed training strategy
  --help                       Show this message and exit.
```

#### TextCNN

```bash
$ langml-cli baseline clf textcnn --help
Usage: langml baseline clf textcnn [OPTIONS]

Options:
  --epoch INTEGER              epochs
  --batch_size INTEGER         batch size
  --learning_rate FLOAT        learning rate
  --embedding_size INTEGER     embedding size
  --filter_size INTEGER        filter size of convolution
  --max_len INTEGER            max len
  --lowercase                  do lowercase
  --tokenizer_type TEXT        specify tokenizer type from [`wordpiece`,
                               `sentencepiece`]

  --monitor TEXT               monitor for keras callback
  --early_stop INTEGER         patience to early stop
  --use_micro                  whether to use micro metrics
  --vocab_path TEXT            vocabulary path  [required]
  --train_path TEXT            train path  [required]
  --dev_path TEXT              dev path  [required]
  --test_path TEXT             test path
  --save_dir TEXT              dir to save model  [required]
  --verbose INTEGER            0 = silent, 1 = progress bar, 2 = one line per
                               epoch

  --distributed_training       distributed training
  --distributed_strategy TEXT  distributed training strategy
  --help                       Show this message and exit.
```

### Named Entity Recognition
<a href='#named-entity-recognition'></a>

Please prepare your data in the following format: use `\t` to separate entity segment and entity type in a sentence, and use `\n\n` to separate different sentences.


An english example:

```
I like    O
apples  Fruit

I like    O
pineapples  Fruit
``` 

A chinese example:

```
我来自  O
中国    LOC

我住在  O
上海    LOC
```

#### Bert-CRF

```bash
$ langml-cli baseline ner bert-crf --help
Usage: langml baseline ner bert-crf [OPTIONS]

Options:
  --backbone TEXT              specify backbone: bert | roberta | albert
  --epoch INTEGER              epochs
  --batch_size INTEGER         batch size
  --learning_rate FLOAT        learning rate
  --dropout_rate FLOAT         dropout rate
  --max_len INTEGER            max len
  --lowercase                  do lowercase
  --tokenizer_type TEXT        specify tokenizer type from [`wordpiece`,
                               `sentencepiece`]

  --config_path TEXT           bert config path  [required]
  --ckpt_path TEXT             bert checkpoint path  [required]
  --vocab_path TEXT            bert vocabulary path  [required]
  --train_path TEXT            train path  [required]
  --dev_path TEXT              dev path  [required]
  --test_path TEXT             test path
  --save_dir TEXT              dir to save model  [required]
  --monitor TEXT               monitor for keras callback
  --early_stop INTEGER         patience to early stop
  --verbose INTEGER            0 = silent, 1 = progress bar, 2 = one line per
                               epoch

  --distributed_training       distributed training
  --distributed_strategy TEXT  distributed training strategy
  --help                       Show this message and exit.
```

#### LSTM-CRF

```bash
$  langml-cli baseline ner lstm-crf --help
Usage: langml baseline ner lstm-crf [OPTIONS]

Options:
  --epoch INTEGER              epochs
  --batch_size INTEGER         batch size
  --learning_rate FLOAT        learning rate
  --dropout_rate FLOAT         dropout rate
  --embedding_size INTEGER     embedding size
  --hidden_size INTEGER        hidden size
  --max_len INTEGER            max len
  --lowercase                  do lowercase
  --tokenizer_type TEXT        specify tokenizer type from [`wordpiece`,
                               `sentencepiece`]

  --vocab_path TEXT            vocabulary path  [required]
  --train_path TEXT            train path  [required]
  --dev_path TEXT              dev path  [required]
  --test_path TEXT             test path
  --save_dir TEXT              dir to save model  [required]
  --monitor TEXT               monitor for keras callback
  --early_stop INTEGER         patience to early stop
  --verbose INTEGER            0 = silent, 1 = progress bar, 2 = one line per
                               epoch

  --distributed_training       distributed training
  --distributed_strategy TEXT  distributed training strategy
  --help                       Show this message and exit.
```

## Pretrained Language Models
<a href='#pretrained-language-models'></a>

#### langml.plm.load_albert(config_path: str, checkpoint_path: str, seq_len: Optional[int] = None, pretraining: bool = False, with_mlm: bool = True, with_nsp: bool = True, lazy_restore: bool = False, weight_prefix: Optional[str] = None, dropout_rate: float = 0.0, **kwargs) -> Union[Tuple[Models, Callable], Tuple[Models, Callable, Callable]]: 🔗


load and restore ALBERT model.


Args:
  - config_path: configure path, str.
  - checkpoint_path: checkpoint path, str,
  - seq_len: sequence length, int.
  - pretraining: pretraining mode, bool. If you want to continue pretraining a language model, set it True
  - with_mlm: use Mask Language Model task, bool. This argument works when pretraining=True.
  - with_nsp: apply Next Sentence Prediction task, bool. This argument works when pretraining=True.
  - lazy_restore: lazy restore pretrained model weights. When applying distributed training strategy, set it as True, and it will return one more callback function.
  - weight_prefix: add prefix name to weights, Optional[str]. For an unshared two-tower / multi-tower model, you can set the different prefixes to different towers.
  - dropout_rate: dropout rate, float.


Return:
  - model: an instance of keras.Model
  - bert: an instance of BERT
  - restore_weight_callback: a callback function to restore model weights. This callback function returns when lazy_restore=True.


**Examples: refer to **[load_bert examples](#jsCdZ)


#### langml.plm.load_bert(config_path: str, checkpoint_path: str, seq_len: Optional[int] = None, pretraining: bool = False, with_mlm: bool = True, with_nsp: bool = True, lazy_restore: bool = False, weight_prefix: Optional[str] = None, dropout_rate: float = 0.0, **kwargs) -> Union[Tuple[Models, Callable], Tuple[Models, Callable, Callable]]


load and restore BERT/RoBERTa model.


Args:
  - config_path: configure path, str.
  - checkpoint_path: checkpoint path, str,
  - seq_len: sequence length, int.
  - pretraining: pretraining mode, bool. If you want to continue pretraining a language model, set it True
  - with_mlm: use Mask Language Model task, bool. This argument works when pretraining=True.
  - with_nsp: apply Next Sentence Prediction task, bool. This argument works when pretraining=True.
  - lazy_restore: lazy restore pretrained model weights. When applying distributed training strategy, set it as True, and it will return one more callback function.
  - weight_prefix: add prefix name to weights, Optional[str]. For an unshared two-tower / multi-tower model, you can set the different prefixes to different towers.
  - dropout_rate: dropout rate, float.


Return:
  - model: an instance of keras.Model
  - bert: an instance of BERT
  - restore_weight_callback: a callback function to restore model weights. This callback function returns when lazy_restore=True.


**Examples:**

<details>
<summary>1. finetune a model (click to expand...)</summary>

```python
from langml.plm import load_bert


bert_model, bert = load_bert(
    config_path = '/path/to/bert_config.json',
    checkpoint_path = '/path/to/bert_model.ckpt'
)

CLS = L.Lambda(lambda x: x[:, 0])(bert_model.output)
output = L.Dense(num_labels,
                 initializer=bert.initializer,
                 activation='softmax')(CLS)

train_model = keras.Model(bert_model.input, output)
train_model.summary()
train_model.compile(keras.optimizers.Adam(1e-5),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
```
</details>

<details>
<summary>2. finetune a model under distributed training (click to expand...)</summary>

```python
from langml.plm import load_bert


strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    bert_model, bert, restore_weight_callback = load_bert(
        config_path = '/path/to/bert_config.json',
        checkpoint_path = '/path/to/bert_model.ckpt',
        lazy_restore=True
    )

    CLS = L.Lambda(lambda x: x[:, 0])(bert_model.output)
    output = L.Dense(num_labels,
                     initializer=bert.initializer,
                     activation='softmax')(CLS)

    train_model = keras.Model(bert_model.input, output)
    train_model.summary()
    train_model.compile(keras.optimizers.Adam(1e-5),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
	# restore weights after compile
    restore_weight_callback(bert_model)
```
</details>

<details>
<summary>3. continue to pretrain a language model(click to expand...)</summary>

```python
from langml.plm import load_bert


bert_model, bert = load_bert(
    config_path = '/path/to/bert_config.json',
    checkpoint_path = '/path/to/bert_model.ckpt',
    pretraning=True,
    dropout_rate=0.2
)

model.summary()
model.compile(keras.optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```
</details>

<details>
<summary>4. finetune a two-tower model with shared weights (click to expand...)</summary>

```python
from langml.plm import load_bert


# left tower
# use the default input placeholder
bert_model, bert = load_bert(
    config_path = '/path/to/bert_config.json',
    checkpoint_path = '/path/to/bert_model.ckpt',
)
# CLS representation
left_output = L.Lambda(lambda x: x[:, 0])(bert_model.ouput)

# right tower
# inputs of right tower
right_token_in = L.Input(shape=(None, ), name='Right-Input-Token')
right_segment_in = L.Input(shape=(None, ), name='Right-Input-Segment')

# outputs of right tower
right_output = bert(inputs=[right_token_in, right_segment_in], return_model=False)
right_output = L.Lambda(lambda x: x[:, 0])(right_output)

# matching operation
matching = L.Lambda(your_matching_layer)([left_output, right_output])

# output
output = L.Dense(num_labels)(matching)
train_model = Model(inputs=(*bert_model.input, right_token_in, right_segment_in),
                    outpus=[output])

train_model.summary()
train_model.compile(keras.optimizers.Adam(1e-5),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
```
</details>


<details>
<summary>5. finetune a two-tower model with unshared weights  (click to expand...)</summary>

```python
from langml.plm import load_bert


# left tower
left_bert_model, _ = load_bert(
    config_path = '/path/to/bert_config.json',
    checkpoint_path = '/path/to/bert_model.ckpt',
    weight_prefix = 'Left'
)
# CLS representation
left_output = L.Lambda(lambda x: x[:, 0])(left_bert_model.ouput)

# right tower
right_bert_model, _ = load_bert(
    config_path = '/path/to/bert_config.json',
    checkpoint_path = '/path/to/bert_model.ckpt',
    weight_prefix = 'Right'
)
# CLS representation
right_output = L.Lambda(lambda x: x[:, 0])(right_bert_model.ouput)

# matching operation
matching = L.Lambda(your_matching_layer)([left_output, right_output])

# output
output = L.Dense(num_labels)(matching)
train_model = Model(inputs=(*bert_model.input, right_token_in, right_segment_in),
                    outpus=[output])

train_model.summary()
train_model.compile(keras.optimizers.Adam(1e-5),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
```
</details>

## Tokenizers
<a href='#tokenizers'></a>

#### langml.tokenizer.WPTokenizer(vocab_path: str, lowercase: bool = False)

Load WordPiece Tokenizer

<details>
<summary>Examples: (click to expand...)</summary>

```python
from langml.tokenizer import WPTokenizer

tokenizer = WPTokenizer('/path/to/vocab.txt')

text = 'hello world'
tokenized = tokenizer.encode(text)

print("token_ids:", tokenized.ids)
print("segment_ids:", tokenized.segment_ids)
```
</details>

#### langml.tokenizer.SPTokenizer

Load Sentencepiece Tokenizer

<details>
<summary>Examples: (click to expand...)</summary>

```python
from langml.tokenizer import SPTokenizer

tokenizer = SPTokenizer('/path/to/vocab.model')

text = 'hello world'
tokenized = tokenizer.encode(text)

print("token_ids:", tokenized.ids)
print("segment_ids:", tokenized.segment_ids)
```
</details>

## Keras Layers
<a href='#keras-layers'></a>

#### langml.layers.CRF(output_dim: int, sparse_target: bool = True, **kwargs)

Args:
  - output_dim: output dimension, int. It's usually equal to the tag size.
  - sparse_target: set sparse_target, bool. If the target is prepared as one-hot encoding, set this argument as `True`.


Return:
  - Tensor


**Examples:**

<details>
<summary>click to expand</summary>

```python
import keras
import keras.layers as L
from langml.layers import CRF


num_labels = 10
embedding_size = 100
hidden_size = 128

# define a CRF layer
crf = CRF(num_labels)

model = keras.Sequential()
model.add(L.Embedding(num_labels, embedding_size))
model.add(L.LSTM(hidden_size, return_sequences=True))
model.add(L.Dense(num_labels))
model.add(crf)
model.summary()
model.compile('adam', loss=crf.loss, metrics=[crf.accuracy])
```

</details>

#### langml.layers.SelfAttention(attention_units: Optional[int] = None, return_attention: bool = False, is_residual: bool = False, attention_activation: Activation = 'relu', attention_epsilon: float = 1e10, kernel_initializer: Initializer = 'glorot_normal', kernel_regularizer: Optional[Regularizer] = None, kernel_constraint: Optional[Constraint] = None, bias_initializer: Union[Initializer, str] = 'zeros', bias_regularizer: Optional[Regularizer] = None, bias_constraint: Optional[Constraint] = None, use_attention_bias: bool = True, attention_penalty_weight: float = 0.0, **kwargs)


**Examples:**


<details>
<summary>click to expand</summary>

```python
import keras
import keras.layers as L
from langml.layers import SelfAttention


model = keras.Sequential()
model.add(L.Embedding(num_labels, embedding_size))
model.add(L.LSTM(hidden_size, return_sequences=True))
model.add(SelfAttention())
model.add(L.Dense(num_labels))
model.summary()
model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

</details>


#### langml.layers.SelfAdditiveAttention(attention_units: Optional[int] = None, return_attention: bool = False, is_residual: bool = False, attention_activation: Activation = 'relu', attention_epsilon: float = 1e10, kernel_initializer: Initializer = 'glorot_normal', kernel_regularizer: Optional[Regularizer] = None, kernel_constraint: Optional[Constraint] = None, bias_initializer: Initializer = 'zeros', bias_regularizer: Optional[Regularizer] = None, bias_constraint: Optional[Constraint] = None, use_attention_bias: bool = True, attention_penalty_weight: float = 0.0, **kwargs)


#### langml.layers.ScaledDotProductAttention(return_attention: bool = False, history_only: bool = False, **kwargs)


#### langml.layers.MultiHeadAttention(head_num: int, return_attention: bool = False, attention_activation: Activation = 'relu', kernel_initializer: Initializer = 'glorot_normal', kernel_regularizer: Optional[Regularizer] = None, kernel_constraint: Optional[Constraint] = None, bias_initializer: Initializer = 'zeros', bias_regularizer: Optional[Regularizer] = None, bias_constraint: Optional[Constraint] = None, use_attention_bias: Optional[bool] = True, **kwargs)


#### langml.layers.LayerNorm(center: bool = True, scale: bool = True, epsilon: float = 1e-7, gamma_initializer: Initializer = 'ones', gamma_regularizer: Optional[Regularizer] = None, gamma_constraint: Optional[Constraint] = None, beta_initializer: Initializer = 'zeros', beta_regularizer: Optional[Regularizer] = None, beta_constraint: Optional[Constraint] = None, **kwargs)




## Save Model
<a href='#save-model'></a>

#### langml.model.save_frozen(model: Models, fpath: str)

freeze model to tensorflow pb.


# Reference
<a href='#reference'></a>

The implementation of pretrained language model is inspired by [CyberZHG/keras-bert](https://github.com/CyberZHG/keras-bert#Download-Pretrained-Checkpoints) and [bojone/bert4keras](https://github.com/bojone/bert4keras).
