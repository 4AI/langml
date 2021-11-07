LangML (**Lang**uage **M**ode**L**) is a Keras-based and TensorFlow-backend language model toolkit, which provides mainstream pre-trained language models, e.g., BERT/RoBERTa/ALBERT, and their downstream application models.


[![](https://img.shields.io/badge/tensorflow-1.14+,2.x-orange.svg?style=for-the-badge#from=url&id=tVzOp&margin=%5Bobject%20Object%5D&originHeight=28&originWidth=197&originalType=binary&ratio=1&status=done&style=none)](https://code.alipay.com/riskstorm/langml/blob/master/) [![](https://img.shields.io/badge/keras-2.3.1+-blue.svg?style=for-the-badge#from=url&id=AIJ4T&margin=%5Bobject%20Object%5D&originHeight=28&originWidth=132&originalType=binary&ratio=1&status=done&style=none)](https://code.alipay.com/riskstorm/langml/blob/master/)

# Features

- Common and widely-used Keras layers: CRF, Attentions, Transformer
- Pretrained Language Models: Bert, RoBERTa, ALBERT. Friendly designed interfaces and easy to implement downstream singleton, shared/unshared two-tower or multi-tower models.
- Tokenizers: WPTokenizer (wordpiece), SPTokenizer (sentencepiece)
- Baseline models: Text Classification, Named Entity Recognition. It's no need to write any code to train the baselines. You just need to preprocess the data into a specific format and use the "langml-cli" to train the model.





# Installation


You can install or upgrade langml/langml-cli via the following command:
```bash
pip install -U langml
```


# Documents
##
## Keras Variants


LangML supports keras and tf.keras. You can configure environment variables to set specific Keras variant.


`export TF_KERAS=0`  # use keras

`export TF_KERAS=1`  # use tf.keras


## NLP Baseline Models


You can train various baseline models use "langml-cli".


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
// TODO




### Named Entity Recognition
// TODO



## Pretrained Languange Models


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


**1. finetune a model**
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


**2. finetune a model under distributed training.**
****

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


**3. continue to pretrain a language model**
****

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
****

**4. finetune a two-tower model with shared weights**
****

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
****

**5. finetune a two-tower model with unshared weights**
****

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
****

## Tokenizers
// TODO



## Keras Layers


#### langml.layers.CRF(output_dim: int, sparse_target: bool = True, **kwargs)


Args:
  - output_dim: output dimension, int. It's usually equal to the tag size.
  - sparse_target: set sparse_target, bool. If the target is prepared as one-hot encoding, set this argument as `True`.


Return:
  - Tensor


**Examples:**


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


#### langml.layers.SelfAttention(attention_units: Optional[int] = None, return_attention: bool = False, is_residual: bool = False, attention_activation: Activation = 'relu', attention_epsilon: float = 1e10, kernel_initializer: Initializer = 'glorot_normal', kernel_regularizer: Optional[Regularizer] = None, kernel_constraint: Optional[Constraint] = None, bias_initializer: Union[Initializer, str] = 'zeros', bias_regularizer: Optional[Regularizer] = None, bias_constraint: Optional[Constraint] = None, use_attention_bias: bool = True, attention_penalty_weight: float = 0.0, **kwargs)


**Examples:**


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


#### langml.layers.SelfAdditiveAttention(attention_units: Optional[int] = None, return_attention: bool = False, is_residual: bool = False, attention_activation: Activation = 'relu', attention_epsilon: float = 1e10, kernel_initializer: Initializer = 'glorot_normal', kernel_regularizer: Optional[Regularizer] = None, kernel_constraint: Optional[Constraint] = None, bias_initializer: Initializer = 'zeros', bias_regularizer: Optional[Regularizer] = None, bias_constraint: Optional[Constraint] = None, use_attention_bias: bool = True, attention_penalty_weight: float = 0.0, **kwargs)


#### langml.layers.ScaledDotProductAttention(return_attention: bool = False, history_only: bool = False, **kwargs)


#### langml.layers.MultiHeadAttention(head_num: int, return_attention: bool = False, attention_activation: Activation = 'relu', kernel_initializer: Initializer = 'glorot_normal', kernel_regularizer: Optional[Regularizer] = None, kernel_constraint: Optional[Constraint] = None, bias_initializer: Initializer = 'zeros', bias_regularizer: Optional[Regularizer] = None, bias_constraint: Optional[Constraint] = None, use_attention_bias: Optional[bool] = True, **kwargs)


#### langml.layers.LayerNorm(center: bool = True, scale: bool = True, epsilon: float = 1e-7, gamma_initializer: Initializer = 'ones', gamma_regularizer: Optional[Regularizer] = None, gamma_constraint: Optional[Constraint] = None, beta_initializer: Initializer = 'zeros', beta_regularizer: Optional[Regularizer] = None, beta_constraint: Optional[Constraint] = None, **kwargs)




## Save Model
#### langml.model.save_frozen(model: Models, fpath: str)


freeze model to tensorflow pb.


# Reference
This project is inspired by [CyberZHG/keras-bert](https://github.com/CyberZHG/keras-bert#Download-Pretrained-Checkpoints) and [bojone/bert4keras](https://github.com/bojone/bert4keras).
