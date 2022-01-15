Use langml-cli to quickly train baseline models
================================
You can use LangML-CLI to quickly train baseline models. You don't need to write any code, just need to prepare the dataset in a specific format.


You can train various baseline models using `langml-cli`:

.. code-block:: bash
    $ langml-cli --help
    Usage: langml [OPTIONS] COMMAND [ARGS]...

    LangML client

    Options:
    --version  Show the version and exit.
    --help     Show this message and exit.

    Commands:
    baseline  LangML Baseline client


Text Classification
------------------------------------

Prepare your data into `JSONLines` format, and provide text and label field in each line, for example:


.. code-block:: json

    {"text": "this is sentence1", "label": "label1"}
    {"text": "this is sentence2", "label": "label2"}


1. Bert

.. code-block:: bash

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


2. BiLSTM

.. code-block:: bash

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


3. TextCNN


.. code-block:: bash

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



Named Entity Recognition
------------------------------------

Prepare your data in the following format: 

use "\t" to separate entity segment and entity type in a sentence, and use "\n\n" to separate different sentences.

An English example:


.. code-block:: plaintext

    I like    O
    apples  Fruit

    I like    O
    pineapples  Fruit


A Chinese example:


.. code-block:: plaintext

    我来自  O
    中国    LOC

    我住在  O
    上海    LOC


1. BERT-CRF

.. code-block: bash

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


2. LSTM-CRF

.. code-block:: bash

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


Contrastive Learning
------------------------------------

Prepare your data into `JSONLines` format:

1) for evaulation, should include `text_left`, `text_right`, and `label` fields

.. code-block:: json

    {"text_left": "text left1", "text_right": "text right1", "label": "0/1"}
    {"text_left": "text left1", "text_right": "text right2", "label": "0/1"}


2) no need to evaluate, just provide `text` field.

.. code-block:: json

    {"text": "this is a text1"}
    {"text": "this is a text2"}


1. simcse

.. code-block:: bash

    $ langml-cli baseline contrastive simcse --help
    Usage: langml baseline contrastive simcse [OPTIONS]

    Options:
        --backbone TEXT              specify backbone: bert | roberta | albert
        --epoch INTEGER              epochs
        --batch_size INTEGER         batch size
        --learning_rate FLOAT        learning rate
        --dropout_rate FLOAT         dropout rate
        --temperature FLOAT          temperature
        --pooling_strategy TEXT      specify pooling_strategy from ["cls", "first-
                                    last-avg", "last-avg"]

        --max_len INTEGER            max len
        --early_stop INTEGER         patience of early stop
        --monitor TEXT               metrics monitor
        --lowercase                  do lowercase
        --tokenizer_type TEXT        specify tokenizer type from [`wordpiece`,
                                    `sentencepiece`]

        --config_path TEXT           bert config path  [required]
        --ckpt_path TEXT             bert checkpoint path  [required]
        --vocab_path TEXT            bert vocabulary path  [required]
        --train_path TEXT            train path  [required]
        --test_path TEXT             test path
        --save_dir TEXT              dir to save model  [required]
        --verbose INTEGER            0 = silent, 1 = progress bar, 2 = one line per
                                    epoch

        --apply_aeda                 apply AEDA to augment data
        --aeda_language TEXT         specify AEDA language, ["EN", "CN"]
        --do_evaluate                do evaluation
        --distributed_training       distributed training
        --distributed_strategy TEXT  distributed training strategy
        --help                       Show this message and exit.
