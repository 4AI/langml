Examples of finetuneing
================================
To finetune a model, you need to prepare pretrained language models (PLMs). Currently, LangML supports BERT/RoBERTa/ALBERT PLMs. You can download PLMs from `google-research/bert <https://github.com/google-research/bert>`_ , `google-research/albert <https://github.com/google-research/albert>`_ , `Chinese RoBERTa <https://github.com/ymcui/Chinese-BERT-wwm>`_ etc.




1. Prepare datasets
------------------------------------
You need to use specific tokenizers in terms of PLMs to initialize a tokenizer and convert texts to vocabulary indexes. LangML wraps `huggingface/tokenizers <https://github.com/huggingface/tokenizers>`_ and `google/sentencepiece <https://github.com/google/sentencepiece>`_ to provided a uniform interface. Specifically, you can initialize a wordpiece tokenizer via `langml.tokenizer.WPTokenizer`, and initialize a sentencepiece tokenizer via `langml.tokenizer.SPTokenizer`.


.. code-block:: python

   from langml import keras, L
   from langml.tokenizer import WPTokenizer


   vocab_path = '/path/to/vocab.txt'
   tokenizer = WPTokenizer(vocab_path)
   # specify max token length
   tokenizer.enable_trunction(max_length=512)


   class DataLoader:
      def __init__(self, tokenizer):
         # define initializer here
         self.tokenizer = tokenizer
      
      def __iter__(self, data):
         # define your data generator here
         for text, label in data:
            tokenized = self.tokenizer.encode(text)
            token_ids = tokenized.ids
            segment_ids = tokenized.segment_ids
            # ...


2. Build models
------------------------------------

You can use `langml.plm.load_bert` to load a BERT/RoBERTa model, and use `langml.plm.load_albert` to load an ALBERT model. 


.. code-block:: python

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


3. Train and Eval
------------------------------------

After defining the data loader and model, you can train and evaluate your model as most Keras models do.
