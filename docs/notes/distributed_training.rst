How to train PLMs distributedly?
================================
To train distributedly, you need to use `tensorflow.keras`. First, you need to define an environment variable `TF_KERAS` and assign `1` to it, for example,  `export TF_KERAS=1` for Linux. Then manually restore PLMs weights after model compiling, as follows:


.. code-block:: python

   from langml import keras, L
   from langml.plm import load_bert

   config_path = '/path/to/bert_config.json'
   ckpt_path = '/path/to/bert_model.ckpt'
   vocab_path = '/path/to/vocab.txt'

   # lazy resotre
   bert_model, bert_instance, restore_weight_callback = load_bert(config_path, ckpt_path, lazy_restore=True)
   # get CLS representation
   cls_output = L.Lambda(lambda x: x[:, 0])(bert_model.output)
   output = L.Dense(2, activation='softmax',
                    kernel_intializer=bert_instance.initializer)(cls_output)
   train_model = keras.Model(bert_model.input, cls_output)
   train_model.summary()
   train_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizer.Adam(1e-5))
   # restore weights
   restore_weight_callback(bert_model)
  