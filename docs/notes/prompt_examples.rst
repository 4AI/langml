Examples of prompt-based tuning
================================
Prompt-based tuning is the latest paradigm to adapt PLMs to downstream NLP tasks, which embeds a textual template into the input text and directly uses the MLM task of PLMs to train models.


Currently support:

- PTuning: `GPT Understands, Too <https://arxiv.org/pdf/2103.10385.pdf>`_


Prompt-based Classification
------------------------------------

There are three steps to build a prompt-based classifier.

1. Define a template

.. code-block:: python
    
    from langml.prompt import Template
    from langml.tokenizer import WPTokenizer

    vocab_path = '/path/to/vocab.txt'

    tokenizer = WPTokenizer(vocab_path, lowercase=True)
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


2. Defina a prompt-based model

.. code-block:: python

    from langml.prompt import PTuniningPrompt, PTuningForClassification

    bert_config_path = '/path/to/bert_config.json'
    bert_ckpt_path = '/path/to/bert_model.ckpt'

    prompt_model = PTuniningPrompt('bert', bert_config_path, bert_ckpt_path,
                                   template, freeze_plm=False, learning_rate=5e-5, encoder='lstm')
    prompt_classifier = PTuningForClassification(prompt_model, tokenizer)


3. Train on dataset

.. code-block:: python

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


For more examples visit `langml/examples <https://github.com/4AI/langml/tree/main/examples/prompt>`_
