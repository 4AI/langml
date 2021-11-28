# -*- coding: utf-8 -*-

""" A text classification example for ptuning
"""


from langml.prompt import Template, PTuniningPrompt, PTuningForClassification
from langml.tokenizer import WPTokenizer


bert_config_path = '/Users/seanlee/Downloads/uncased_L-2_H-128_A-2/bert_config.json'
bert_ckpt_path = '/Users/seanlee/Downloads/uncased_L-2_H-128_A-2/bert_model.ckpt'
vocab_path = '/Users/seanlee/Downloads/uncased_L-2_H-128_A-2/vocab.txt'

tokenizer = WPTokenizer(vocab_path, lowercase=True)
template = Template(
    template=['it', 'was', '[MASK]', '.'],
    label_tokens_map={
        'positive': ['good'],
        'negative': ['bad', 'terrible']
    },
    tokenizer=tokenizer
)

prompt_model = PTuniningPrompt('bert', bert_config_path, bert_ckpt_path,
                               template, freeze_plm=False, learning_rate=5e-5, encoder='lstm')
prompt_classifier = PTuningForClassification(prompt_model, tokenizer)
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
