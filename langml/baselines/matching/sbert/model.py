# -*- coding: utf-8 -*-

from langml import keras, K, L
from langml.plm import load_albert, load_bert
from langml.baselines import BaselineModel, Parameters
from langml.tensor_typing import Models, Tensors


class SentenceBert(BaselineModel):
    def __init__(self,
                 config_path: str,
                 ckpt_path: str,
                 params: Parameters,
                 backbone: str = 'roberta'):
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.params = params
        assert backbone in ['bert', 'roberta', 'albert']
        self.backbone = backbone
        if backbone == 'albert':
            self.load_plm = load_albert
        else:
            self.load_plm = load_bert

        self.get_cls_lambda = L.Lambda(lambda x: x[:, 0], name='cls')
        self.get_mean_lambda = L.Lambda(lambda x: K.mean(x, axis=1), name='mean-pooling')
        self.get_avg_lambda = L.Average(name='avg')
        self.get_max_lambda = L.Lambda(lambda x: K.max(x, axis=1), name='max-pooling')

    def get_pooling_output(self, model: Models, output_index: int, pooling_strategy: str = 'cls') -> Tensors:
        """ get pooling output
        Args:
          model: keras.Model, BERT model
          output_index: int, specify output index of feedforward layer.
          pooling_strategy: str, specify pooling strategy from ['cls', 'first-last-avg', 'last-avg'], default `cls`
        """
        assert pooling_strategy in ['cls', 'mean', 'max']

        if pooling_strategy == 'cls':
            return self.get_cls_lambda(model.output)

        if pooling_strategy == 'max':
            return self.get_max_lambda(model.output)

        outputs, idx = [], 0
        if self.backbone == 'albert':
            while True:
                try:
                    output = model.get_layer('Transformer-FeedForward-Norm').get_output_at(idx)
                    outputs.append(output)
                    idx += 1
                except Exception:
                    break
            N = len(outputs)
            if output_index == 0:
                outputs = outputs[:N // 2]
            elif output_index == 1:
                outputs = outputs[N // 2:]
        else:
            while True:
                try:
                    output = model.get_layer(
                        'Transformer-%d-FeedForward-Norm' % idx
                    ).get_output_at(output_index)
                    outputs.append(output)
                    idx += 1
                except Exception:
                    break
        outputs = [self.get_mean_lambda(output) for output in outputs]
        return self.get_avg_lambda(outputs)

    def build_model(self,
                    task: str = 'regression',
                    pooling_strategy: str = 'cls',
                    lazy_restore: bool = False) -> Models:
        assert task in ['regression', 'classification']
        assert pooling_strategy in ['cls', 'mean', 'max']
        if lazy_restore:
            model, bert, restore_bert_weights = self.load_plm(
                self.config_path, self.ckpt_path, lazy_restore=True)
        else:
            model, bert = self.load_plm(
                self.config_path, self.ckpt_path, dropout_rate=self.params.dropout_rate)

        right_text_in = L.Input(shape=(None, ), name='Input-Right-Token')
        right_segment_in = L.Input(shape=(None, ), name='Input-Right-Segment')
        right_text, right_segment = right_text_in, right_segment_in
        right_model = bert(inputs=[right_text, right_segment])

        pooling = self.get_pooling_output(model, 0, pooling_strategy)
        right_pooling = self.get_pooling_output(right_model, 1, pooling_strategy)

        if task == 'regression':
            output = L.Dot(axes=1, normalize=True)([pooling, right_pooling])
            loss = 'mse'
        else:
            output = L.Concatenate(axis=1)([
                pooling, right_pooling,
                L.Lambda(lambda x: K.abs(x[0] - x[1]))([pooling, right_pooling])
            ])
            output = L.Dense(self.params.tag_size, activation='softmax')(output)
            loss = 'sparse_categorical_crossentropy'

        encoder = keras.Model(inputs=model.input, outputs=[pooling])
        train_model = keras.Model((*model.input, *right_model.input), output)
        train_model.summary()
        train_model.compile(keras.optimizers.Adam(self.params.learning_rate),
                            loss=loss,
                            metrics=['accuracy'])
        # For distributed training, restoring bert weight after model compiling.
        if lazy_restore:
            restore_bert_weights(model)

        return train_model, encoder
