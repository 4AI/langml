# -*- coding: utf-8 -*-

from langml import keras, K, L
from langml.plm import load_albert, load_bert
from langml.baselines import BaselineModel, Parameters
from langml.tensor_typing import Models, Tensors


def simcse_loss(y_true, y_pred):
    y_true = K.cast(K.arange(0, K.shape(y_pred)[0]), dtype=K.floatx())
    return K.mean(K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True))


class SimCSE(BaselineModel):
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
        self.get_first_last_avg_lambda = L.Average(name='first-last-avg')
        self.get_last_avg_lambda = L.Lambda(lambda x: K.mean(x, axis=1), name='last-avg')

    def get_pooling_output(self, model: Models, output_index: int, pooling_strategy: str = 'cls') -> Tensors:
        """ get pooling output
        Args:
          model: keras.Model, BERT model
          output_index: int, specify output index of feedforward layer.
          pooling_strategy: str, specify pooling strategy from ['cls', 'first-last-avg', 'last-avg'], default `cls`
        """
        assert pooling_strategy in ['cls', 'first-last-avg', 'last-avg']

        if pooling_strategy == 'cls':
            return self.get_cls_lambda(model.output)

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

        if pooling_strategy == 'first-last-avg':
            outputs = [
                L.Lambda(lambda x: K.mean(x, axis=1))(outputs[0]),
                L.Lambda(lambda x: K.mean(x, axis=1))(outputs[-1])
            ]
            output = self.get_first_last_avg_lambda(outputs)
        elif pooling_strategy == 'last-avg':
            output = self.get_last_avg_lambda(outputs[-1])
        else:
            raise NotImplementedError

        return output

    def build_model(self, pooling_strategy: str = 'cls', lazy_restore: bool = False) -> Models:
        assert pooling_strategy in ['cls', 'first-last-avg', 'last-avg']
        if lazy_restore:
            model, bert, restore_bert_weights = self.load_plm(self.config_path, self.ckpt_path, lazy_restore=True)
        else:
            model, bert = self.load_plm(self.config_path, self.ckpt_path, dropout_rate=self.params.dropout_rate)

        augmented_text_in = L.Input(shape=(None, ), name='Input-Augmented-Token')
        augmented_segment_in = L.Input(shape=(None, ), name='Input-Augmented-Segment')
        augmented_text, augmented_segment = augmented_text_in, augmented_segment_in
        augmented_model = bert(inputs=[augmented_text, augmented_segment])

        output = self.get_pooling_output(model, 0, pooling_strategy)
        augmented_output = self.get_pooling_output(augmented_model, 1, pooling_strategy)

        l2_normalize = L.Lambda(lambda x: K.l2_normalize(x, axis=1), name='l2_norm')
        similarity = L.Lambda(
            lambda x: K.dot(x[0], K.transpose(x[1]) / self.params.temperature),
            name='similarity'
        )([l2_normalize(output), l2_normalize(augmented_output)])

        encoder = keras.Model(inputs=model.input, outputs=[output])
        train_model = keras.Model((*model.input, *augmented_model.input), similarity)
        train_model.summary()
        train_model.compile(keras.optimizers.Adam(self.params.learning_rate),
                            loss=simcse_loss)
        # For distributed training, restoring bert weight after model compiling.
        if lazy_restore:
            restore_bert_weights(model)

        return train_model, encoder
