# -*- coding: utf-8 -*-

from langml import keras, K, L
from langml.plm import load_albert, load_bert
from langml.baselines import BaselineModel, Parameters
from langml.tensor_typing import Models


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
        if backbone == 'albert':
            self.load_plm = load_albert
        else:
            self.load_plm = load_bert

    def get_pooling_output(model: Models, pooling_strategy: str = 'cls'):
        """ get pooling output
        Args:
        model: keras.Model, BERT model
        pooling_strategy: str, specify pooling strategy from ['cls', 'first-last-avg', 'last-avg'], default `cls`
        """
        assert pooling_strategy in ['cls', 'first-last-avg', 'last-avg']
        outputs, idx = [], 0
        while True:
            try:
                output = model.get_layer(
                    'Transformer-%d-FeedForward-Norm' % idx
                ).output
                outputs.append(output)
                idx += 1
            except Exception:
                break

        if pooling_strategy == 'cls':
            output = keras.layers.Lambda(lambda x: x[:, 0])(outputs[-1])
        elif pooling_strategy == 'first-last-avg':
            outputs = [
                L.GlobalAveragePooling1D()(outputs[0]),
                L.GlobalAveragePooling1D()(outputs[-1])
            ]
            output = keras.layers.Average()(outputs)
        elif pooling_strategy == 'last-avg':
            output = keras.layers.GlobalAveragePooling1D()(outputs[-1])
        else:
            raise NotImplementedError

        return output

    def build_model(self, pooling_strategy: str = 'cls', lazy_restore: bool = False) -> Models:
        assert pooling_strategy in ['cls', 'first-last-avg', 'last-avg']
        if lazy_restore:
            model, bert, restore_bert_weights = self.load_plm(self.config_path, self.ckpt_path, lazy_restore=True)
        else:
            model, bert = self.load_plm(self.config_path, self.ckpt_path)

        augmented_text_in = L.Input(shape=(None, ), name='Input-Augmented-Token')
        augmented_segment_in = L.Input(shape=(None, ), name='Input-Augmented-Segment')
        augmented_text, augmented_segment = augmented_text_in, augmented_segment_in
        augmented_model = bert(inputs=[augmented_text, augmented_segment])

        output = self.get_pooling_output(model, pooling_strategy)
        augmented_output = self.get_pooling_output(augmented_model, pooling_strategy)

        l2_normalize = L.Lambda(lambda x: K.l2_normalize(x, axis=1), name='l2_norm')
        output = l2_normalize(output)
        augmented_output = l2_normalize(augmented_output)
        similarity = L.Lambda(
            lambda x: K.dot(x[0], K.transpose(x[1]) / self.params.temperature),
            name='similarity'
        )([output, augmented_output])

        encoder = keras.Model(inputs=model.input, outputs=[model.output])
        train_model = keras.Model((*model.input, *augmented_model.input), similarity)
        train_model.summary()
        train_model.compile(keras.optimizers.Adam(self.params.learning_rate),
                            loss=simcse_loss)
        # For distributed training, restoring bert weight after model compiling.
        if lazy_restore:
            restore_bert_weights(model)

        return train_model, encoder
