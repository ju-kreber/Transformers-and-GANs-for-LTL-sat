# implementation based on DeepLTL https://github.com/reactive-systems/deepltl

import tensorflow as tf

import dlsgs.transformer.positional_encoding as pe
from dlsgs.transformer.base import Transformer, TransformerEncoder
from dlsgs.transformer.common import create_padding_mask



class EncoderOnlyTransformer(Transformer):
    def __init__(self, params):
        tf.keras.Model.__init__(self)
        self.__dict__['params'] = params

        self.encoder_embedding = tf.keras.layers.Embedding(params['input_vocab_size'], params['d_embed_enc'], dtype=params['dtype'])
        self.encoder_positional_encoding = pe.positional_encoding(params['max_encode_length'], params['d_embed_enc'], dtype=params['dtype'])
        self.encoder_dropout = tf.keras.layers.Dropout(params['dropout'])
        self.encoder_stack = TransformerEncoder(params)

        self.final_projection = tf.keras.layers.Dense(params['target_vocab_size'])
        self.softmax = tf.keras.layers.Softmax(dtype=params['dtype'])

    def get_config(self):
        return {
            'params': self.params
        }

    def call(self, inputs, training, return_quantities=[]):
        """
            inputs:
                indata: int tensor with shape (batch_size, input_length)
                (positional_encoding: float tensor with shape (batch_size, input_length, d_embed_enc), custom postional encoding)
                (target: int tensor with shape (batch_size, 1))
        """
        indata = inputs['indata']
        input_padding_mask = create_padding_mask(indata, self.params['input_pad_id'], self.params['dtype'])
        if 'positional_encoding' in inputs:
            positional_encoding = inputs['positional_encoding']
        else:
            seq_len = tf.shape(indata)[1]
            positional_encoding = self.encoder_positional_encoding[:, :seq_len, :]

        encoder_outdata = self.encode(indata, input_padding_mask, positional_encoding, training)
        predictions = self.predict_(encoder_outdata, input_padding_mask, training)

        returns = {}
        if 'predictions' in return_quantities:
            returns['predictions'] = tf.expand_dims(predictions, 1)
        if 'decodings' in return_quantities:
            returns['decodings'] = tf.expand_dims(tf.argmax(predictions, axis=-1), -1)
        return returns


    def predict_(self, encoder_outdata, input_padding_mask, training):
        if self.params['enc_accumulation'] == 'first':
            encoder_outdata = encoder_outdata[:, 0, :]
        elif self.params['enc_accumulation'] == 'mean-before':
            encoder_outdata = tf.reduce_mean(encoder_outdata, axis=1)
        projected = self.final_projection(encoder_outdata)
        if self.params['enc_accumulation'] == 'mean-after':
            projected = tf.reduce_mean(projected, axis=1)
        predictions = self.softmax(projected)
        return predictions
