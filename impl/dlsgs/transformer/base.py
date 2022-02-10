# implementation based on DeepLTL https://github.com/reactive-systems/deepltl

import tensorflow as tf

from dlsgs.transformer import attention
from dlsgs.transformer import positional_encoding as pe
from dlsgs.transformer.common import create_padding_mask, create_look_ahead_mask
from dlsgs.transformer.beam_search import BeamSearch



class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, params):
        """
            params: hyperparameter dictionary containing the following keys:
                d_embed_enc: int, dimension of encoder embedding
                d_ff: int, hidden dimension of feed-forward networks
                dropout: float, percentage of droped out units
                ff_activation: string, activation function used in feed-forward networks
                num_heads: int, number of attention heads
        """
        super(TransformerEncoderLayer, self).__init__()
        self.__dict__['params'] = params

        self.multi_head_attn = attention.MultiHeadAttention(params['d_embed_enc'], params['num_heads'], dtype=params['dtype'])

        if 'leaky_relu' in params['ff_activation']:
            alpha = float(params['ff_activation'].split('$')[1])
            ff_activation = lambda x: tf.nn.leaky_relu(x, alpha=alpha)
        else:
            ff_activation = params['ff_activation']
        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(params['d_ff'], activation=ff_activation),
            tf.keras.layers.Dense(params['d_embed_enc'])
        ])

        self.norm_attn = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_ff = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout_attn = tf.keras.layers.Dropout(params['dropout'])
        self.dropout_ff = tf.keras.layers.Dropout(params['dropout'])

    def call(self, input, mask, training):
        """
        Args:
            input: float tensor with shape (batch_size, input_length, d_embed_dec)
            mask: float tensor with shape (batch_size, 1, 1, input_length)
            training: bool, whether layer is called in training mode or not
        """
        attn, _ = self.multi_head_attn(input, input, input, mask)
        attn = self.dropout_attn(attn, training=training)
        norm_attn = self.norm_attn(attn + input) # res connection

        ff_out = self.ff(norm_attn)
        ff_out = self.dropout_ff(ff_out, training=training)
        norm_ff_out = self.norm_ff(ff_out + norm_attn)

        return norm_ff_out



class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, params):
        """
                d_embed_dec: int, dimension of decoder embedding
                d_ff: int, hidden dimension of feed-forward networks
                dropout: float, percentage of droped out units
                ff_activation: string, activation function used in feed-forward networks
                num_heads: int, number of attention heads
        """
        super(TransformerDecoderLayer, self).__init__()
        self.__dict__['params'] = params

        self.multi_head_self_attn = attention.MultiHeadAttention(
            params['d_embed_dec'], params['num_heads'], dtype=params['dtype'])
        self.multi_head_enc_dec_attn = attention.MultiHeadAttention(
            params['d_embed_dec'], params['num_heads'], dtype=params['dtype'])

        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(params['d_ff'], activation=params['ff_activation']),
            tf.keras.layers.Dense(params['d_embed_dec'])
        ])

        self.norm_self_attn = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_enc_dec_attn = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_ff = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout_self_attn = tf.keras.layers.Dropout(params['dropout'])
        self.dropout_enc_dec_attn = tf.keras.layers.Dropout(params['dropout'])
        self.dropout_ff = tf.keras.layers.Dropout(params['dropout'])

    def call(self, input, enc_output, look_ahead_mask, padding_mask, training, cache=None):
        """
        Args:
            input: float tensor with shape (batch_size, target_length, d_embed_dec)
            enc_output: float tensor with shape (batch_size, input_length, d_embed_enc)
            look_ahead_mask: float tensor with shape (1, 1, target_length, target_length)
            padding_mask: float tensor with shape (batch_size, 1, 1, input_length)
            training: bool, whether layer is called in training mode or not
            cache: dict
        """
        self_attn, _ = self.multi_head_self_attn(input, input, input, look_ahead_mask, cache)
        self_attn = self.dropout_self_attn(self_attn, training=training)
        norm_self_attn = self.norm_self_attn(self_attn + input)

        enc_dec_attn, _ = self.multi_head_enc_dec_attn(norm_self_attn, enc_output, enc_output, padding_mask)
        enc_dec_attn = self.dropout_enc_dec_attn(enc_dec_attn, training=training)
        norm_enc_dec_attn = self.norm_enc_dec_attn(enc_dec_attn + norm_self_attn)

        ff_out = self.ff(norm_enc_dec_attn)
        ff_out = self.dropout_ff(ff_out, training=training)
        norm_ff_out = self.norm_ff(ff_out + norm_enc_dec_attn)

        return norm_ff_out



class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, params):
        """
                d_embed_enc: int, dimension of encoder embedding
                d_ff: int, hidden dimension of feed-forward networks
                dropout: float, percentage of droped out units
                ff_activation: string, activation function used in feed-forward networks
                input_vocab_size: int, size of input vocabulary
                num_heads: int, number of attention heads
                num_layers: int, number of encoder / decoder layers
        """
        super(TransformerEncoder, self).__init__()
        self.__dict__['params'] = params
        self.enc_layers = [TransformerEncoderLayer(params) for _ in range(params['num_layers'])]

    def call(self, x, padding_mask, training):
        for i in range(self.params['num_layers']):
            x = self.enc_layers[i](x, padding_mask, training)
        return x



class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, params):
        """
                d_embed_dec: int, dimension of decoder embedding
                d_ff: int, hidden dimension of feed-forward networks
                dropout: float, percentage of droped out units
                ff_activation: string, activation function used in feed-forward networks
                num_heads: int, number of attention heads
                num_layers: int, number of encoder / decoder layers
                target_vocab_size: int, size of target vocabulary         
        """
        super(TransformerDecoder, self).__init__()
        self.__dict__['params'] = params
        self.dec_layers = [TransformerDecoderLayer(params) for _ in range(params['num_layers'])]

    def call(self, x, enc_output, look_ahead_mask, padding_mask, training, cache=None):
        for i in range(self.params['num_layers']):
            layer_cache = cache[f'layer_{i}'] if cache is not None else None
            x = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask, training, layer_cache)
        return x


class Transformer(tf.keras.Model):
    def __init__(self, params):
        """
                alpha: float, strength of normalization in beam search algorithm
                beam_size: int, number of beams kept by beam search algorithm
                d_embed_enc: int, dimension of encoder embedding
                d_embed_dec: int, dimension of decoder embedding
                d_ff: int, hidden dimension of feed-forward networks
                ff_activation: string, activation function used in feed-forward networks
                num_heads: int, number of attention heads
                num_layers: int, number of encoder / decoder layer
                input_vocab_size: int, size of input vocabulary
                max_encode_length: int, maximum length of input sequence
                max_decode_length: int, maximum lenght of target sequence
                dropout: float, percentage of droped out units
                dtype: tf.dtypes.Dtype(), datatype for floating point computations
                target_start_id: int, encodes the start token for targets
                target_eos_id: int, encodes the end of string token for targets
                target_vocab_size: int, size of target vocabulary
        """
        super(Transformer, self).__init__()
        self.__dict__['params'] = params

        self.encoder_embedding = tf.keras.layers.Embedding(params['input_vocab_size'], params['d_embed_enc'], dtype=params['dtype'])
        self.encoder_positional_encoding = pe.positional_encoding(params['max_encode_length'], params['d_embed_enc'], dtype=params['dtype'])
        self.encoder_dropout = tf.keras.layers.Dropout(params['dropout'])

        self.encoder_stack = TransformerEncoder(params)

        self.decoder_embedding = tf.keras.layers.Embedding(params['target_vocab_size'], params['d_embed_dec'], dtype=params['dtype'])
        self.decoder_positional_encoding = pe.positional_encoding(params['max_decode_length'], params['d_embed_dec'], dtype=params['dtype'])
        self.decoder_dropout = tf.keras.layers.Dropout(params['dropout'])

        self.decoder_stack = TransformerDecoder(params)

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
                (target: int tensor with shape (batch_size, target_length))
        """
        indata = inputs['indata']

        input_padding_mask = create_padding_mask(indata, self.params['input_pad_id'], self.params['dtype'])

        if 'positional_encoding' in inputs:
            positional_encoding = inputs['positional_encoding']
        else:
            seq_len = tf.shape(indata)[1]
            positional_encoding = self.encoder_positional_encoding[:, :seq_len, :]
        encoder_outdata = self.encode(indata, input_padding_mask, positional_encoding, training)

        if 'targets' in inputs:
            targets = inputs['targets']
            predictions = self.predict_tf(targets, encoder_outdata, input_padding_mask, training)
            return {'predictions': predictions}
        else:
            return self.decode(encoder_outdata, input_padding_mask, training, return_quantities)


    def encode(self, indata, padding_mask, positional_encoding, training):
        """
            indata: int tensor with shape (batch_size, input_length)
            padding_mask: float tensor with shape (batch_size, 1, 1, input_length)
            positional_encoding: float tensor with shape (batch_size, input_length, d_embed_enc)
            training: boolean, specifies whether in training mode or not
        """
        input_embedding = self.encoder_embedding(indata)
        input_embedding *= tf.math.sqrt(tf.cast(self.params['d_embed_enc'], self.params['dtype']))
        input_embedding += positional_encoding
        input_embedding = self.encoder_dropout(input_embedding, training=training)
        encoder_output = self.encoder_stack(input_embedding, padding_mask, training)
        return encoder_output


    def predict_tf(self, targets, encoder_outdata, input_padding_mask, training):
        """
            target: int tensor with shape (bath_size, target_length)
            encoder_output: float tensor with shape (batch_size, input_length, d_embedding)
            input_padding_mask: float tensor with shape (batch_size, 1, 1, input_length)
            training: boolean, specifies whether in training mode or not
        """
        target_length = tf.shape(targets)[1]
        look_ahead_mask = create_look_ahead_mask(target_length, self.params['dtype'])
        target_padding_mask = create_padding_mask(targets, self.params['input_pad_id'], self.params['dtype'])
        decoder_mask = tf.maximum(look_ahead_mask, target_padding_mask)

        # shift targets to the right, insert start_id at first postion, and remove last element
        decoder_indata = tf.pad(targets, [[0, 0], [1, 0]], constant_values=self.params['target_start_id'])[:, :-1]

        decoder_embedding = self.decoder_embedding(decoder_indata)  # (batch_size, target_length, d_embedding)
        decoder_embedding *= tf.math.sqrt(tf.cast(self.params['d_embed_dec'], self.params['dtype']))
        decoder_embedding += self.decoder_positional_encoding[:, :target_length, :]
        decoder_embedding = self.decoder_dropout(decoder_embedding, training=training)

        decoder_outdata = self.decoder_stack(decoder_embedding, encoder_outdata, decoder_mask, input_padding_mask, training)
        outdata_logits = self.final_projection(decoder_outdata) # (batch_size, target_length, target_vocab_size)
        predictions = self.softmax(outdata_logits)
        return predictions


    def decode(self, encoder_outdata, input_padding_mask, training, return_quantities):
        """
            encoder_output: float tensor with shape (batch_size, input_length, d_embedding)
            encoder_attn_weights: dictionary, self attention weights of the encoder
            input_padding_mask: flaot tensor with shape (batch_size, 1, 1, input_length)
            training: boolean, specifies whether in training mode or not
        """
        batch_size = tf.shape(encoder_outdata)[0]

        def logits_fn(ids, i, cache):
            """
            Args:
                ids: int tensor with shape (batch_size * beam_size, index + 1)
                index: int, current index
                cache: dictionary storing encoder output, previous decoder attention values
            Returns:
                logits with shape (batch_size * beam_size, vocab_size) and updated cache
            """
            # set input to last generated id
            decoder_input = ids[:, -1:]
            decoder_input = self.decoder_embedding(decoder_input)
            decoder_input *= tf.math.sqrt(tf.cast(self.params['d_embed_dec'], self.params['dtype']))
            decoder_input += self.decoder_positional_encoding[:, i:i + 1, :]

            look_ahead_mask = create_look_ahead_mask(self.params['max_decode_length'], self.params['dtype'])
            self_attention_mask = look_ahead_mask[:, :, i:i + 1, :i + 1]
            decoder_output = self.decoder_stack(decoder_input, cache['encoder_output'], self_attention_mask, cache['input_padding_mask'], training, cache)

            output = self.final_projection(decoder_output)
            probs = self.softmax(output)
            probs = tf.squeeze(probs, axis=[1])
            return probs, cache

        initial_ids = tf.ones([batch_size], dtype=tf.int32) * self.params['target_start_id']

        num_heads = self.params['num_heads']
        d_heads = self.params['d_embed_dec'] // num_heads
        # create cache structure for decoder attention
        cache = {
            'layer_%d' % layer: {
                'keys': tf.zeros([batch_size, 0, num_heads, d_heads], dtype=self.params['dtype']),
                'values': tf.zeros([batch_size, 0, num_heads, d_heads], dtype=self.params['dtype'])
            } for layer in range(self.params['num_layers'])
        }
        # add encoder output to cache
        cache['encoder_output'] = encoder_outdata
        cache['input_padding_mask'] = input_padding_mask

        beam_search = BeamSearch(logits_fn, batch_size, self.params)
        decoded_ids, scores = beam_search.search(initial_ids, cache)

        top_decoded_ids = decoded_ids[:, 0, 1:]
        top_scores = scores[:, 0]

        res = {}
        for q in return_quantities:
            if q == 'decodings':
                res['decodings'] = top_decoded_ids
            elif q == 'scores':
                res['scores'] = top_scores
            else:
                raise ValueError("Don't know " + q)
        return res

