# implementation based on DeepLTL https://github.com/reactive-systems/deepltl

import tensorflow as tf


def create_padding_mask(indata, pad_id, dtype=tf.float32):
    """
        indata: int tensor with shape (batch_size, input_length)
        pad_id: int, encodes the padding token
        dtype: tf.dtypes.Dtype(), data type of padding mask
    Returns:
        padding mask with shape (batch_size, 1, 1, input_length) that indicates padding with 1 and 0 everywhere else
    """
    mask = tf.cast(tf.math.equal(indata, pad_id), dtype)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size, dtype=tf.float32):
    """
    creates a look ahead mask that masks future positions in a sequence, e.g., [[[[0, 1, 1], [0, 0, 1], [0, 0, 0]]]] for size 3
    Args:
        size: int, specifies the size of the look ahead mask
        dtype: tf.dtypes.Dtype(), data type of look ahead mask
    Returns:
        look ahead mask with shape (1, 1, size, size) that indicates masking with 1 and 0 everywhere else
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size), dtype), -1, 0)
    return tf.reshape(mask, [1, 1, size, size])



def create_model(cls, params, inputs={'indata' : tf.int32}, outputs=['predictions'], training=False):
    """
    Args:
        params: dict, hyperparameter dictionary
        training: bool, whether model is called in training mode or not
        custom_pos_enc, bool, whether a custom postional encoding is provided as additional input
        attn_weights: bool, whether attention weights are part of the output
    """
    assert 'indata' in inputs
    impl_inputs = {}
    model_inputs = []
    assert not training or 'targets' in inputs

    for input_, info in inputs.items():
        if input_ == 'indata':
            indata_placeholder = tf.keras.layers.Input((None,), dtype=info, name='indata')
            impl_inputs = {'indata': indata_placeholder}
            model_inputs.append(indata_placeholder)
        elif input_ == 'positional_encoding':
            positional_encoding_placeholder = tf.keras.layers.Input((None, None,), dtype=info, name='positional_encoding')
            impl_inputs['positional_encoding'] = positional_encoding_placeholder
            model_inputs.append(positional_encoding_placeholder)
        elif input_ == 'targets':
            targets_placeholder = tf.keras.layers.Input((None,), dtype=info, name='targets')
            impl_inputs['targets'] = targets_placeholder
            model_inputs.append(targets_placeholder)
        else:
            raise ValueError("Don't know " + input_)

    impl = cls(params)
    impl_outputs = impl(impl_inputs, training=training, return_quantities=outputs)
    if 'targets' in inputs:
        targets = impl_inputs['targets']
        assert 'predictions' in impl_outputs
        assert outputs == ['predictions']
        predictions = impl_outputs['predictions']
        predictions = Seq2SeqMetricsLayer(params)([predictions, targets])
        model = tf.keras.Model(model_inputs, predictions) # MODEL: single output
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        mask = tf.cast(tf.math.logical_not(tf.math.equal(targets, params['target_pad_id'])), params['dtype'])
        loss = tf.keras.layers.Lambda(lambda x: loss_object(x[0], x[1], sample_weight=x[2]))((targets, predictions, mask))
        model.add_loss(loss)
        return model
    else:
        model_outputs = []
        for output_ in outputs:
            assert output_ in impl_outputs
            model_outputs.append(impl_outputs[output_])
        return tf.keras.Model(model_inputs, model_outputs) # MODEL: multi-output



class Seq2SeqMetricsLayer(tf.keras.layers.Layer):

    def __init__(self, params):
        """
        Args:
            params: hyperparameter dictionary containing the following keys:
                dtype: tf.dtypes.Dtype(), datatype for floating point computations
                target_pad_id: int, encodes the padding token for targets
        """
        super(Seq2SeqMetricsLayer, self).__init__()
        self.accuracy_mean = tf.keras.metrics.Mean('accuracy')
        self.accuracy_per_sequence_mean = tf.keras.metrics.Mean('accuracy_per_sequence')
        self.__dict__['params'] = params

    def get_config(self):
        return {
            'params': self.params
        }

    def call(self, inputs):
        predictions, targets = inputs[0], inputs[1]
        weights = tf.cast(tf.not_equal(targets, self.params['target_pad_id']), self.params['dtype'])
        outputs = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        targets = tf.cast(targets, tf.int32)

        # accuracy
        correct_predictions = tf.cast(tf.equal(outputs, targets), self.params['dtype'])
        accuracy = self.accuracy_mean(*(correct_predictions, weights))
        self.add_metric(accuracy)

        # accuracy per sequence
        one = tf.constant(1.0, dtype=self.params['dtype'])
        incorrect_predictions = tf.cast(tf.not_equal(outputs, targets), self.params['dtype']) * weights
        correct_sequences = one - tf.minimum(one, tf.reduce_sum(incorrect_predictions, axis=-1))
        accuracy_per_sequence = self.accuracy_per_sequence_mean(correct_sequences, one)
        self.add_metric(accuracy_per_sequence)

        return predictions