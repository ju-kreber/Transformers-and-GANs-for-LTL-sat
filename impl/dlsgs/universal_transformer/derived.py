import tensorflow as tf

import dlsgs.transformer.positional_encoding as posenc
from dlsgs.transformer.common import create_padding_mask
from dlsgs.transformer.base import TransformerEncoder
from dlsgs.universal_transformer.base import UniversalTransformerEncoder
from dlsgs.universal_transformer.utils import StoppedPercentMetric



class UniversalEncoderOnlyTransformer(tf.keras.Model):
    def __init__(self, params, training):
        super().__init__()
        self.params = params
        batch_size, max_seq_len, max_iterations, d_embed, dtype = params['batch_size'], params['max_encode_length'], params['ut_max_iterations'], params['d_embed_enc'], params['dtype']
        self.embedding = tf.keras.layers.Embedding(params['input_vocab_size'], d_embed, dtype=dtype)
        self.std_pe = posenc.positional_encoding(max_seq_len, d_embed, dtype=dtype)
        self.embedding_dropout = tf.keras.layers.Dropout(params['dropout'])
        predict_fn = lambda state, info, training: self.final_predict(state[0], training=training)
        if params['ut_pre_layers'] > 0:
            assert params['ut_gradient_method'] == 'tape'
            pre_params = params.copy()
            pre_params['num_layers'] = params['ut_pre_layers']
            self.pre_encoder = TransformerEncoder(pre_params)
        self.ut_encoder = UniversalTransformerEncoder(params, predict_fn=predict_fn)
        if params['ut_post_layers'] > 0:
            assert params['ut_gradient_method'] == 'tape'
            post_params = params.copy()
            post_params['num_layers'] = params['ut_post_layers']
            self.post_encoder = TransformerEncoder(post_params)
        self.final_projection = tf.keras.layers.Dense(params['target_vocab_size'])
        self.softmax = tf.keras.layers.Softmax(dtype=params['dtype'])
        

        self.epoch = 0
        self.epoch_steps = 0
        self.is_training = training


    def get_config(self):
        return {'params' : self.params}


    def prepare(self, indata, positional_encoding=None, training=False):
        input_padding_mask = create_padding_mask(indata, self.params['input_pad_id'], self.params['dtype'])
        batch_size, seq_len = tf.shape(indata)
        if positional_encoding is None:
            positional_encoding = self.std_pe[:, :seq_len, :]
        
        x = self.embedding(indata)
        x *= tf.math.sqrt(tf.cast(self.params['d_embed_enc'], self.params['dtype']))
        x += positional_encoding
        x = self.embedding_dropout(x, training=training)

        info = {'positional_encoding' : positional_encoding, 'padding_mask' : input_padding_mask}
        return x, info


    def final_predict(self, x, training):
        if self.params['enc_accumulation'] == 'first':
            x = x[:, 0, :]
        elif self.params['enc_accumulation'] == 'mean-before':
            x = tf.reduce_mean(x, axis=1)
        projected = self.final_projection(x)
        if self.params['enc_accumulation'] == 'mean-after':
            projected = tf.reduce_mean(projected, axis=1)
        predictions = self.softmax(projected)
        return predictions


    def call(self, indata, positional_encoding=None, training=False, record_history=False):
        x, info = self.prepare(indata, positional_encoding, training)
        if self.params['ut_pre_layers'] > 0:
            x = self.pre_encoder(x, padding_mask=info['padding_mask'], training=training)
        x, info = self.ut_encoder(x, info, max_iterations=self.get_iterations(), training=training, record_history=record_history)
        if self.params['ut_post_layers'] > 0:
            x = self.post_encoder(x, padding_mask=info['padding_mask'], training=training)
        y = self.final_predict(x, training=training)
        return y


    def test_step(self, data):
        x, y_target = data
        x, _ = x
        x_, info_ = self.prepare(x, None, training=False)
        if self.params['ut_pre_layers'] > 0:
            x_ = self.pre_encoder(x_, padding_mask=info_['padding_mask'], training=False)
        
        for what, iterations in enumerate([self.get_iterations()] + self.params['ut_test_iterations']):
            info = info_.copy()
            x, info = self.ut_encoder(x_, info, max_iterations=iterations, training=False, record_history=False)
            if self.params['ut_post_layers'] > 0:
                x = self.post_encoder(x, padding_mask=info['padding_mask'], training=False)
            predictions = self.final_predict(x, training=False)
            if what == 0:
                self.compiled_loss(y_target, predictions) # apparently updates something?
                # self.compiled_metrics.update_state(y_target, predictions)
                for m in self.metrics:
                    if isinstance(m, StoppedPercentMetric):
                        m.update_state(percent_stopped=info['percent_stopped'])
                    elif not '_at_' in m.name:
                        m.update_state(y_target, predictions)
            else:
                for m in self.metrics:
                    if m.name.endswith('_at_{}_it'.format(iterations)):
                        m.update_state(y_target, predictions)

        return {m.name: m.result() for m in self.metrics}


    def train_step(self, data):
        x, y_target  = data
        x, _ = x

        if 'tape' in self.params['ut_gradient_method']:
            with tf.GradientTape() as single_tape:
                x_, info = self.prepare(x, None, training=True)
                if self.params['ut_pre_layers'] > 0:
                    x_ = self.pre_encoder(x_, padding_mask=info['padding_mask'], training=True)
                x_, final_info = self.ut_encoder(x_, info, max_iterations=self.get_iterations(), training=True, record_history=False, step=self.optimizer.iterations)
                if self.params['ut_post_layers'] > 0:
                    x_ = self.post_encoder(x_, padding_mask=info['padding_mask'], training=True)
                predictions = self.final_predict(x_, training=True)
                loss_single = self.compiled_loss(y_target, predictions, regularization_losses=self.losses)
            vars_single = self.trainable_variables
            grads_single = single_tape.gradient(loss_single, vars_single)
            if self.params['ut_gradient_method'] == 'tape':
                all_vars = vars_single
                all_grads = grads_single
        
        if 'checkpoint' in self.params['ut_gradient_method']:
            assert self.params['ut_pre_layers'] == 0 and self.params['ut_post_layers'] == 0
            with tf.GradientTape() as initial_tape:
                initial_x, initial_info = self.prepare(x, None, training=True)

            # forward!
            max_iterations = self.get_iterations(step=self.optimizer.iterations, epoch=self.epoch)
            final_x, final_info = self.ut_encoder(initial_x, initial_info, max_iterations=max_iterations, training=True, record_history=True, step=self.optimizer.iterations)

            with tf.GradientTape() as final_tape:
                final_tape.watch(final_x)
                predictions = self.final_predict(final_x, training=True)
                loss = self.compiled_loss(y_target, predictions, regularization_losses=self.losses)

            # backward!
            projection_vars = self.final_projection.trainable_variables
            _dl = final_tape.gradient(loss, [final_x] + projection_vars)
            dfinal_x, dproject_var = _dl[:1], _dl[1:]

            dinitial_x, loop_grads, loop_vars, stats = self.ut_encoder.backward(dfinal_x)

            embedding_vars = self.embedding.trainable_variables
            dembedding_var = initial_tape.gradient(initial_x, embedding_vars, output_gradients=dinitial_x)        
            all_vars = projection_vars + loop_vars + embedding_vars
            all_grads = dproject_var + loop_grads + dembedding_var

            self.avg_reconstruction += stats['reconstruction_error']

        if self.params['ut_gradient_method'] == 'tape+checkpoint':
            assert len(vars_single) == len(all_vars)
            for v_alt, g_alt in zip(vars_single, grads_single):
                ind = [i for i, v in enumerate(all_vars) if v.name == v_alt.name]
                assert len(ind) == 1
                g = all_grads[ind[0]]
                if isinstance(g, tf.IndexedSlices):
                    assert tf.reduce_all(g.indices == g_alt.indices)
                    acc = tf.constant(0.)
                    for i, q in enumerate(g.indices):
                        acc += tf.reduce_mean((g.values[i] - g_alt.values[i]) ** 2)
                    diff = acc/len(g.indices)
                else:
                    diff = tf.reduce_mean((g - g_alt)**2)
                print('grad diff', v_alt.name, diff.numpy())

        self.epoch_steps += 1
        self.optimizer.apply_gradients(zip(all_grads, all_vars))
        self.compiled_metrics.update_state(y_target, predictions)
        for m in self.metrics:
            if isinstance(m, StoppedPercentMetric):
                m.update_state(percent_stopped=final_info['percent_stopped'])
        return {m.name: m.result() for m in self.metrics}


    def get_iterations(self, step=None, epoch=None):
        max_iterations = self.params['ut_max_iterations']
        if not self.is_training or self.params['ut_iterations'] == 'max':
            return max_iterations
        assert self.params['ut_iterations'] == 'schedule'
        step = step or self.optimizer.iterations.read_value()
        epoch = epoch or self.epoch
        value = epoch
        borders = [5, 10, 15, 20, 25, 30, 35, 40]
        targets = [2,  5,  7, 10, 12, 15, 17, 20]
        # value = step
        # borders = [ 7810, 13277, 17182, 21087]
        # targets = [ 5,       10,    15,    20]
        for border, target in zip(borders, targets):
            if value < border:
                return min(target, max_iterations)
        return max_iterations

    def on_epoch_begin(self, epoch, logs=None):
        self.avg_reconstruction = 0
        self.epoch_steps = 0
        self.epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        if self.params['ut_iterations'] == 'schedule':
            print('Epoch', epoch + 1, 'currently', self.get_iterations(), 'target iterations')
        if 'checkpoint' in self.params['ut_gradient_method']:
            if self.avg_reconstruction / self.epoch_steps > 0.001:
                print('WARNING: Average reconstruction error', (self.avg_reconstruction / self.epoch_steps).numpy())
