# partly based on https://github.com/cerebroai/reformers under MIT License

import tensorflow as tf
import numpy as np

import dlsgs.transformer.attention as attention


class UniversalTransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, params, optimizer=None, state_format=['x'], predict_fn=None, change_fn=None, final_fn=lambda s,i,t: s[0]):
        super().__init__()
        self.params = params
        self.optimizer = optimizer # only for iterations count -> seed
        self.predict_fn = predict_fn
        self.change_fn = change_fn
        self.final_fn = final_fn
        self.state_format = state_format
        self.base_layers = [UTBaseLayer(params) for _ in range(params['ut_base_layers'])]
        self.__dict__['state_hist'] = []
        self.__dict__['info_hist'] = []
        self.seed = np.random.randint(2**16)
        self.step = None
        #self.i = tf.Variable(0, trainable=False)
        

    def prepare(self, x, info, training):
        batch_size, seq_len, d_emb = tf.shape(x)
        self.__dict__['state_hist'] = []
        self.__dict__['info_hist'] = []
        state = [None] * len(self.state_format)
        # state[1] = tf.zeros([50], dtype=tf.bool)
        # state[2] = tf.constant((), shape=[batch_size, 0, self.params['input_vocab_size']])
        assert self.state_format[0] == 'x'
        state[0] = x
        stopped_elements = tf.zeros([batch_size], dtype=tf.bool)
        info['stopped_elements'] = stopped_elements
        info['finished_state'] = False
        info['shape'] = tf.shape(x)
        info['i'] = -1
        return state, info


    def loop(self, state, info, max_iterations, training, record_history):
        self.update_info(info, training=training)
        if record_history:
            self.state_hist.append(state)
            self.info_hist.append(info)
            info = info.copy()
        state = self.base_layers[0](state, info, training=training)
        if self.change_fn is not None:
            state = self.change_fn(state, info, training)
        stop = self.check_break(state, info, training=training)
        self.update_info(info, training=training)

        while info['i'] < max_iterations and not stop:
            if record_history:
                self.state_hist.append(state)
                self.info_hist.append(info)
                info = info.copy()
            num_base_layers = len(self.base_layers)
            for j in range(num_base_layers):
                if info['i'] % num_base_layers == j:
                    state = self.base_layers[j](state, info, training=training)
            if self.change_fn is not None:
                state = self.change_fn(state, info, training)
            stop = self.check_break(state, info, training=training)
            self.update_info(info, training=training)
        info['finished_state'] = True
        if record_history:
            self.state_hist.append(state)
            self.info_hist.append(info)

        info['percent_stopped'] = tf.reduce_mean(tf.cast(info['stopped_elements'], tf.float32)) * 100
        return state, info

    def call(self, x, info, max_iterations=None, training=False, record_history=False, step=None):
        self.step = step
        if max_iterations is None:
            max_iterations = self.params['ut_max_iterations']
        state, info = self.prepare(x, info, training=training)
        state, info = self.loop(state, info, max_iterations, training, record_history)
        return self.final_fn(state, info, training), info


    def backward(self, upstream_grads, with_final=True):
        all_vars = []
        all_grads = []
        final_state, final_info = self.state_hist.pop(), self.info_hist.pop()
        assert final_info['finished_state']
        assert not self.info_hist[-1]['finished_state']

        current_state = final_state # will immediately get reassigned
        len_state = len(current_state)
        dcurrent_state = upstream_grads
        layer_vars = {}
        layer_grads = {}
        reconstruction_error = 0
        iterations = final_info['i']
        for i in reversed(range(iterations)):
            i_layer = i % len(self.base_layers)
            weighting = 1 # / math.sqrt(max(i, 1))
            y_state = current_state
            current_state = self.state_hist.pop()
            current_info = self.info_hist.pop()
            with tf.GradientTape() as tape:
                tape.watch(current_state)
                res_state = self.base_layers[i_layer](current_state, current_info, training=True)
            for a, b in zip(res_state, y_state):
                reconstruction_error += tf.reduce_mean((a - b)**2)
            vars_ = self.base_layers[i_layer].trainable_variables
            _dl = tape.gradient(res_state, current_state + vars_, output_gradients=dcurrent_state)
            dcurrent_state, dlayer_vars = _dl[:len_state], _dl[len_state:]
            if not i_layer in layer_vars:
                layer_vars[i_layer] = vars_
                layer_grads[i_layer] = dlayer_vars
            else:
                layer_grads[i_layer] = [ old + q * weighting for old, q in zip(layer_grads[i_layer], dlayer_vars) ]
        for i_layer in layer_vars:
            all_vars.extend(layer_vars[i_layer])
            all_grads.extend(layer_grads[i_layer])

        reconstruction_error /= tf.cast(iterations, tf.float32)
        return dcurrent_state, all_grads, all_vars, {'reconstruction_error' : reconstruction_error}


    def update_info(self, info, training=True):
        info['i'] += 1
        step = None
        if self.step is not None:
            step = self.step
        elif training:
            assert self.optimizer is not None, "need step for seed during training!"
            step = tf.cast(self.optimizer.iterations, tf.int32)
        if step is not None:
            info['seed'] = self.seed * 10000 + step * 100 + info['i']
        else:
            info['seed'] = None


    def check_break(self, state, info, training=True):
        batch_size = info['shape'][0]
        if self.predict_fn is not None:
            if self.params['ut_stop_map_certainty']:
                predictions = self.predict_fn(state, info, training=training)
                highest = tf.reduce_max(predictions, axis=-1)
                element_stop_probs = self.params['ut_stop_base'] * tf.where(highest > 0.5, 16*(highest - 0.5)**4, tf.zeros_like(highest))
            else:
                element_stop_probs = self.params['ut_stop_base'] * tf.ones([batch_size])
            stop_elements = tf.random.uniform([batch_size]) < element_stop_probs
            next_stopped = tf.logical_or(info['stopped_elements'], stop_elements)
            info['stopped_elements'] = next_stopped
            all_stopped = tf.reduce_all(next_stopped)
        else:
            all_stopped = False
        break_all = training and tf.random.uniform(()) < self.params['ut_random_all_stop']
        return all_stopped or break_all




class UTBaseLayer(tf.keras.Model):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.multi_head_attn = attention.MultiHeadAttention(params['d_embed_enc'], params['num_heads'], dtype=params['dtype'])
        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(params['d_ff'], activation=params['ff_activation']),
            tf.keras.layers.Dense(params['d_embed_enc'])
        ])
        self.norm_attn = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_ff = tf.keras.layers.LayerNormalization(epsilon=1e-6)


    def call(self, state, info, training):
        x, *rest = state
        x_attended, _ = self.multi_head_attn(x, x, x, info['padding_mask'])

        if training:
            seed = info['seed']
            if seed is not None:
                tf.random.set_seed(seed*2 + 0)
            x_attended = tf.nn.dropout(x_attended, self.params['dropout'])
        x_normed = self.norm_attn(x_attended + x) # res connection

        x_ffed = self.ff(x_normed)
        if training:
            if seed is not None:
                tf.random.set_seed(seed*2 + 1)
            x_ffed = tf.nn.dropout(x_ffed, self.params['dropout'])
        y = self.norm_ff(x_ffed + x_normed)

        outstate = [y] + rest
        return outstate
