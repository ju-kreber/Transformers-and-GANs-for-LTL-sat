import tensorflow as tf

from dlsgs.transformer.base import TransformerEncoder
from dlsgs.transformer import positional_encoding



class TransformerCritic(tf.keras.layers.Layer):
    def __init__(self, params, latent_mode=False, sigmoid=False):
        super().__init__()
        params = params.copy()
        self.params = params
        self.latent_mode = latent_mode
        self.transformer_encoder = TransformerEncoder(params)
        if params['gan_critic_class_layers'] > 0:
            class_enc_params = params.copy()
            class_enc_params['num_layers'] = params['gan_critic_class_layers']
            self.class_encoder = TransformerEncoder(class_enc_params)
        else:
            self.class_encoder = None
        if params['gan_critic_critic_layers'] > 0:
            critic_enc_params = params.copy()
            critic_enc_params['num_layers'] = params['gan_critic_critic_layers']
            self.critic_encoder = TransformerEncoder(class_enc_params)
        else:
            self.critic_encoder = None
        self.final_projection = tf.keras.layers.Dense(3, activation=('sigmoid' if sigmoid else None))
        if latent_mode:
            if self.params['gan_latent_upper_proj']:
                self.initial_ff = tf.keras.layers.Dense(params['d_embed_enc'], kernel_initializer='uniform')
        else:
            self.embed = tf.keras.layers.Dense(params['d_embed_enc'], kernel_initializer='uniform') # identical
        self.stdpe = positional_encoding.positional_encoding(params['max_encode_length'], params['d_embed_enc'], dtype=params['dtype'])
        self.dropout = tf.keras.layers.Dropout(params['dropout'])


    def call(self, x, positive_mask, pe=None, training=False):
        batch_size, seq_len, d_v = tf.shape(x)
        padding_mask = tf.reshape(tf.cast(tf.logical_not(positive_mask), tf.float32), [batch_size, 1, 1, seq_len])

        if pe is None:
            pe = self.stdpe[:, :seq_len, :]

        if not self.latent_mode:
            x = self.embed(x)
            x *= tf.math.sqrt(tf.cast(self.params['d_embed_enc'], self.params['dtype']))
            x += pe
        else:
            if self.params['gan_latent_upper_proj']:
                x = self.initial_ff(x)
        x = self.dropout(x, training=training)

        x = self.transformer_encoder(x, padding_mask, training=training)
        if self.class_encoder is not None:
            x_class = self.class_encoder(x, padding_mask, training=training)
        else:
            x_class = x
        if self.critic_encoder is not None:
            x_critic = self.critic_encoder(x, padding_mask, training=training)
        else:
            x_critic = x
        y_class = self.final_projection(x_class)
        y_critic = self.final_projection(x_critic)
        y = tf.concat([y_class[:, :, :1], y_critic[:, :, 1:]], axis=-1)
        y = tf.reduce_mean(y, axis=1)
        return y



class TransformerEmbedder(tf.keras.layers.Layer):
    def __init__(self, params):
        super().__init__()
        params = params.copy()
        params['num_layers'] = params['gan_embedder_layers']
        self.params = params
        
        self.stdpe = positional_encoding.positional_encoding(params['max_encode_length'], params['d_embed_enc'], dtype=params['dtype'])
        if params['num_layers'] > 0:
            self.dropout = tf.keras.layers.Dropout(params['dropout'])
            self.transformer_encoder = TransformerEncoder(params)
            self.embed = tf.keras.layers.Dense(params['d_embed_enc'], kernel_initializer='uniform') # todo check
            if params['gan_latent_lower_proj']:
                self.proj_up = tf.keras.layers.Dense(params['gan_latent_dim'])#, activation='tanh')
        else:
            self.embed = tf.keras.layers.Dense(params['gan_latent_dim'], kernel_initializer='uniform') # todo check
        
        if params['gan_embedder_dff'] > 0:
            self.pre_embed = tf.keras.layers.Dense(params['gan_embedder_dff'], kernel_initializer='uniform') # todo


    def call(self, in_soft, positive_mask, training=False):
        batch_size, seq_len, dvocab = tf.shape(in_soft)

        if self.params['gan_embedder_dff'] > 0:
            x = self.pre_embed(in_soft)
        else:
            x = in_soft
        x = self.embed(x)
        x *= tf.math.sqrt(tf.cast(self.params['d_embed_enc'], self.params['dtype']))
        pe = self.stdpe[:, :seq_len, :]
        x += pe
        if self.params['num_layers'] > 0:
            x = self.dropout(x, training=training)
            padding_mask = tf.reshape(tf.cast(tf.logical_not(positive_mask), tf.float32), [batch_size, 1, 1, seq_len])
            x = self.transformer_encoder(x, padding_mask, training=training)
            if self.params['gan_latent_lower_proj']:
                x = self.proj_up(x)
        return x



class TransformerProjector(tf.keras.layers.Layer):
    def __init__(self, params):
        super().__init__()
        params = params.copy()
        params['num_layers'] = params['gan_projector_layers']
        self.params = params
        if params['num_layers'] > 0:
            self.initial_proj = tf.keras.layers.Dense(params['d_embed_enc'], kernel_initializer='uniform')
            self.dropout = tf.keras.layers.Dropout(params['dropout'])
            self.stdpe = positional_encoding.positional_encoding(params['max_encode_length'], params['d_embed_enc'], dtype=params['dtype'])
            self.transformer_encoder = TransformerEncoder(params)
        if self.params['gan_latent_upper_proj']:
            self.initial_ff = tf.keras.layers.Dense(params['d_embed_enc'], kernel_initializer='uniform')
        if params['gan_projector_dff'] > 0:
            self.middle_projection = tf.keras.layers.Dense(params['gan_projector_dff'], activation='relu')
        self.final_projection = tf.keras.layers.Dense(params['input_vocab_size'])

    def call(self, x, positive_mask, targets, training=False):
        batch_size, seq_len, d_v = tf.shape(x)
        padding_mask = tf.reshape(tf.cast(tf.logical_not(positive_mask), tf.float32), [batch_size, 1, 1, seq_len])

        if self.params['num_layers'] > 0:
            if self.params['gan_latent_upper_proj']:
                x = self.initial_ff(x)
            x = self.dropout(x, training=training)
            x = self.transformer_encoder(x, padding_mask, training=training)
        if self.params['gan_projector_dff'] > 0:
            x = self.middle_projection(x)
        logits = self.final_projection(x)
        return logits

    def infer(self, v, positive_mask, training=False):
        logits = self(v, positive_mask, None, training=training)
        softs = tf.nn.softmax(logits)
        w = tf.cast(positive_mask, tf.float32)
        ent_per_pos = -tf.reduce_sum(softs * tf.math.log(softs), axis=-1)
        soft_entropy = tf.reduce_mean(tf.reduce_sum(ent_per_pos * w / tf.reduce_sum(w, axis=1, keepdims=True), axis=1))
        return tf.argmax(softs, axis=-1), None, soft_entropy



class TransformerGenerator(tf.keras.layers.Layer):
    def __init__(self, params, latent_mode=False, proc_logits_fn=None):
        super().__init__()
        params = params.copy()
        params['num_layers'] = params['gan_generator_layers']
        self.params = params
        self.latent_mode = latent_mode
        if not latent_mode:
            self.proc_logits_fn = proc_logits_fn
        self.transformer_encoder = TransformerEncoder(params)
        self.stdpe = positional_encoding.positional_encoding(params['max_encode_length'], params['d_embed_enc'], dtype=params['dtype'])
        self.dropout = tf.keras.layers.Dropout(params['dropout'])
        self.embedz = tf.keras.layers.Dense(params['d_embed_enc'], kernel_initializer='glorot_uniform')
        if not latent_mode:
            self.final_proj = tf.keras.layers.Dense(params['input_vocab_size'])
        elif self.params['gan_latent_lower_proj']:
            self.proj_up = tf.keras.layers.Dense(params['gan_latent_dim'], kernel_initializer='uniform')


    def call(self, z, positive_mask, pe=None, training=False):
        batch_size, seq_len, d_z = tf.shape(z)
        padding_mask = tf.reshape(tf.cast(tf.logical_not(positive_mask), tf.float32), [batch_size, 1, 1, seq_len])

        x = self.embedz(z)
        x *= tf.math.sqrt(tf.cast(self.params['d_embed_enc'], self.params['dtype']))
        if pe is None:
            pe = self.stdpe[:, :seq_len, :]

        x += pe
        x = self.dropout(x, training=True) # yes

        x = self.transformer_encoder(x, padding_mask, training=True) # yes
        if not self.latent_mode:
            x = self.final_proj(x)
            x = self.proc_logits_fn(x)
        else:
            if self.params['gan_latent_lower_proj']:
                x = self.proj_up(x)

        return x