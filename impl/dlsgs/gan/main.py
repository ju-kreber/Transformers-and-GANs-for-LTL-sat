# pylint: disable=line-too-long, invalid-name

from functools import partial
import os

import tensorflow as tf

from dlsgs.transformer import lr_schedules
from dlsgs.gan.utils import ReplayBuffer, CreatedBuffer, parse_score, parse_score_math, get_corrects
from dlsgs.gan.base import TransformerGenerator, TransformerProjector, TransformerEmbedder, TransformerCritic
from dlsgs.utils.utils import increment

ASSERT_FINITE = False

class TransformerGAN(tf.keras.Model):
    def __init__(self, params, vocab, log_dir):
        super().__init__()
        # instance parameters
        self.__dict__['params'] = params # do not checkpoint!
        self.latent_mode = params['gan_latent_mode'] if params['gan_latent_mode'] != 'false' else False
        self.generate_classes = params['gan_generate_classes']
        self.generate_confusion = params['gan_generate_confusion']
        self.inherent_class_loss = self.generate_classes or self.generate_confusion

        if params['gan_latent_dim'] != params['d_embed_enc']:
            assert params['gan_latent_lower_proj'] and params['gan_latent_upper_proj'], "Must project if latent dim does not match embedding dim"

        self.__dict__['objectives'] = params['objectives'].split(',') # do not checkpoint!
        if params['tree_pe']:
            assert params['gan_copy_shape_critic'] and params['gan_copy_shape_generator'] and params['gan_copy_shape_val']
        self.warnings = {}
        self.vocab = vocab
        self.tb_writer = tf.summary.create_file_writer(log_dir + '/train_custom')
        self.epoch, self.epoch_steps, self.test_steps, self.total_steps = 0, 0, 0, 0
        self.last_ana = None

        # build child instances
        if self.latent_mode:
            self.embedder = TransformerEmbedder(params)
            self.projector = TransformerProjector(params)
            proc_logits_fn = None
        else:
            proc_logits_fn = partial(proc_logits, normalize=True, tau=1, sample=False, calc_entropy=False)
        self.generator = TransformerGenerator(params, latent_mode=self.latent_mode, proc_logits_fn=proc_logits_fn)
        self.critic = TransformerCritic(params, latent_mode=self.latent_mode, sigmoid=False)
        if params['gan_replay_buffer_fraction'] > 0:
            self.replay_buffer = ReplayBuffer(params, update_method='reservoir') # update_method='constant$0.33'
        if params['gan_incremental_learning_mode']:
            self.created_buffer = CreatedBuffer(params)

        # build losses and optimizers
        self.dyn_learning_rate = lr_schedules.TransformerSchedule(params['d_embed_enc'], warmup_steps=params['warmup_steps']) # only for pure classification
        fixed_learning_rate = self.params['gan_learning_rate']
        if self.latent_mode:
            self.project_back_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        if self.params['gan_class_loss'] == 'crossentropy':
            self.class_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0)
        elif self.params['gan_class_loss'] == 'hinge':
            self.class_loss = tf.keras.losses.Hinge()
        if 'class' in self.objectives and not 'gan' in self.objectives and not params['gan_force_constant_lr']:
            if self.latent_mode:
                self.opti_emb = tf.keras.optimizers.Adam(learning_rate=self.dyn_learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
                self.opti_proj = tf.keras.optimizers.Adam(learning_rate=self.dyn_learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
            self.opti_c = tf.keras.optimizers.Adam(learning_rate=self.dyn_learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        else:
            if self.latent_mode:
                self.opti_emb = tf.keras.optimizers.Adam(learning_rate=fixed_learning_rate, beta_1=0., beta_2=0.9)
                self.opti_proj = tf.keras.optimizers.Adam(learning_rate=fixed_learning_rate, beta_1=0., beta_2=0.9)
            self.opti_c = tf.keras.optimizers.Adam(learning_rate=fixed_learning_rate, beta_1=0., beta_2=0.9)
        if 'gan' in self.objectives:
            self.opti_g = tf.keras.optimizers.Adam(learning_rate=fixed_learning_rate, beta_1=0., beta_2=0.9)
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


    def input_noise(self, dimensions_per_position, batch_size_or_mask, len_mode='uniform', min_len=1, max_len=None, random_dist='uniform', add_classes=False):
        if max_len is None:
            max_len = self.params['max_encode_length']
        if len_mode == 'copy':
            batch_size, max_len = tf.shape(batch_size_or_mask)
            positive_mask = tf.identity(batch_size_or_mask)
            lengths = tf.reduce_sum(tf.cast(positive_mask, tf.int32), axis=1)
        else:
            batch_size = batch_size_or_mask
        if random_dist == 'uniform':
            z = tf.random.uniform((batch_size, max_len, dimensions_per_position), 0, 1, dtype=tf.float32)
        else:
            raise NotImplementedError # TODO implement normal?
        if len_mode != 'copy':
            if len_mode == 'uniform':
                lengths = tf.random.uniform((batch_size,), min_len, max_len, dtype=tf.int32) + 1
            z = z[:, :max_len, :]
            range_ = tf.reshape(tf.range(0, max_len), (1, max_len))
            positive_mask = range_ < lengths[:, tf.newaxis]
            assert all(tf.shape(positive_mask) == (batch_size, max_len))
        if add_classes:
            z_seq_len = tf.shape(z)[1]
            gen_classes = tf.cast(tf.random.uniform([batch_size], 0, 2, dtype=tf.int32), tf.float32)
            gen_classes_rep = tf.repeat(gen_classes[:, tf.newaxis, tf.newaxis], [z_seq_len], axis=1)
            z = tf.concat([z, gen_classes_rep], axis=-1) # add class as input
        else:
            gen_classes = None
        return z, positive_mask, gen_classes


    def encode_real(self, x):
        sigma = self.params['gan_sigma_real']
        real_positive_mask = x != self.params['input_pad_id']
        real_samples_cont = tf.one_hot(x, self.params['input_vocab_size'], axis=-1)
        if sigma == 0:
            return real_samples_cont, real_positive_mask
        real_samples_cont += tf.math.abs(tf.random.normal(tf.shape(real_samples_cont), mean=0, stddev=sigma))
        real_samples_cont /= tf.reduce_sum(real_samples_cont, axis=-1, keepdims=True)
        return real_samples_cont, real_positive_mask


    def train_step(self, data):
        x, y_target = data
        if self.params['tree_pe']:
            x, pe, _ = x
        else:
            x, _ = x
            pe = None
        batch_size, seq_len = tf.shape(x)
        critic_variables, generator_variables, embedder_variables, projector_variables = None, None, None, None
        metrics = {}

        # Incremental learning, created buffer
        if self.params['gan_incremental_learning_mode']:
            zero_step, full_step = self.params['gan_incremental_usage_zero_step'], self.params['gan_incremental_usage_full_step'] 
            percentage_from_buffer = (self.total_steps - zero_step) / (full_step - zero_step)
            percentage_from_buffer = min(max(0, percentage_from_buffer), 1)
            items_from_buffer = tf.cast(self.params['batch_size'] * percentage_from_buffer, tf.int32)
            if self.created_buffer.buffer_items < items_from_buffer:
                if not 'created_buffer_health' in self.warnings:
                    self.warnings['created_buffer_health'] = f'Created buffer too low! Requested {items_from_buffer} from {self.created_buffer.buffer_items}'
                items_from_buffer = self.created_buffer.buffer_items
            if items_from_buffer > 0:
                from_buffer = self.created_buffer.get(items_from_buffer)
                x_from_buffer = from_buffer[:, :-1]
                y_target_from_buffer = from_buffer[:, -1:]
                x = tf.concat([x_from_buffer, x[items_from_buffer:]], axis=0)
                y_target = tf.concat([y_target_from_buffer, y_target[items_from_buffer:]], axis=0)
            with self.tb_writer.as_default(step=self.total_steps):
                tf.summary.scalar('4extra/4created_buffer_percentage', percentage_from_buffer)
                tf.summary.scalar('4extra/4created_buffer_total_items', self.created_buffer.total_items)


        x_soft, real_positive_mask = self.encode_real(x)
        y_t = tf.squeeze(tf.cast(y_target == 2, tf.float32))    
        for c_step in range(self.params['gan_critic_steps']):
            # Real samples & class
            with tf.GradientTape() as real_tape:
                if self.latent_mode:
                    ν_real = self.embedder(x_soft, real_positive_mask, training='embed' in self.objectives)
                else:
                    ν_real = x_soft
                pred_raw = self.critic(ν_real, real_positive_mask, pe=pe, training=True)

                if 'gan' in self.objectives:
                    pred_logits_gan_real = pred_raw[:, 1]
                    pred_score_real = tf.nn.sigmoid(tf.clip_by_value(pred_logits_gan_real, -10., 10.))
                    if self.params['gan_critic_target_fn'] == 'logits':
                        critic_loss_real = pred_logits_gan_real
                        assert self.params['gan_critic_target_mode'] == 'direct'
                    elif self.params['gan_critic_target_fn'].startswith('sigmoid'):
                        critic_loss_real = pred_score_real
                    else:
                        critic_loss_real = 0 # ..
                    if self.params['gan_critic_target_fn'].endswith('log'):
                        critic_loss_real = tf.math.log(critic_loss_real)
                    critic_loss_real = - tf.reduce_mean(critic_loss_real)
                    crossentropy_loss = self.crossentropy(tf.ones_like(pred_logits_gan_real), tf.clip_by_value(pred_logits_gan_real, -10., 10.))
                    if self.params['gan_critic_target_fn'] == 'crossentropy':
                        critic_loss_real = crossentropy_loss
                else:
                    critic_loss_real = 0
                pred_logits_class_real = pred_raw[:, 0]
                if 'class' in self.objectives or self.inherent_class_loss:
                    class_loss_real = self.class_loss(y_t, tf.clip_by_value(pred_logits_class_real, -10., 10.))
                    critic_loss_real += class_loss_real * self.params['gan_objweight_class']
                if 'project' in self.objectives:
                    x_re_logits = self.projector(ν_real, real_positive_mask, x, training=True)
                    pb_loss = self.project_back_loss(x, x_re_logits, tf.cast(real_positive_mask, tf.float32))
                    critic_loss_real += self.params['gan_objweight_project_back'] * pb_loss

            critic_variables = self.critic.trainable_variables if critic_variables is None  else critic_variables
            _vars = critic_variables
            if 'embed' in self.objectives:
                assert self.latent_mode
                embedder_variables = self.embedder.trainable_variables if embedder_variables is None else embedder_variables
                _vars.extend(embedder_variables)
            if 'project' in self.objectives:
                projector_variables = self.projector.trainable_variables if projector_variables is None else projector_variables
                _vars.extend(projector_variables)
                increment(metrics, 'project_back_real', pb_loss)
            _grads = real_tape.gradient(critic_loss_real, _vars)
            critic_grads, _grads = _grads[:len(critic_variables)], _grads[len(critic_variables):]
            assert_finite(critic_grads, 'Critic real grads')
            if 'embed' in self.objectives:
                embedder_grads, _grads = _grads[:len(embedder_variables)], _grads[len(embedder_variables):]
            if 'project' in self.objectives:
                projector_grads, _grads = _grads[:len(projector_variables)], _grads[len(projector_variables):]
            assert len(_grads) == 0

            if 'class' in self.objectives or self.inherent_class_loss:
                increment(metrics, 'class_loss', class_loss_real)
                increment(metrics, 'class_acc', tf.keras.metrics.binary_accuracy(y_t, tf.nn.sigmoid(pred_logits_class_real)))
                real_class_prob = tf.nn.sigmoid(pred_logits_class_real)
                increment(metrics, 'class_entropy', -tf.reduce_mean(real_class_prob * tf.math.log(real_class_prob) + (1-real_class_prob) * tf.math.log(1-real_class_prob)))
                if self.params['gan_incremental_learning_mode']:
                    increment(metrics, 'class_acc_from_buffer', tf.keras.metrics.binary_accuracy(y_t[:items_from_buffer], tf.nn.sigmoid(pred_logits_class_real[:items_from_buffer])))
                    increment(metrics, 'class_acc_from_dataset', tf.keras.metrics.binary_accuracy(y_t[items_from_buffer:], tf.nn.sigmoid(pred_logits_class_real[items_from_buffer:])))

            if 'gan' in self.objectives:
                mean_pred_score_real = tf.reduce_mean(pred_score_real)
                if self.params['gan_critic_target_fn'] == 'logits':
                    mean_logits_real = tf.reduce_mean(pred_logits_gan_real)
                    increment(metrics, 'logits_real', mean_logits_real)
                else:
                    increment(metrics, 'score_real', mean_pred_score_real)
                    increment(metrics, 'crossentropy_real', crossentropy_loss)

            if 'gan' in self.objectives:
                # Generated samples
                if self.params['gan_copy_shape_critic']:
                    z, generated_positive_mask, gen_classes = self.input_noise(1, real_positive_mask, len_mode='copy', add_classes=self.generate_classes)
                else:
                    z, generated_positive_mask, gen_classes = self.input_noise(1, batch_size, len_mode='uniform', add_classes=self.generate_classes)
                ν_gen = self.generator(z, generated_positive_mask, pe=pe, training=True)

                # Replay buffer (unused)
                if self.params['gan_replay_buffer_fraction'] > 0:
                    assert pe is None
                    num_from_buffer = int(tf.cast(batch_size, tf.float32) * self.params['gan_replay_buffer_fraction'])
                    if self.replay_buffer.buffer_items >= num_from_buffer:
                        from_buffer, from_buffer_mask = self.replay_buffer.get(num_from_buffer)
                        from_buffer_mask = tf.cast(from_buffer_mask, tf.bool)
                        ν_gen_train = tf.concat([from_buffer, ν_gen[num_from_buffer:]], axis=0) # overwrite for now
                        gen_mask_train = tf.concat([from_buffer_mask, generated_positive_mask[num_from_buffer:]], axis=0)
                    else:
                        ν_gen_train = ν_gen
                        gen_mask_train = generated_positive_mask
                    self.replay_buffer.update(ν_gen, tf.cast(generated_positive_mask, tf.int32))
                else:
                    ν_gen_train = ν_gen
                    gen_mask_train = generated_positive_mask

                # Generated samples, critic training
                with tf.GradientTape() as fooled_tape:
                    pred_raw = self.critic(ν_gen_train, gen_mask_train, pe=pe, training=True)
                    pred_logits_gen = pred_raw[:, 1]
                    pred_score_gen = tf.nn.sigmoid(tf.clip_by_value(pred_logits_gen, -10., 10.))

                    if self.params['gan_critic_target_fn'] == 'logits':
                        critic_loss_gen = pred_logits_gen
                        assert self.params['gan_critic_target_mode'] == 'direct'
                    elif self.params['gan_critic_target_fn'].startswith('sigmoid'):
                        critic_loss_gen = pred_score_gen
                        if self.params['gan_critic_target_mode'] == 'one-minus':
                            critic_loss_gen = 1 - critic_loss_gen
                    else:
                        critic_loss_gen = 0 # ..
                    if self.params['gan_critic_target_fn'].endswith('log'):
                        critic_loss_gen = tf.math.log(critic_loss_gen)
                    if self.params['gan_critic_target_mode'] == 'one-minus':
                        critic_loss_gen = - critic_loss_gen # - nochmal wegen 1 - vorher
                    critic_loss_gen = tf.reduce_mean(critic_loss_gen)
                    crossentropy_loss = self.crossentropy(tf.zeros_like(pred_logits_gen), tf.clip_by_value(pred_logits_gen, -10., 10.))
                    if self.params['gan_critic_target_fn'] == 'crossentropy':
                        critic_loss_gen = crossentropy_loss

                    pred_logits_class_gen = pred_raw[:, 0]
                    if self.generate_classes:
                        class_loss_gen = self.class_loss(gen_classes, tf.clip_by_value(pred_logits_class_gen, -10., 10.))
                        critic_loss_gen += class_loss_gen * self.params['gan_objweight_genclass']
                    #TODO check how!!
                critic_grads_ = fooled_tape.gradient(critic_loss_gen, critic_variables, unconnected_gradients='none')
                assert_finite(critic_grads_, 'Critic generated grads')
                critic_grads = [a + b for a, b in zip(critic_grads, critic_grads_)]

                mean_logits_gen = tf.reduce_mean(pred_logits_gen)
                if self.params['gan_critic_target_fn'] == 'logits':
                    increment(metrics, 'logits_gen', tf.reduce_mean(mean_logits_gen))
                    increment(metrics, 'wasserstein', mean_logits_real - mean_logits_gen)
                mean_pred_score_gen = tf.reduce_mean(pred_score_gen)
                if self.params['gan_critic_target_fn'] != 'logits':
                    increment(metrics, 'score_gen', mean_pred_score_gen)
                    increment(metrics, 'crossentropy_gen', crossentropy_loss)
                    increment(metrics, 'crossentropy_genalt', self.crossentropy(tf.ones_like(pred_logits_gen), pred_logits_gen))

                if self.inherent_class_loss:
                    gen_class_prob = tf.nn.sigmoid(tf.clip_by_value(pred_logits_class_gen, -10., 10.))
                    if self.generate_classes:
                        increment(metrics, 'genclass_acc', tf.keras.metrics.binary_accuracy(gen_classes, gen_class_prob))
                    if self.generate_confusion:
                        increment(metrics, 'genclass_entropy', -tf.reduce_mean(gen_class_prob * tf.math.log(gen_class_prob) + (1-gen_class_prob) * tf.math.log(1-gen_class_prob)))


                # AE gen reverse (not implemented)
                # if self.latent_mode and self.params['gan_train_ae_gen_reverse']:
                #     with tf.GradientTape() as ae_tape:
                #         gen_re_logits = self.projector(v_gen, generated_positive_mask, x, training=True)
                #         gen_re_samples = self.sample_projection(gen_re_logits)
                #         v_gen_re = self.embedder(gen_re_samples, generated_positive_mask, training=True)
                #         sample_weight = tf.cast(real_positive_mask, tf.float32)
                #         sample_weight /= tf.reduce_sum(sample_weight, axis=-1, keepdims=True)
                #         ae_gen_reverse_loss = self.mse_loss(v_gen, v_gen_re, sample_weight) #/ tf.cast(100, tf.float32)
                #     ae_variables = self.projector.trainable_variables
                #     ae_grad = ae_tape.gradient(ae_gen_reverse_loss, ae_variables)
                #     self.opti_aux.apply_gradients(zip(ae_grad, ae_variables))
                #     metrics['ae_gen_reverse'] += ae_gen_reverse_loss


                if self.params['gan_intgrad_method'] == 'none':
                    assert self.params['gan_gradient_penalty'] == 0
                else:
                    gen_seq_len = tf.shape(ν_gen)[1]
                    len_diff = (seq_len - gen_seq_len).numpy()
                    assert len_diff == 0
                    assert tf.math.reduce_all(real_positive_mask == generated_positive_mask)
                    assert self.params['gan_copy_shape_critic']

                    # Uniform line samples
                    eps_lines = tf.random.uniform((batch_size,), 0, 1, dtype=tf.float32)[:, tf.newaxis, tf.newaxis]
                    ν_interleaved = eps_lines * ν_real + (1 - eps_lines) * ν_gen
                    interleaved_mask = tf.logical_and(real_positive_mask, generated_positive_mask)

                    def calc_input_gradients(ν_interleaved):
                        with tf.GradientTape(watch_accessed_variables=False) as inner_penalty_tape:
                            inner_penalty_tape.watch(ν_interleaved)
                            pred_raw = self.critic(ν_interleaved, interleaved_mask, training=True)
                            pred_interleaved = pred_raw[:, 1]
                            # note: no mean here, since each output only depends on one input. Gradient is for each input separately.
                        grad_interleaved = inner_penalty_tape.gradient(pred_interleaved, ν_interleaved)
                        len_grad = tf.reduce_sum(grad_interleaved**2, axis=-1)
                        len_grad *= tf.cast(interleaved_mask, tf.float32)
                        len_grad = tf.math.sqrt(tf.reduce_sum(len_grad, axis=-1)) # one batch of scalars
                        return len_grad

                    if self.params['gan_gradient_penalty'] > 0:
                        with tf.GradientTape() as penalty_tape:
                            len_grad = calc_input_gradients(ν_interleaved)
                            loss_gradient_penalty = self.params['gan_gradient_penalty'] * (len_grad - self.params['gan_intgrad_target']) ** 2
                            loss_gradient_penalty = tf.reduce_mean(loss_gradient_penalty)
                        critic_grads_ = penalty_tape.gradient(loss_gradient_penalty, critic_variables, unconnected_gradients='zero') # careful!
                        assert_finite(critic_grads_, 'Critic GP grads')
                        critic_grads = [a + b for a, b in zip(critic_grads, critic_grads_)]
                    else:
                        len_grad = calc_input_gradients(ν_interleaved)
                    if self.params['gan_intgrad_method'] == 'uniform':
                        increment(metrics, 'intgrad_len_uniform', tf.reduce_mean(len_grad))
                # -- end GAN critic objective

            ## Apply gradients!
            if critic_variables is not None:
                self.opti_c.apply_gradients(zip(critic_grads, critic_variables))
            if embedder_variables is not None:
                self.opti_emb.apply_gradients(zip(embedder_grads, embedder_variables))
            if projector_variables is not None:
                self.opti_proj.apply_gradients(zip(projector_grads, projector_variables))
            # -- end critic training loop
        for k in metrics:
            metrics[k] /= self.params['gan_critic_steps']


        # Generator training #
        if self.params['gan_copy_shape_generator']:
            z, generated_positive_mask, gen_classes = self.input_noise(1, real_positive_mask, len_mode='copy', add_classes=self.generate_classes)
        else:
            z, generated_positive_mask, gen_classes = self.input_noise(1, batch_size, add_classes=self.generate_classes)

        if 'gan' in self.objectives:
            with tf.GradientTape() as generator_tape:
                ν_gen = self.generator(z, generated_positive_mask, pe=pe, training=True)
                pred_raw = self.critic(ν_gen, generated_positive_mask, pe=pe, training=True)
                pred_logits_gen = pred_raw[:, 1]
                pred_score_gen = tf.nn.sigmoid(pred_logits_gen)
                mean_pred_score_gen = tf.reduce_mean(pred_score_gen)

                if self.params['gan_generator_target_fn'] == 'logits':
                    generator_loss = pred_logits_gen
                    assert self.params['gan_generator_target_mode'] == 'direct'
                elif self.params['gan_generator_target_fn'].startswith('sigmoid'):
                    generator_loss = pred_score_gen
                    if self.params['gan_generator_target_mode'] == 'one-minus':
                        generator_loss = 1 - generator_loss
                else:
                    generator_loss = 0
                if self.params['gan_generator_target_fn'].endswith('log'):
                    generator_loss = tf.math.log(generator_loss)
                if self.params['gan_generator_target_mode'] == 'direct':
                    generator_loss = - generator_loss
                generator_loss = tf.reduce_mean(generator_loss)
                crossentropy_loss = self.crossentropy(tf.ones_like(pred_logits_gen), pred_logits_gen)
                if self.params['gan_generator_target_fn'] == 'crossentropy':
                    generator_loss = crossentropy_loss
                pred_logits_class_gen = pred_raw[:, 0]
                if self.generate_classes:
                    gen_class_loss = self.class_loss(gen_classes, pred_logits_class_gen)
                    generator_loss += self.params['gan_objweight_genclass'] * gen_class_loss # TODO check how!!
                if self.generate_confusion and self.total_steps >= self.params['gan_delay_confusion_steps']:
                    gen_class_prob = tf.nn.sigmoid(pred_logits_class_gen)
                    gen_class_neg_entropy = gen_class_prob * tf.math.log(tf.clip_by_value(gen_class_prob, 1e-20, 1))
                    gen_class_neg_entropy += (1-gen_class_prob) * tf.math.log(tf.clip_by_value(1-gen_class_prob, 1e-20, 1))
                    if self.params['gan_confusion_loss'] == 'entropy':
                        gen_class_confusion_loss = gen_class_neg_entropy
                    elif self.params['gan_confusion_loss'] == 'mse':
                        gen_class_confusion_loss = pred_logits_class_gen ** 2
                    elif self.params['gan_confusion_loss'] == 'mae':
                        gen_class_confusion_loss = tf.math.abs(pred_logits_class_gen)
                    generator_loss += self.params['gan_objweight_confusion'] * tf.reduce_mean(gen_class_confusion_loss)

            generator_variables = self.generator.trainable_variables
            generator_grads = generator_tape.gradient(generator_loss, generator_variables)
            assert_finite(generator_grads, 'Generator grads')
            self.opti_g.apply_gradients(zip(generator_grads,  generator_variables))
            if self.params['gan_generator_target_fn'] != 'logits':
                metrics['score_gen_alt'] = mean_pred_score_gen
            metrics['min_logits_gen'] = tf.reduce_min(pred_logits_gen)


            # Analyze generated
            if (self.total_steps % self.params['gan_trainsteps_infer_interval'] == 0):
                num_analyze = min(tf.shape(ν_gen)[0].numpy(), 150)
                generated_tokens, ana, _ = self.get_predictions(num_analyze, ν_gen, generated_positive_mask)
                ana.update(self.analyze_generated(generated_tokens))
                if not self.latent_mode:
                    # full batch
                    generated_hards = proc_logits(ν_gen, generated_positive_mask, sample=True, tau=0, calc_entropy=False)
                    hard_gen_soft, _ = self.encode_real(generated_hards)
                    pred_hard_gen = self.critic(hard_gen_soft, generated_positive_mask, pe=pe, training=True) # training=True damit ähnliche scores
                    pred_gan_logits_hard_gen = pred_hard_gen[:, 1]
                    if self.params['gan_generator_target_fn'] == 'logits':
                        ana['logits_gen_hard'] = tf.reduce_mean(pred_gan_logits_hard_gen)
                    else:
                        ana['score_gen_hard'] = tf.reduce_mean(tf.nn.sigmoid(pred_gan_logits_hard_gen))
                    if self.generate_confusion:
                        hard_class_logits = pred_hard_gen[:, 0]
                        hard_class_prob = tf.nn.sigmoid(tf.clip_by_value(hard_class_logits, -10., 10.))
                        increment(metrics, 'genclass_logits_hard_mean', tf.reduce_mean(hard_class_logits))
                        entropies = -(hard_class_prob * tf.math.log(hard_class_prob) + (1-hard_class_prob) * tf.math.log(1-hard_class_prob))
                        ana['genclass_entropy_hard'] = tf.reduce_mean(entropies)
                ana = {k:v for k,v in ana.items() if v is not None}
                self.last_ana = ana
            else:
                num_analyze = 0
            metrics.update(self.last_ana)

            if (self.epoch_steps % 50 == 49):
                num_print = 3
                generated_tokens, ana, generated_hards = self.get_predictions(num_print, ν_gen, generated_positive_mask)
                join_str = ' ' if self.params['problem'] == 'math' else ''
                print(f'step {self.epoch_steps+1:3d}: ' + ', '.join([join_str.join(q) for q in generated_tokens]))

            # -- end GAN generator objective

        if not 'gan' in self.objectives and self.params['gan_process_generated_samples']:
            ν_gen = self.generator(z, generated_positive_mask, pe=pe, training=True)

        if (self.total_steps % self.params['gan_trainsteps_process_interval'] == 0) and self.params['gan_process_generated_samples']:
            generated_hards = proc_logits(ν_gen, generated_positive_mask, sample=True, tau=0, calc_entropy=False)
            hard_gen_soft, _ = self.encode_real(generated_hards)
            pred_hard_gen = self.critic(hard_gen_soft, generated_positive_mask, pe=pe, training=True) # training=True damit ähnliche scores
            hard_class_prob = tf.nn.sigmoid(pred_hard_gen[:, 0])
            entropies = -(hard_class_prob * tf.math.log(hard_class_prob) + (1-hard_class_prob) * tf.math.log(1-hard_class_prob))
            candidate_mask = entropies > self.params['gan_filter_generated_entropy_threshold']
            candidate_hards, candidate_positive_mask = tf.boolean_mask(generated_hards, candidate_mask), tf.boolean_mask(generated_positive_mask, candidate_mask)
            entropies = tf.boolean_mask(entropies, candidate_mask)
            candidate_tokens = self.hard_decode(candidate_hards, candidate_positive_mask)
            if self.params['gan_save_candidate_samples']:
                candidate_strings = [''.join(q) for q in candidate_tokens]
                with open(os.path.join(self.params['job_dir'], self.params['run_name'], 'generated_samples.txt'), 'a') as save_file:
                    save_file.write('\n'.join(candidate_strings))
            solved_indices_sat, _, solved_indices_unsat, _ = get_corrects(candidate_tokens, self.params, self.total_steps, entropies=entropies.numpy())
            if self.params['gan_incremental_learning_mode']:
                solved_indices_sat = tf.stack(solved_indices_sat)
                solved_indices_unsat = tf.stack(solved_indices_unsat)
                labels = tf.concat([tf.ones((len(solved_indices_sat),), dtype=tf.int32) * 2, tf.ones((len(solved_indices_unsat),), dtype=tf.int32) * 1], axis=0) # hardcoded :/
                solved_indices = tf.concat([solved_indices_sat, solved_indices_unsat], axis=0)
                if solved_indices.shape[0] > 0: # if we have any indices at all
                    candidate_hards = tf.where(candidate_positive_mask, candidate_hards, tf.ones_like(candidate_hards, dtype=tf.int32)* self.params['input_pad_id'])
                    to_save = tf.concat([tf.gather(candidate_hards, solved_indices, axis=0), labels[:, tf.newaxis]], axis=1) # save label and input together :)
                    self.created_buffer.update(to_save)


        with self.tb_writer.as_default(step=self.total_steps): #pylint:disable=not-context-manager
            name_mapping = {'score_real' : '1critic/1score_real', 'score_gen' : '1critic/1score_gen', 'class_acc' : '1critic/2class_acc',
                'intgrad_len_uniform' : '1critic/3intgrad_len_uniform', 
                'seq_entropy' : '1generator/1seq_entropy', 'parse_fragments' : '1generator/2parse_fragments', 'parse_coverage' : '1generator/3parse_coverage',
                'fully_correct' : '1generator/3fully_correct', 'soft_entropy' : '1generator/4soft_entropy', 'genclass_acc' : '1generator/5genclass_acc',
                'genclass_entropy' : '1generator/5genclass_entropy', 
                'project_back_real' : '2autoenc/1real_projback', 'score_gen_alt' : '1generator/1score_gen',
                'crossentropy_real' : '4extra/1ce_real', 'crossentropy_gen' : '4extra/2ce_gen', 'crossentropy_genalt' : '4extra/3ce_genalt',
                'wasserstein' : '1critic/1wasserstein', 'logits_real' : '1critic/1logits_real', 'logits_gen' : '1critic/1logits_gen',
                'class_entropy' : '1critic/2class_entropy', 'score_gen_hard' : '1generator/6score_gen_hard', 'logits_gen_hard' : '1generator/6logits_gen_hard',
                'genclass_entropy_hard' : '1generator/5genclass_entropy_hard',
                'genclass_logits_hard_mean' : '1generator/5genclass_logits_hard',
                'class_acc_from_buffer' : '4extra/6class_acc_from_buffer', 'class_acc_from_dataset' : '4extra/6class_acc_from_dataset', 'min_logits_gen' : '4extra/7min_logits_gen',
                'class_loss' : '4extra/8class_loss_real'
                }
            for k, v in name_mapping.items():
                if k in metrics:
                    tf.summary.scalar(v, metrics[k])

        proforma_loss = 0 # TODO
        metrics['loss'] = proforma_loss
        if 'soft_entropy' in metrics and metrics['soft_entropy'] == 0:
            del metrics['soft_entropy']

        self.epoch_steps += 1
        self.total_steps += 1
        return metrics        




    def test_step(self, data):
        x, y_target = data
        if self.params['tree_pe']:
            x, pe, _ = x
        else:
            x, _ = x
            pe = None
        batch_size = tf.shape(x)[0]
        res = {}

        # Real (class)
        x_soft, x_mask = self.encode_real(x) # x_mask = x != self.params['input_pad_id']
        y_t = tf.squeeze(tf.cast(y_target == 2, tf.float32))
        if self.latent_mode:
            ν_real = self.embedder(x_soft, x_mask, training=False) # lol?
        else:
            ν_real = x_soft
        pred_raw = self.critic(ν_real, x_mask, pe=pe, training=False)

        if 'class' in self.objectives or self.inherent_class_loss:
            res['class_acc'] = tf.keras.metrics.binary_accuracy(y_t, tf.nn.sigmoid(pred_raw[:, 0]))
            if self.latent_mode: # proj back
                x_re_logits = self.projector(ν_real, x_mask, x, training=False)
                res['project_back_real'] = self.project_back_loss(x, x_re_logits, tf.cast(x_mask, tf.float32))
                if self.test_steps == 0:
                    stuff, stuff_scores, soft_entropy = self.projector.infer(ν_real, x_mask, training=False)
                    if soft_entropy is not None:
                        res['soft_entropy'] = soft_entropy
                    print('pred  ', ', '.join(self.hard_decode(stuff[:3], x_mask[:3], full=False, as_list=False)))
                    print('should', ', '.join([self.vocab.decode(list(q), full=False) for q in x[:3]]))

        if 'gan' in self.objectives:
            pred_real = tf.nn.sigmoid(pred_raw[:, 1])
            res['score_real'] = tf.reduce_mean(pred_real)
            # Generated
            if self.params['gan_copy_shape_val']:
                z, generated_positive_mask, gen_classes = self.input_noise(1, x_mask, len_mode='copy', add_classes=self.generate_classes)
            else:
                z, generated_positive_mask, gen_classes = self.input_noise(1, batch_size, add_classes=self.generate_classes)
            ν_gen = self.generator(z, generated_positive_mask, pe=pe, training=True) # keep training=True
            predictions_gen_raw = self.critic(ν_gen, generated_positive_mask, pe=pe, training=False)
            predictions_gen = tf.nn.sigmoid(predictions_gen_raw[:, 1])
            res['score_gen'] = tf.reduce_mean(predictions_gen)

            # Analysis
            num_analyze = min(200, batch_size.numpy())
            generated_tokens, ana, generated_hards = self.get_predictions(num_analyze, ν_gen, generated_positive_mask)
            ana.update(self.analyze_generated(generated_tokens))
            res.update(ana)
            if self.test_steps == 0:
                join_str = ' ' if self.params['problem'] == 'math' else ''
                print(f'test: ' + ', '.join([join_str.join(q) for q in generated_tokens[:6]]))


        # "loss" for checkpoint name
        if 'class' in self.objectives and not 'gan' in self.objectives:
            res['loss'] = self.class_loss(y_t, pred_raw[:, 0])
        elif 'gan' in self.objectives:
            res['loss'] = - tf.reduce_mean(tf.math.log(pred_real)) - tf.reduce_mean(tf.math.log(1 - predictions_gen))

        self.test_steps += 1
        with self.tb_writer.as_default(step=self.total_steps): #pylint:disable=not-context-manager
            if 'class' in self.objectives or self.inherent_class_loss:
                tf.summary.scalar('1critic/2class_acc_val', res['class_acc'])
        return res


    def analyze_generated(self, token_things):
        res = {}
        num_analyze = len(token_things)
        hard_entropy, fragments, coverage, fully_correct = 0, 0, 0, 0
        for q in token_things:
            if self.params['problem'] == 'ltl':
                e, f, c = parse_score(''.join(q))
            elif self.params['problem'] == 'math':
                f, c = parse_score_math(q)
                e = 0
            if e is None:
                num_analyze -= 1
                continue
            hard_entropy += e
            fragments += f
            coverage += c
            if f == 1 and c == 1.:
                fully_correct += 1
        res['seq_entropy'] = hard_entropy / num_analyze if num_analyze > 0 else None
        res['parse_fragments'] = fragments / num_analyze if num_analyze > 0 else None
        res['parse_coverage'] = coverage / num_analyze if num_analyze > 0 else None
        res['fully_correct'] = fully_correct / num_analyze if num_analyze > 0 else None
        return res


    def hard_decode(self, x, pos_mask, full=True, as_list=True):
        return [self.vocab.decode(list(q[:tf.reduce_sum(tf.cast(m, tf.int32)).numpy()].numpy()), full=full, as_list=as_list) for q, m in zip(x, pos_mask)]


    def get_predictions(self, num, ν_gen, generated_positive_mask):
        ana = {}
        if self.latent_mode:
            generated_hards, _, soft_entropy = self.projector.infer(ν_gen[:num], generated_positive_mask[:num], training=False)
        else:
            generated_hards, soft_entropy = proc_logits(ν_gen, generated_positive_mask, sample=True, tau=0, calc_entropy=True)
        ana['soft_entropy'] = soft_entropy
        generated_tokens = self.hard_decode(generated_hards, generated_positive_mask[:num])
        return generated_tokens, ana, generated_hards


    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_steps = 0
        self.test_steps = 0
        self.epoch = epoch


    def on_epoch_end(self, epoch, logs=None):
        print(self.total_steps, 'steps so far')
        for k, v in self.warnings.items():
            print('Warning:', v)
        self.warnings = {}



def proc_logits(logits, mask=None, normalize=False, sample=False, tau=1, reduce=True, calc_entropy=False):
    if calc_entropy:
        if normalize:
            softs = tf.nn.softmax(logits)
        else:
            softs = logits
        w = tf.cast(mask, tf.float32)
        ent_per_pos = -tf.reduce_sum(softs * tf.math.log(softs), axis=-1)
        soft_entropy = tf.reduce_mean(tf.reduce_sum(ent_per_pos * w / tf.reduce_sum(w, axis=1, keepdims=True), axis=1))
    if normalize:
        x = tf.nn.softmax(logits / tau)
    else:
        x = logits
    if not sample:
        return (x, soft_entropy) if calc_entropy else x
    if tau != 0:
        raise NotImplementedError
    else: # tau == 0
        res = tf.argmax(x, axis=-1, output_type=tf.dtypes.int32) # int32?
    if not reduce:
        res = tf.one_hot(res, tf.shape(logits)[-1])
    return (res, soft_entropy) if calc_entropy else res


def assert_finite(values, message, info=None):
    if ASSERT_FINITE:
        if not all([tf.reduce_all(tf.math.is_finite(q)) for q in values]):
            print('--------------------- FINITE ASSERTION FAILED :', message)
            if info is not None:
                for k, v in info.items():
                    print(k, v)
            assert False
