import tensorflow as tf

class StoppedPercentMetric(tf.keras.metrics.Metric):
    def __init__(self):
        super().__init__(name='stopped_pct')
        self.n_updates = self.add_weight('n_updates')
        self.n_updates.assign(0)
        self.accu_percent = self.add_weight('accu_percent')
        self.accu_percent.assign(0)

    def update_state(self, *args, **kwargs):
        if 'percent_stopped' in kwargs:
            self.accu_percent.assign_add(kwargs['percent_stopped'])
            self.n_updates.assign_add(1)
    
    def result(self):
        return self.accu_percent / self.n_updates


class DetailedTFCallback(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir, log_lr=True, log_target_iterations=False, **kwargs):
        super().__init__(log_dir, write_graph=False, **kwargs)
        self.log_lr = log_lr
        self.log_target_iterations = log_target_iterations

    def on_epoch_end(self, epoch, logs=None): #thanks https://stackoverflow.com/questions/49127214/keras-how-to-output-learning-rate-onto-tensorboard
        logs = logs or {}
        last_step = self.model.optimizer.iterations.numpy() - 1 # the step the last training step of this epoch ran with
        if self.log_lr:
            if hasattr(self.model.optimizer.lr, '__call__'):
                logs['lr'] = self.model.optimizer.lr(last_step)
            else:
                logs['lr'] = self.model.optimizer.lr
        if self.log_target_iterations:
            logs['target_iterations'] = self.model.get_iterations(step=last_step, epoch=epoch)

        super().on_epoch_end(epoch, logs)


class ProxyCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.on_epoch_begin(epoch, logs)
    def on_epoch_end(self, epoch, logs=None):
        self.model.on_epoch_end(epoch, logs)

