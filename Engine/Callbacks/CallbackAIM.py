import aim
import tensorflow


class AimCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, run_name="experiment"):
        super().__init__()
        self.run = aim.Run(run_name)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            self.run.track(value, name=key, step=epoch)