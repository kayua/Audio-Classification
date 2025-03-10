import aim
import tensorflow


import aim
import os

class AimCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, run_name="experiment", repo_path=None):
        super().__init__()
        if repo_path is None:
            repo_path = os.path.abspath("/home/kayua/Área de Trabalho/Mosquitoes-Classification-Models/.aim")
        self.run = aim.Run(run_name, repo=repo_path)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            self.run.track(value, name=key, step=epoch)