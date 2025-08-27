import tensorflow as tf
import datetime

class TensorBoardLogger:
    def __init__(self, log_dir_prefix="logs/gradient_tape/", name=None):
        log_dir = log_dir_prefix + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + name
        self.summary_writer = tf.summary.create_file_writer(log_dir)

    def log_scalar(self, tag, value, step):
        with self.summary_writer.as_default():
            tf.summary.scalar(tag, value, step=step)

    def log_histogram(self, tag, values, step):
        with self.summary_writer.as_default():
            tf.summary.histogram(tag, values, step=step)

    def log_image(self, tag, image, step):
        with self.summary_writer.as_default():
            tf.summary.image(tag, image, step=step)

    def flush(self):
        self.summary_writer.flush()