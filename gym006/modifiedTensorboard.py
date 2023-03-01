from keras.callbacks import TensorBoard
import tensorflow as tf
import os

class ModifiedTensorBoard(TensorBoard):

    # Overiding init untuk mengatur initial step dan writer dimana kita menginginkan satu log file untuk semua kelas fit()
    def __init__(self,tf, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir
    
    # Overiding method ini untuk berhenti membuat log writer default
    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False
    
    # Overrided, saves log dengan step number yang kita set (jika tidak setiap fit() akan memulai menulis di step ke 0)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)
    
    #Overrided, kita melatih untuk satu batch saja, tidak perlu untuk save semuanya di akhir
    def on_batch_end(self,batch, logs=None):
        pass
    
    #Overrided, sehingga tidak akan menutup writer
    def on_train_end(self, _):
        pass

    # custom method. Kita buat writer yang menulis metrics
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)
    
    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()