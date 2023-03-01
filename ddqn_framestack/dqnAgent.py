from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from collections import deque
import time, random
from modifiedTensorboard import ModifiedTensorBoard
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

import numpy as np

REPLAY_MEMORY_SIZE = 5000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 10
DISCOUNT = 0.99
MODEL_NAME = "CARLA_DoubleDQN"

# Digunakan jika mau melanjutkan training model
class DQNAgent:

    def __init__(self, action_space_size, observation_space_dim, epstart, model_path, logfile):
        self.action_space_size = action_space_size
        self._logfile = logfile
        self._logfile.write("halo ini dari dqn")
        self.model_name = MODEL_NAME
        self.OBSERVATION_SPACE_DIMS = observation_space_dim

        if(epstart > 0): 
            self.model = load_model("modelhandoko/"+model_path)
        else:
            self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # Kita melatih data acak dari replay memory
        self.replay_memory = deque(maxlen = REPLAY_MEMORY_SIZE)
        now = time.strftime('%Y_%m_%d %H_%M_%S', time.localtime(int(time.time())))
        self.tensorboard = ModifiedTensorBoard(tf, log_dir=f"logs/{MODEL_NAME}")

        # Variable untuk menelusuri kapan waktunya untuk update target model
        self.target_update_counter = 0

        self.last_logged_episode = 0
    
    def create_model(self):
        model = Sequential()

        model.add(Conv2D(32, (5, 5), input_shape=self.OBSERVATION_SPACE_DIMS))  
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))

        model.add(Dense(self.action_space_size, activation='softmax'))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Method untuk mendapatkan q value untuk prediksi
    def get_qs(self,state):
        self._logfile.write(f"GET QS SHAPE STATE = {state.shape}")
        test = np.array(state).reshape(-1, *state.shape)/255
        self._logfile.write(f"GET QS PREDICT STATE = {test.shape}")
        predict = self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
        self._logfile.write(f"PREDICT Q VALUE = {predict}")
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
    
    def train(self):
        print("!!!!ini kita train loh!!!!")
        # Jika kita tidak memiliki sampel maka tidak lakukan apa2
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # saat kita mendapatkan minibatch maka kita ambil nilai q saat ini dan dimasa depan
        # transisi adalah transition = (current_state, action, reward, new_state, done)
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)
        
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        # Kita buat input X dan output y
        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                # max_future_q = np.max(future_qs_list[index])
                # new_q = reward + DISCOUNT * max_future_q
                # nst_predict_target = self.model_target.predict(nst) #Predict from the TARGET
                # dimana future_qs_list = nst_predict_target
                # nst_action_predict_target = nst_predict_target[index]
                # nst_predict = self.model.predict(nst)
                # nst_action_predict_model = nst_predict[index]
                # target = reward + self.gamma * nst_action_predict_target[np.argmax(nst_action_predict_model)]
                max_future_q = future_qs_list[index]
                new_q = reward + DISCOUNT * max_future_q[np.argmax(current_qs_list[index])]
            else:
                new_q = reward
            
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)
        
        # Kita ingin log per episode, tidak setiap training step
        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_logged_episode = self.tensorboard.step
        
        # Melakukan model fitting
        self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose = 0, shuffle = False, callbacks=[self.tensorboard] if log_this_step else None)
        
        # Jika log_this_step true maka lakukan tracking
        if log_this_step:
            self.target_update_counter += 1
        
        # Selanjutnya kita ingin mengecek apakah sudah saatnya melakukan update target model
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0