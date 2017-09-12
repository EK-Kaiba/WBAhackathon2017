# coding: utf-8

import random
import numpy as np
from chainer import cuda

class Experience:
    def __init__(self, use_gpu=0, data_size=10**5, replay_size=32, hist_size=1, initial_exploration=10**3, dim=10240):

        self.use_gpu = use_gpu
        self.data_size = data_size
        self.replay_size = replay_size
        self.hist_size = hist_size
        # self.initial_exploration = 10
        self.initial_exploration = initial_exploration
        self.dim = dim

        self.d = [np.zeros((self.data_size, self.hist_size, self.dim), dtype=np.uint8),
                  np.zeros(self.data_size, dtype=np.uint8),
                  np.zeros((self.data_size, 1), dtype=np.int8),
                  np.zeros((self.data_size, self.hist_size, self.dim), dtype=np.uint8),
                  np.zeros((self.data_size, 1), dtype=np.bool)]

    def stock(self, time, state, action, reward, state_dash, episode_end_flag):
        data_index = time % self.data_size

        if episode_end_flag is True:
            self.d[0][data_index] = state
            self.d[1][data_index] = action
            self.d[2][data_index] = reward
        else:
            self.d[0][data_index] = state
            self.d[1][data_index] = action
            self.d[2][data_index] = reward
            self.d[3][data_index] = state_dash
        if self.d[4][data_index]:
            print('##################################')
            print('True!!! data_index = %s' % data_index)
            for d in self.d[4][:data_index + 1]:
                print('%s' % d),
            print('\n##################################')
        else:
            self.d[4][data_index] = episode_end_flag

    def retrieve_sequence_replay_index(self, length):
        print('length = %s' % length)
        tmp_episode_end_flags = self.d[4].T[0].copy()
        print('tmp_episode_end_flags = %s' % tmp_episode_end_flags)
        indices = [i for i in range(len(tmp_episode_end_flags)) if tmp_episode_end_flags[i]]
        print('indices = %s' % indices)

        while True:
            selected_end_index_of_indices = random.randint(1, len(indices) - 1)
            print('selected_end_index_of_indices = %s' % selected_end_index_of_indices)
            if (indices[selected_end_index_of_indices] - self.replay_size + 1) - (indices[selected_end_index_of_indices - 1] + 1) >= 0:
                print('broken')
                break

        selected_start_index = random.randint(indices[selected_end_index_of_indices - 1] + 1, indices[selected_end_index_of_indices] - self.replay_size + 1)
        print('selected_start_index = %s' % selected_start_index)

        a = np.array([range(selected_start_index, selected_start_index + self.replay_size)]).T
        print(a)
        return a


    def replay(self, time):
        replay_start = False
        if self.initial_exploration < time:
            replay_start = True
            # Pick up replay_size number of samples from the Data
            if time < self.data_size:  # during the first sweep of the History Data
                #replay_index = np.random.randint(0, time, (self.replay_size, 1))
                replay_index = self.retrieve_sequence_replay_index(time)
            else:
                #replay_index = np.random.randint(0, self.data_size, (self.replay_size, 1))
                replay_index = self.retrieve_sequence_replay_index(self.data_size)

            s_replay = np.ndarray(shape=(self.replay_size, self.hist_size, self.dim), dtype=np.float32)
            a_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.uint8)
            r_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.float32)
            s_dash_replay = np.ndarray(shape=(self.replay_size, self.hist_size, self.dim), dtype=np.float32)
            episode_end_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.bool)
            for i in xrange(self.replay_size):
                s_replay[i] = np.asarray(self.d[0][replay_index[i]], dtype=np.float32)
                a_replay[i] = self.d[1][replay_index[i]]
                r_replay[i] = self.d[2][replay_index[i]]
                s_dash_replay[i] = np.array(self.d[3][replay_index[i]], dtype=np.float32)
                episode_end_replay[i] = self.d[4][replay_index[i]]

            if self.use_gpu >= 0:
                s_replay = cuda.to_gpu(s_replay)
                s_dash_replay = cuda.to_gpu(s_dash_replay)

            return replay_start, s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay

        else:
            return replay_start, 0, 0, 0, 0, False

    def end_episode(self, time, last_state, action, reward):
        self.stock(time, last_state, action, reward, last_state, True)
        replay_start, s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay = \
            self.replay(time)

        return replay_start, s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay
