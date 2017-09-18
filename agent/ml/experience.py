# coding: utf-8

import random
import multiprocessing
import numpy as np
from chainer import cuda
import utils

def retrieve_sequence_replay_index(d, length):
    tmp_episode_end_flags = d[4].T[0].copy()
    indices = [i for i in range(len(tmp_episode_end_flags)) if tmp_episode_end_flags[i]]
    indices.insert(0, -1)

    tmp_rewards = d[2].T[0].copy()

    while True:
        selected_end_index_of_indices = random.randint(1, len(indices) - 1)
        replay_size = indices[selected_end_index_of_indices] - indices[selected_end_index_of_indices - 1]
        if replay_size > 0:
            break

    selected_start_index = indices[selected_end_index_of_indices - 1] + 1

    a = [selected_start_index, selected_start_index + replay_size] # left is inclusive, right is exclusive.
    return a

def experience_replay(queue, d, batch_size, time, dim, data_size=10**5):
    print('experience_replay is started.')
    replay_start = True
    replay_indices = []
    replay_sizes = []
    replay_max_size = 0
    for i in xrange(batch_size):
        # Pick up replay_size number of samples from the Data
        if time < data_size:  # during the first sweep of the History Data
            replay_indices.append(retrieve_sequence_replay_index(d, time))
        else:
            replay_indices.append(retrieve_sequence_replay_index(d, data_size))
        tmp_replay_size = replay_indices[-1][1] - replay_indices[-1][0]
        if replay_max_size < tmp_replay_size:
            replay_max_size = tmp_replay_size
        replay_sizes.append(tmp_replay_size)

    s_replay = -np.ones(shape=(replay_max_size, batch_size, dim), dtype=np.float32)
    a_replay = np.ones(shape=(replay_max_size, batch_size), dtype=np.uint8)
    r_replay = np.zeros(shape=(replay_max_size, batch_size), dtype=np.float32)
    s_dash_replay = -np.ones(shape=(replay_max_size, batch_size, dim), dtype=np.float32)
    episode_end_replay = np.zeros(shape=(replay_max_size, batch_size), dtype=np.bool)
    for j in xrange(batch_size):
        s_replay[:, j] = np.append(np.asarray(d[0][replay_indices[j][0]:replay_indices[j][1]])[:,0,:], -np.ones((replay_max_size - replay_sizes[j], 10240), dtype=np.float32), axis=0)
        a_replay[:, j] = np.append(d[1][replay_indices[j][0]:replay_indices[j][1]], np.ones(replay_max_size - replay_sizes[j], dtype=np.uint8))
        r_replay[:, j] = np.append(d[2][replay_indices[j][0]:replay_indices[j][1]], np.zeros(replay_max_size - replay_sizes[j], dtype=np.float32))
        s_dash_replay[:, j] = np.append(np.array(d[3][replay_indices[j][0]:replay_indices[j][1]])[:,0,:], -np.ones((replay_max_size - replay_sizes[j], 10240), dtype=np.float32), axis=0)
        episode_end_replay[:, j] = np.append(d[4][replay_indices[j][0]:replay_indices[j][1]], np.zeros(replay_max_size - replay_sizes[j], dtype=np.bool))

    print('experience_replay is ended:)')
    queue.put([s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay])
    print('experience_replay is truely ended, i.e. queue.put is also ended.')


class Experience:
    RIPPLING_MAX_TIME = 50
    BATCH_SIZE = 1
    def __init__(self, use_gpu=0, data_size=10**5, replay_size=32, hist_size=1, initial_exploration=10**3, dim=10240):

        self.use_gpu = use_gpu
        self.data_size = data_size
        self.replay_size = replay_size
        self.hist_size = hist_size
        # self.initial_exploration = 10
        self.initial_exploration = initial_exploration
        self.dim = dim

        self.is_ripple_firing = False

        self.d = [np.zeros((self.data_size, self.hist_size, self.dim), dtype=np.uint8),
                  np.zeros(self.data_size, dtype=np.uint8),
                  np.zeros((self.data_size, 1), dtype=np.int8),
                  np.zeros((self.data_size, self.hist_size, self.dim), dtype=np.uint8),
                  np.zeros((self.data_size, 1), dtype=np.bool)]

        self.process = None
        manager = multiprocessing.Manager()
        self.queue = manager.Queue()

    def stock(self, time, state, action, reward, state_dash, episode_end_flag):
        data_index = time % self.data_size

        #if self.d[2][data_index][0] > 0.9:
        #    print('%s !!! data_index = %s' % (self.d[2][data_index][0], data_index,))

        if self.d[4][data_index]:
            print('##################################')
            print('True!!! data_index = %s' % data_index)
            for d in self.d[4][:data_index + 1]:
                print('%s' % d),
            print('')
            for r in self.d[2][:data_index + 1]:
                print('%s' % r),
            print('\n##################################')
        else:
            self.d[4][data_index] = episode_end_flag

            if episode_end_flag is True:
                self.d[0][data_index] = state
                self.d[1][data_index] = action
                self.d[2][data_index] = reward
            else:
                self.d[0][data_index] = state
                self.d[1][data_index] = action
                self.d[2][data_index] = reward
                self.d[3][data_index] = state_dash

    '''
    def retrieve_sequence_replay_index(self, length):
        #print('length = %s' % length)
        tmp_episode_end_flags = self.d[4].T[0].copy()
        #print('tmp_episode_end_flags = %s' % tmp_episode_end_flags)
        indices = [i for i in range(len(tmp_episode_end_flags)) if tmp_episode_end_flags[i]]
        indices.insert(0, -1)
        #print('indices = %s' % indices)

        tmp_rewards = self.d[2].T[0].copy()
        #print('tmp_rewards = %s' % tmp_rewards)
#        rewards = []
#        for i in range(len(indices) - 2):
#            rewards.append(reduce(lambda x, y: x + y, tmp_rewards[indices[i] + 1:indices[i + 1] + 1]))
        #each_sum_rewards = [reduce(lambda x, y: x + y, tmp_rewards[indices[i] + 1:indices[i + 1] + 1]) for i in range(len(indices) - 1)]
        #each_sum_rewards = [sum(tmp_rewards[indices[i] + 1:indices[i + 1] + 1]) for i in range(len(indices) - 1)]
        #print('each_sum_rewards = %s' % each_sum_rewards)
        #each_sum_rewards = utils.softmax(each_sum_rewards)
        #print('softmaxed each_sum_rewards = %s' % each_sum_rewards)

        while True:
            selected_end_index_of_indices = random.randint(1, len(indices) - 1)
            #selected_end_index_of_indices = np.random.choice(len(indices) - 1, p=each_sum_rewards) + 1
            #print('selected_end_index_of_indices = %s' % selected_end_index_of_indices)
            self.replay_size = indices[selected_end_index_of_indices] - indices[selected_end_index_of_indices - 1]
            if self.replay_size > 0:
                #print('broken')
                break

        #selected_start_index = random.randint(indices[selected_end_index_of_indices - 1] + 1, indices[selected_end_index_of_indices] - self.replay_size + 1)
        selected_start_index = indices[selected_end_index_of_indices - 1] + 1
        #print('selected_start_index = %s, self.replay_size = %s' % (selected_start_index, self.replay_size,))

        #a = np.array([range(selected_start_index, selected_start_index + self.replay_size)]).T
        a = [selected_start_index, selected_start_index + self.replay_size] # left is inclusive, right is exclusive.
        #print(a)
        return a
    '''


    def replay(self, time):
        replay_start = False
        if self.initial_exploration < time:
            replay_start = True
            replays = [0, 0, 0, 0, False]
            is_ripple_firing = False
            if self.process is None or self.process.exitcode is not None:
                if self.process is not None and self.process.exitcode is not None and self.process.exitcode == 0:
                    replays = self.queue.get()
                    print('%%%%%%%%%%%%%%%%%%%%%%%%')
                    print('Get replays!!!')
                    print('%%%%%%%%%%%%%%%%%%%%%%%%')
                    is_ripple_firing = True
                self.process = multiprocessing.Process(target=experience_replay, args=(self.queue, self.d, self.BATCH_SIZE, time, self.dim, self.data_size))
                self.process.start()
            print('Experience: self.process = %s, self.process.exitcode = %s' % (self.process, self.process.exitcode,))

            return replay_start, replays[0], replays[1], replays[2], replays[3], replays[4], is_ripple_firing
            '''
            replay_indices = []
            replay_sizes = []
            replay_max_size = 0
            for i in xrange(self.BATCH_SIZE):
                # Pick up replay_size number of samples from the Data
                if time < self.data_size:  # during the first sweep of the History Data
                    replay_indices.append(self.retrieve_sequence_replay_index(time))
                else:
                    replay_indices.append(self.retrieve_sequence_replay_index(self.data_size))
                tmp_replay_size = replay_indices[-1][1] - replay_indices[-1][0]
                if replay_max_size < tmp_replay_size:
                    replay_max_size = tmp_replay_size
                replay_sizes.append(self.replay_size)

            s_replay = -np.ones(shape=(replay_max_size, self.BATCH_SIZE, self.dim), dtype=np.float32)
            a_replay = np.ones(shape=(replay_max_size, self.BATCH_SIZE), dtype=np.uint8)
            r_replay = np.zeros(shape=(replay_max_size, self.BATCH_SIZE), dtype=np.float32)
            s_dash_replay = -np.ones(shape=(replay_max_size, self.BATCH_SIZE, self.dim), dtype=np.float32)
            episode_end_replay = np.zeros(shape=(replay_max_size, self.BATCH_SIZE), dtype=np.bool)
            #import bpdb; bpdb.set_trace()
            for j in xrange(self.BATCH_SIZE):
                s_replay[:, j] = np.append(np.asarray(self.d[0][replay_indices[j][0]:replay_indices[j][1]])[:,0,:], -np.ones((replay_max_size - replay_sizes[j], 10240), dtype=np.float32), axis=0)
                a_replay[:, j] = np.append(self.d[1][replay_indices[j][0]:replay_indices[j][1]], np.ones(replay_max_size - replay_sizes[j], dtype=np.uint8))
                r_replay[:, j] = np.append(self.d[2][replay_indices[j][0]:replay_indices[j][1]], np.zeros(replay_max_size - replay_sizes[j], dtype=np.float32))
                s_dash_replay[:, j] = np.append(np.array(self.d[3][replay_indices[j][0]:replay_indices[j][1]])[:,0,:], -np.ones((replay_max_size - replay_sizes[j], 10240), dtype=np.float32), axis=0)
                episode_end_replay[:, j] = np.append(self.d[4][replay_indices[j][0]:replay_indices[j][1]], np.zeros(replay_max_size - replay_sizes[j], dtype=np.bool))
            #print('s_replay = %s' % s_replay)

            if self.use_gpu >= 0:
                s_replay = cuda.to_gpu(s_replay)
                s_dash_replay = cuda.to_gpu(s_dash_replay)
            '''

            return replay_start, s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay, True

        else:
            return replay_start, 0, 0, 0, 0, False, False

    def end_episode(self, time, last_state, action, reward):
        self.stock(time, last_state, action, reward, last_state, True)
        replay_start, s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay, is_ripple_firing = \
            self.replay(time)

        return replay_start, s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay, is_ripple_firing
