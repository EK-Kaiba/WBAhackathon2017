# -*- coding: utf-8 -*-

import multiprocessing
import copy
import numpy as np
from chainer import cuda, FunctionSet, Variable, optimizers
import chainer.functions as F
import chainer.links as L

from config.log import APP_KEY
import logging
app_logger = logging.getLogger(APP_KEY)


class QNet:
    # Hyper-Parameters
    gamma = 0.99  # Discount factor
    initial_exploration = 300#10**3  # Initial exploratoin. original: 5x10^4
    #replay_size = 32  # Replay (batch) size
    target_model_update_freq = 10**4  # Target update frequancy. original: 10^4
    data_size = 10**5  # Data size of history. original: 10^6
    hist_size = 1  # original: 4

    def __init__(self, use_gpu, enable_controller, dim, epsilon, epsilon_delta, min_eps):
        self.use_gpu = use_gpu
        self.num_of_actions = len(enable_controller)
        self.enable_controller = enable_controller
        self.dim = dim
        self.epsilon = epsilon
        self.epsilon_delta = epsilon_delta
        self.min_eps = min_eps
        self.time = 0

        app_logger.info("Initializing Q-Network...")

        hidden_dim = 256
        self.model = FunctionSet(
            #l4=F.Linear(self.dim*self.hist_size, hidden_dim, wscale=np.sqrt(2)),
            l4=L.LSTM(self.dim*self.hist_size, hidden_dim),
            q_value=F.Linear(hidden_dim, self.num_of_actions,
                             initialW=np.zeros((self.num_of_actions, hidden_dim),
                                               dtype=np.float32))
        )
        if self.use_gpu >= 0:
            self.model.to_gpu()

        self.model_target = copy.deepcopy(self.model)
        self.model_used_in_step = copy.deepcopy(self.model)

        self.optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.0001)
        self.optimizer.setup(self.model.collect_parameters())

        # History Data :  D=[s, a, r, s_dash, end_episode_flag]
        self.d = [np.zeros((self.data_size, self.hist_size, self.dim), dtype=np.uint8),
                  np.zeros(self.data_size, dtype=np.uint8),
                  np.zeros((self.data_size, 1), dtype=np.int8),
                  np.zeros((self.data_size, self.hist_size, self.dim), dtype=np.uint8),
                  np.zeros((self.data_size, 1), dtype=np.bool)]

    def reset_lstm_state_of_model(self):
        self.model.l4.reset_state()

    def reset_lstm_state_of_target_model(self):
        self.model_target.l4.reset_state()

    def reset_lstm_state_of_model_used_in_step(self):
        self.model_used_in_step.l4.reset_state()

    def forward(self, state, action, reward, state_dash, episode_end):
        #num_of_batch = state.shape[0]
        replay_size = state.shape[0]
        batch_size = state.shape[1]
        print('replay_size = %s' % replay_size)

        s = [Variable(one_state) for one_state in state]
        print('s = %s' % map(lambda x: x.data, s))
        s_dash = [Variable(one_state_dash) for one_state_dash in state_dash]
        print('s_dash = %s' % map(lambda x: x.data, s_dash))

        #q = self.q_func(s)  # Get Q-value
        q = [self.q_func(one_s) for one_s in s]
        print('q = %s' % map(lambda x: x.data, q))

        # Generate Target Signals
        #tmp = self.q_func_target(s_dash)  # Q(s',*)
        tmp = [self.q_func_target(one_s_dash) for one_s_dash in s_dash]
        print('tmp = %s' % map(lambda x: x.data, tmp))
        if self.use_gpu >= 0:
            #tmp = list(map(np.max, tmp.data.get()))  # max_a Q(s',a)
            tmp = list(map(lambda x: np.max(x.data, axis=1), tmp))
        else:
            #tmp = list(map(np.max, tmp.data))  # max_a Q(s',a)
            tmp = list(map(lambda x: np.max(x.data, axis=1), tmp))
        print('post tmp = %s' % tmp)

        max_q_dash = np.asanyarray(tmp, dtype=np.float32)
        print('max_q_dash = %s' % max_q_dash)
        if self.use_gpu >= 0:
            #target = np.asanyarray(q.data.get(), dtype=np.float32)
            target = np.asanyarray(map(lambda x: x.data, q), dtype=np.float32)
        else:
            # make new array
            #target = np.array(q.data, dtype=np.float32)
            target = np.asanyarray(map(lambda x: x.data, q), dtype=np.float32)
        print('target = %s' % target)

        #for i in xrange(num_of_batch):
        print('reward = %s' % reward)
        for j in xrange(batch_size):
            for i in xrange(replay_size):
                print('reward[%s] = %s' % (i, reward[i],))
                if not episode_end[i][j]:
                    tmp_ = reward[i][j] + self.gamma * max_q_dash[i][j]
                else:
                    tmp_ = reward[i][j]

                action_index = self.action_to_index(action[i][j])
                target[i, j, action_index] = tmp_
        print('')
        print('post target = %s' % target)

        # TD-error clipping
        if self.use_gpu >= 0:
            target = cuda.to_gpu(target)
        #td = Variable(target) - q  # TD error
        td = [Variable(one_target) - one_q for one_target, one_q in zip(target, q)]
        app_logger.info('TD error: {}'.format(map(lambda x: x.data, td)))
        #td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)  # Avoid zero division
        td_tmp = [one_td.data + 1000.0 * (abs(one_td.data) <= 1) for one_td in td]
        #print('td_tmp = %s' % td_tmp)
        #td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)
        td_clip = [one_td * (abs(one_td.data) <= 1) + one_td/abs(one_td_tmp) * (abs(one_td.data) > 1) for one_td, one_td_tmp in zip(td, td_tmp)]
        #print('td_clip = %s' % td_clip)

        zero_val = np.zeros((replay_size, batch_size, self.num_of_actions), dtype=np.float32)
        if self.use_gpu >= 0:
            zero_val = cuda.to_gpu(zero_val)
        zero_val = Variable(zero_val)
        loss = sum([F.mean_squared_error(one_td_clip, one_zero_val) for one_td_clip, one_zero_val in zip(td_clip, zero_val)])
        print('loss = %s' % loss.data)
        return loss, q

    def q_func_step(self, state):
        h4 = self.model_used_in_step.l4(state / 255.0)
        q = self.model_used_in_step.q_value(h4)
        print('every q = %s' % q.data)
        return q

    def q_func(self, state):
        #h4 = F.relu(self.model.l4(state / 255.0))
        h4 = self.model.l4(state / 255.0)
        q = self.model.q_value(h4)
        minus1s = np.zeros(shape=q.shape, dtype=np.float32)
        # for -1 in batching blank
        enable = (state.data[:,0] > -0.5)[:, np.newaxis]
        enable = Variable(np.array([[one_enable.item() for i in range(self.num_of_actions)] for one_enable in enable]))
        q = F.where(enable, q, minus1s)
        return q

    def q_func_target(self, state):
        #h4 = F.relu(self.model_target.l4(state / 255.0))
        h4 = self.model_target.l4(state / 255.0)
        q = self.model_target.q_value(h4)
        minus1s = np.zeros(shape=q.shape, dtype=np.float32)
        enable = (state.data[:,0] > -0.5)[:, np.newaxis]
        enable = Variable(np.array([[one_enable.item() for i in range(self.num_of_actions)] for one_enable in enable]))
        #enable = Variable(state[:,:,0] > -0.5)
        #enable = Variable(h4 != -1)
        q = F.where(enable, q, minus1s)
        return q

    def e_greedy(self, state, epsilon):
        s = Variable(state)
        q = self.q_func_step(s)
        q = q.data

        if np.random.rand() < epsilon:
            index_action = np.random.randint(0, self.num_of_actions)
            app_logger.info(" Random")
        else:
            if self.use_gpu >= 0:
                index_action = np.argmax(q.get())
            else:
                index_action = np.argmax(q)
            app_logger.info("#Greedy")
        return self.index_to_action(index_action), q

    def step_model_update(self):
        self.model_used_in_step = copy.deepcopy(self.model)

    def target_model_update(self):
        self.model_target = copy.deepcopy(self.model)

    def index_to_action(self, index_of_action):
        return self.enable_controller[index_of_action]

    def action_to_index(self, action):
        return self.enable_controller.index(action)

    def start(self, feature):
        #print('$$$$$$$$$$$$$$$$$$$$$q_net.start$$$$$$$$$$$$$$$$$$$$')
        self.state = np.zeros((self.hist_size, self.dim), dtype=np.uint8)
        self.state[0] = feature

        state_ = np.asanyarray(self.state.reshape(1, self.hist_size, self.dim), dtype=np.float32)
        if self.use_gpu >= 0:
            state_ = cuda.to_gpu(state_)

        # Update model used in every step.
        self.step_model_update()
        self.reset_lstm_state_of_model_used_in_step()

        # Generate an Action e-greedy
        action, q_now = self.e_greedy(state_, self.epsilon)
        return_action = action

        return return_action

    def update_model(self, replayed_experience):
        is_ripple_now = replayed_experience[6]

        if replayed_experience[0] and is_ripple_now:
            self.reset_lstm_state_of_model()
            self.reset_lstm_state_of_target_model()
            self.optimizer.zero_grads()
            loss, _ = self.forward(replayed_experience[1], replayed_experience[2],
                                        replayed_experience[3], replayed_experience[4], replayed_experience[5])
            loss.backward()
            self.optimizer.update()

        # Target model update
        if replayed_experience[0] and np.mod(self.time, self.target_model_update_freq) == 0:
            app_logger.info("Model Updated")
            self.target_model_update()

        if not is_ripple_now:
            self.time += 1
            app_logger.info("step: {}".format(self.time))

    def step(self, features):
        if self.hist_size == 4:
            self.state = np.asanyarray([self.state[1], self.state[2], self.state[3], features], dtype=np.uint8)
        elif self.hist_size == 2:
            self.state = np.asanyarray([self.state[1], features], dtype=np.uint8)
        elif self.hist_size == 1:
            self.state = np.asanyarray([features], dtype=np.uint8)
        else:
            app_logger.error("self.DQN.hist_size err")

        state_ = np.asanyarray(self.state.reshape(1, self.hist_size, self.dim), dtype=np.float32)
        if self.use_gpu >= 0:
            state_ = cuda.to_gpu(state_)

        # Exploration decays along the time sequence
        if self.initial_exploration < self.time:
            self.epsilon -= self.epsilon_delta
            if self.epsilon < self.min_eps:
                self.epsilon = self.min_eps
            eps = self.epsilon
        else:  # Initial Exploation Phase
            app_logger.info("Initial Exploration : {}/{} steps".format(self.time, self.initial_exploration))
            eps = 1.0

        # Generate an Action by e-greedy action selection
        action, q_now = self.e_greedy(state_, eps)

        if self.use_gpu >= 0:
            q_max = np.max(q_now.get())
        else:
            q_max = np.max(q_now)

        return action, eps, q_max
