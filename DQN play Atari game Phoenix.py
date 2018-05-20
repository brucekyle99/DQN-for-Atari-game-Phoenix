# -*- coding: utf-8 -*-
'''
based on paper by Volodymyr Mnih
address: https://arxiv.org/pdf/1602.01783v1.pdf
'''
from __future__ import division, print_function, absolute_import

import threading
import random
import numpy as np
import time
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque

import gym
import tensorflow as tf
import tflearn
from keras import backend as K
from keras.layers import Conv2D, Flatten, Dense, Input
from keras.models import Model

writer_summary = tf.summary.FileWriter
merge_all_summaries = tf.summary.merge_all
histogram_summary = tf.summary.histogram
scalar_summary = tf.summary.scalar

# To define test or training purpose
testing = False
# Model path (to load when testing)
test_model_path = 'D:\\Q-learning\\qlearning.tflearn.ckpt'
# Use Atari game environment
game = 'Phoenix-v0'
# Learning threads for training
n_threads = 8

#   Training Parameters
# Max training steps
max_steps = 80000000
# Current training step
step = 0
# Consecutive screen frames during model training
action_repeat = 4
# Async gradient update frequency of each learning thread
I_AsyncUpdate = 5
# Timestep to reset the target network
I_target = 40000
# Learning rate
learning_rate = 0.001
# Reward discount rate
gamma = 0.99
# Number of timesteps to anneal epsilon value from initial to final
anneal_epsilon_timesteps = 400000

#   Parameters
# Display or not gym evironment screens
show_training = True
# Directory for storing tensorboard summaries
summary_dir = '/tmp/tflearn_logs/'
summary_interval = 100
checkpoint_path = 'qlearning.tflearn.ckpt'
checkpoint_interval = 2000
# Number of episodes to run gym evaluation
num_eval_episodes = 100

# build a deep Q network with keras/tflearn
def build_dqn(num_actions, action_repeat):

    # Inputs shape: [batch, channel, height, width] need to be changed into
    # shape [batch, height, width, channel], [None, 84, 84, action_repeat]

    input = tf.placeholder(tf.float32, [None, action_repeat, 84, 84])
    inputs = tf.transpose(input, [0, 2, 3, 1])
    # use keras
    '''
    model = Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), activation='relu', padding='same',
                   data_format='channels_first')(inputs)
    model = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same',
                   data_format='channels_first')(model)
    # model = Conv2D(filter=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(model)
    model = Flatten()(model)
    model = Dense(256, activation='relu')(model)
    q_values = Dense(num_actions)(model)

    return input, q_values
    '''
   # use Tflearn
    model = tflearn.conv_2d(inputs, 32, 8, strides=4, activation='relu') # 32filters, 8X8
    model = tflearn.conv_2d(model, 64, 4, strides=2, activation='relu') # 64filters, 4X4
    model = tflearn.fully_connected(model, 256, activation='relu')
    q_values = tflearn.fully_connected(model, num_actions)
    return input, q_values

#  Atari game environment
class AtariEnvironment(object):

    def __init__(self, gym_env, action_repeat):
        self.env = gym_env
        self.action_repeat = action_repeat

        # Available actions / controls in game
        self.gym_actions = range(gym_env.action_space.n)
        # use deque buffer to store the game samples for training, size  [1, action_repeat, 84, 84]
        self.state_buffer = deque()

    def get_initial_state(self):

        # Reset game and clear the state buffer
        self.state_buffer = deque()

        x_t = self.env.reset()
        x_t = self.get_preprocessed_frame(x_t)
        # Preprocess the game frame and stack 4 frames to know the game state s_t
        s_t = np.stack([x_t for i in range(self.action_repeat)], axis=0)

        # store into buffer for training
        for i in range(self.action_repeat-1):
            self.state_buffer.append(x_t)
        return s_t

    def get_preprocessed_frame(self, observation):
        # turn the game to gray image for easier processing, and rescale
        return resize(rgb2gray(observation), (110, 84))[13:110 - 13, :]

    def step(self, action_index):

        # execute an action and get the return value: state, reward, terminal (game end or not)
        x_t1, r_t, terminal, info = self.env.step(self.gym_actions[action_index])
        x_t1 = self.get_preprocessed_frame(x_t1)

        previous_frames = np.array(self.state_buffer)
        s_t1 = np.empty((self.action_repeat, 84, 84))
        s_t1[:self.action_repeat-1, :] = previous_frames
        s_t1[self.action_repeat-1] = x_t1

        # delete the old frames, and store new frames into the deque
        self.state_buffer.popleft()
        self.state_buffer.append(x_t1)

        return s_t1, r_t, terminal, info

# 1-step Q-learning
def sample_final_epsilon():

    # sample a fianal epsilon value and anneal to it
    final_epsilons = np.array([.1, .01, .5])
    probabilities = np.array([0.4, 0.3, 0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]

def actor_learner_thread(thread_id, env, session, graph_ops, num_actions,
                         summary_ops, saver):

    # implement one-step Q-learning for different actor-learner threads, these threads are asynchronous
    global max_steps, step

    # Unpack graph ops
    s = graph_ops["s"]
    q_values = graph_ops["q_values"]
    st = graph_ops["st"]
    target_q_values = graph_ops["target_q_values"]
    reset_target_network_params = graph_ops["reset_target_network_params"]
    a = graph_ops["a"]
    y = graph_ops["y"]
    grad_update = graph_ops["grad_update"]

    summary_placeholders, assign_ops, summary_op = summary_ops

    # use the Atari game environment
    env = AtariEnvironment(gym_env=env,
                           action_repeat=action_repeat)

    # Initialize Q-net status, action, label
    s_batch = []
    a_batch = []
    y_batch = []

    final_epsilon = sample_final_epsilon()
    initial_epsilon = 1.0
    epsilon = 1.0

    print("Thread " + str(thread_id) + " - Final epsilon: " + str(final_epsilon))

    time.sleep(3*thread_id)
    t = 0
    while step < max_steps:
        # Get initial game state: observation
        s_t = env.get_initial_state()
        # game is not over
        terminal = False

        # Set up per-episode counters
        ep_reward = 0
        episode_ave_max_q = 0
        ep_t = 0

        while True:
            # Forward the deep q network, get Q(s,a) values
            readout_t = q_values.eval(session=session, feed_dict={s: [s_t]})

            # Choose next action by epsilon-greedy policy
            a_t = np.zeros([num_actions])
            if random.random() <= epsilon:
                # choose a random action
                action_index = random.randrange(num_actions)
            else:
                action_index = np.argmax(readout_t)
            # execute the action at action_index by greedy policy, chance is 1-epsilon
            a_t[action_index] = 1

            # gradually scale down the epsilon value
            if epsilon > final_epsilon:
                epsilon -= (initial_epsilon - final_epsilon) / anneal_epsilon_timesteps


            # Execute game steps on game and return the state, reward, terminal
            s_t1, r_t, terminal, info = env.step(action_index)

            # Accumulate gradients
            readout_j1 = target_q_values.eval(session = session,
                                              feed_dict = {st : [s_t1]})
            clipped_r_t = np.clip(r_t, -1, 1)
            if terminal:
                # if game is ended
                y_batch.append(clipped_r_t)
            else:
                # execute q-algorithm
                y_batch.append(clipped_r_t + gamma * np.max(readout_j1))

            # append action and status into the batch
            a_batch.append(a_t)
            s_batch.append(s_t)


            # s_t is updated to next state
            s_t = s_t1
            step += 1
            t += 1

            ep_t += 1
            ep_reward += r_t
            episode_ave_max_q += np.max(readout_t)

            # Optionally update target network
            if step % I_target == 0:
                session.run(reset_target_network_params)

            # Optionally update online network
            if t % I_AsyncUpdate == 0 or terminal:
                if s_batch:
                    session.run(grad_update, feed_dict={y: y_batch,
                                                        a: a_batch,
                                                        s: s_batch})
                # Clear gradients
                s_batch = []
                a_batch = []
                y_batch = []

            # Save the model
            if t % checkpoint_interval == 0:
                saver.save(session, "D:\\Q-learning\\qlearning.ckpt", global_step=t)

            # feedback the game states if the gams is over
            if terminal:
                stats = [ep_reward, episode_ave_max_q/float(ep_t), epsilon]
                for i in range(len(stats)):
                    session.run(assign_ops[i],
                                {summary_placeholders[i]: float(stats[i])})
                print("| Thread %.2i" % int(thread_id), "| Step", t,
                      "| Reward: %.2i" % int(ep_reward), " Qmax: %.4f" %
                      (episode_ave_max_q/float(ep_t)),
                      " Epsilon: %.5f" % epsilon, " Epsilon progress: %.6f" %
                      (t/float(anneal_epsilon_timesteps)))
                break


def build_graph(num_actions):
    # Create shared deep q network
    s, q_network = build_dqn(num_actions=num_actions,
                             action_repeat=action_repeat)
    network_params = tf.trainable_variables()
    q_values = q_network

    # Create shared target network for threads
    st, target_q_network = build_dqn(num_actions=num_actions,
                                     action_repeat=action_repeat)
    target_network_params = tf.trainable_variables()[len(network_params):]
    target_q_values = target_q_network

    # Periodically update target network with online network weights
    reset_target_network_params = \
        [target_network_params[i].assign(network_params[i])
         for i in range(len(target_network_params))]

    # Define cost and gradient update op
    a = tf.placeholder("float", [None, num_actions])
    y = tf.placeholder("float", [None])
    action_q_values = tf.reduce_sum(tf.matmul(q_values, a), reduction_indices=1)
    # define loss by mean square of action _q_vallue and y
    loss = tflearn.mean_square(action_q_values, y)
    optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    grad_update = optimizer.minimize(loss, var_list=network_params)

    graph_ops = {"s": s,
                 "q_values": q_values,
                 "st": st,
                 "target_q_values": target_q_values,
                 "reset_target_network_params": reset_target_network_params,
                 "a": a,
                 "y": y,
                 "grad_update": grad_update}

    return graph_ops


# Set up some episode summary ops to visualize on tensorboard.
def build_summaries():
    episode_reward = tf.Variable(0.)
    scalar_summary("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    scalar_summary("Qmax Value", episode_ave_max_q)
    logged_epsilon = tf.Variable(0.)
    scalar_summary("Epsilon", logged_epsilon)
    # Threads shouldn't modify the main graph, so we use placeholders
    # to assign the value of every summary (instead of using assign method
    # in every thread, that would keep creating new ops in the graph)
    summary_vars = [episode_reward, episode_ave_max_q, logged_epsilon]
    summary_placeholders = [tf.placeholder("float")
                            for i in range(len(summary_vars))]
    assign_ops = [summary_vars[i].assign(summary_placeholders[i])
                  for i in range(len(summary_vars))]
    summary_op = merge_all_summaries()
    return summary_placeholders, assign_ops, summary_op


def get_num_actions():

    # get possible actions for the atari game
    env = gym.make(game)
    num_actions = env.action_space.n
    return num_actions


def train(session, graph_ops, num_actions, saver):

    # train the model
    # Set up game environments for different threads
    envs = [gym.make(game) for i in range(n_threads)]

    summary_ops = build_summaries()
    summary_op = summary_ops[-1]

    # Initialize variables
    session.run(tf.initialize_all_variables())
    writer = writer_summary(summary_dir + "/qlearning", session.graph)

    # Initialize target network weights
    session.run(graph_ops["reset_target_network_params"])

    # Start n_threads actor-learner training threads
    actor_learner_threads = \
        [threading.Thread(target=actor_learner_thread,
                          args=(thread_id, envs[thread_id], session,
                                graph_ops, num_actions, summary_ops, saver))
         for thread_id in range(n_threads)]
    for t in actor_learner_threads:
        t.start()
        time.sleep(0.01)

    # Show the training status and write summary statistics
    last_summary_time = 0
    while True:
        if show_training:
            for env in envs:
                env.render()
        now = time.time()
        if now - last_summary_time > summary_interval:
            summary_str = session.run(summary_op)
            writer.add_summary(summary_str, float(step))
            last_summary_time = now
    for t in actor_learner_threads:
        t.join()


def evaluation(session, graph_ops, saver):

    # test the saves model, see the actual effect
    saver.restore(session, test_model_path)
    print("Restored model weights from ", test_model_path)
    monitor_env = gym.make(game)


    # Unpack graph ops
    s = graph_ops["s"]
    q_values = graph_ops["q_values"]

    # use atari game environment
    env = AtariEnvironment(gym_env=monitor_env,
                           action_repeat=action_repeat)

    for i_episode in range(num_eval_episodes):
        s_t = env.get_initial_state()
        ep_reward = 0
        terminal = False
        while not terminal:
            monitor_env.render()
            readout_t = q_values.eval(session=session, feed_dict={s : [s_t]})
            action_index = np.argmax(readout_t)
            s_t1, r_t, terminal, info = env.step(action_index) # game status feedback
            s_t = s_t1
            ep_reward += r_t
        print(ep_reward)
    monitor_env.monitor.close()


def main(_):
    with tf.Session() as session:
        num_actions = get_num_actions()
        graph_ops = build_graph(num_actions)
        # keep max 5 ckpt files of trained models
        saver = tf.train.Saver(max_to_keep=5)

        if testing:
            # for testing purpose
            evaluation(session, graph_ops, saver)
        else:
            # for training purpose
            train(session, graph_ops, num_actions, saver)

if __name__ == "__main__":
    tf.app.run()