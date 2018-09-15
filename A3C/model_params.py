import tensorflow as tf
import gym
import numpy as np
from gym import envs
import time
from PIL import Image
import random
import os
import tensorflow.contrib.slim as slim
import sys
import pickle
import argparse
import pprint
import glob
import datetime
from datetime import datetime

dir_path = os.path.dirname(os.path.realpath(__file__))

Experiment_params = {

    "n_max_episodes" : 100000,
    "n_max_timesteps_per_episode" : 250,
    "render" : False,
    "debug"  : False,
    "state_rep_size" : 4,
    "n_rows_cropped" : 10,
    "n_cols_cropped" : 4,
    "n_actions" : 6,
    "greedy_action_prob" : 0.1,
    "greedy_action_end_prob" : 0.01,
    "n_pre_train_steps" : 0,
    "n_annealing_steps": 1000000,
    "discount_factor" : 0.99,
    "model_path" : dir_path + "/A3C-checkpoints",
    "n_processed_episodes" : 0,
    "n_processed_time_steps" : 0,
    "n_discount_steps" : 8,
    "rewards":  {},
}


A3C_nn_architecture = {

    "input_layer": {
            "shape" : [None,84,84,Experiment_params["state_rep_size"]]
    },

    "conv_layer_1" : {
            "n_filters" : 16,
            "filter_size": [8, 8],
            "stride": 4,
    },

    "conv_layer_2" : {
            "n_filters" : 32,
            "filter_size": [4,4],
            "stride" : 2,
    },

    "dense_layer_1": {
            "n_units": 256,
    },

    "dense_layer_2": {
            "n_units": Experiment_params["n_actions"],
    },

    "dense_layer_3" : {
            "n_units": 1,
    },

    "optimization_params" : {
            "learning_rate": 1e-4,
    }
}

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class A3C_Network():
    def __init__(self, A3C_nn_architecture, scope):

        self.scope = scope


        with tf.variable_scope(scope):

            with tf.device('/cpu:0'):
                self.input_layer = tf.placeholder(tf.float32, shape=A3C_nn_architecture["input_layer"]["shape"])


            with tf.device("/device:GPU:0") :
                self.conv_layers = []

                for i in xrange(0,4) :
                    if i == 0 :
                        new_conv_layer = tf.layers.conv2d(inputs=self.input_layer,filters=32,kernel_size=[3,3],strides=2,activation=tf.nn.relu)
                        self.conv_layers.append(new_conv_layer)
                    else:
                        new_conv_layer = tf.layers.conv2d(inputs=self.conv_layers[i-1], filters=32, kernel_size=[3, 3],
                                                          strides=2, activation=tf.nn.relu)
                        self.conv_layers.append(new_conv_layer)


                self.flattened_layer = slim.flatten(self.conv_layers[-1])

                self.dense_layer_1 = tf.contrib.layers.fully_connected(
                    inputs=self.flattened_layer,
                    num_outputs=A3C_nn_architecture["dense_layer_1"]["n_units"],
                    activation_fn=tf.nn.relu,
                    biases_initializer=None,
                    #weights_initializer=normalized_columns_initializer(0.01)
                )

                #Add a LSTM layer here to capture temporal dependencies between sequences of images and incorporate these
                #dependencies in estimating the policy and value functions
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(256,state_is_tuple=True)
                c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
                h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
                self.lstm_state_init = [c_init, h_init]
                self.c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
                self.h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
                lstm_in = tf.expand_dims(self.dense_layer_1, [0])
                step_size = tf.shape(self.input_layer)[:1]
                state_in = tf.contrib.rnn.LSTMStateTuple(self.c_in, self.h_in)
                lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell, lstm_in, initial_state=state_in, sequence_length=None,time_major=False)
                lstm_c, lstm_h = lstm_state
                self.lstm_state_out = (lstm_c[:1, :], lstm_h[:1, :])
                self.lstm_out = tf.reshape(lstm_outputs, [-1, 256])
                #End of definition of lstm layer



                self.action_probabilities = tf.contrib.layers.fully_connected(
                    inputs=self.lstm_out,
                    num_outputs=A3C_nn_architecture["dense_layer_2"]["n_units"],
                    activation_fn=tf.nn.softmax,
                    biases_initializer=None,
                    weights_initializer=normalized_columns_initializer(0.01)
                )

                self.output_value = tf.contrib.layers.fully_connected(inputs=self.lstm_out, num_outputs=1,
                                                                      activation_fn=None, weights_initializer=normalized_columns_initializer(1.0))

            with tf.device("/cpu:0") :

                self.chosen_actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.chosen_actions_onehot = tf.one_hot(self.chosen_actions, Experiment_params["n_actions"],
                                                        dtype=tf.float32)
                self.actions_log_prob = tf.log(tf.reduce_sum(tf.multiply(self.action_probabilities, self.chosen_actions_onehot),axis=1) + 1e-10)

                self.expected_values = tf.placeholder(shape=[None], dtype=tf.float32)
                self.input_advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.policy_entropy = -1.0*tf.reduce_sum(self.action_probabilities * tf.log(self.action_probabilities + 1e-10))
                self.loss_policy = -1.0*tf.reduce_sum(self.actions_log_prob * self.input_advantages) - 0.01 * self.policy_entropy
                self.loss_value_fn = 0.5 * tf.reduce_sum(tf.square(self.expected_values - tf.reshape(self.output_value,[-1])))

                self.total_loss = self.loss_value_fn + self.loss_policy

                trainable_local_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.total_loss, trainable_local_variables)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)
                self.grad_norms = tf.global_norm(grads,name='norm')

                #global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                #self.apply_grads = self.trainer.apply_gradients(zip(grads, global_vars))


                self.policy_update = tf.train.RMSPropOptimizer(learning_rate=A3C_nn_architecture["optimization_params"]["learning_rate"]).minimize(self.loss_policy)
                self.value_update = tf.train.RMSPropOptimizer(learning_rate=A3C_nn_architecture["optimization_params"]["learning_rate"]).minimize(self.loss_value_fn)

                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=A3C_nn_architecture["optimization_params"]["learning_rate"])
                self.loss_update = self.optimizer.apply_gradients(zip(grads,trainable_local_variables))

                #Gradients for adversarial example generation
                self.best_action = tf.argmax(self.action_probabilities)
                self.best_action_pdf = tf.one_hot(self.best_action, Experiment_params["n_actions"], dtype=tf.float32)
                self.adv_loss = -1.0 * tf.reduce_sum(tf.multiply(self.best_action_pdf, tf.log(self.action_probabilities + 1e-100)))
                self.adv_grads_sign = tf.sign(tf.gradients(self.adv_loss, self.input_layer))
