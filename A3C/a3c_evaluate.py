# Evaluating a stored checkpoint


import sys
import argparse
from model_params import A3C_nn_architecture
from model_params import A3C_Network
from utils import *
import sys
import argparse
import matplotlib
from matplotlib import pyplot as plt
import os
import pickle
import numpy as np
import gym
import tensorflow as tf
import pprint
import time
import random
from worker import *




dir_path = os.path.dirname(os.path.realpath(__file__))


def get_cropped_image_pixels(env,Experiment_params) :
    image_size = env.observation_space.shape
    downsampled_rows = [2 * i for i in xrange(0, image_size[0] / 2)]
    downsampled_cols = [2 * i for i in xrange(Experiment_params["n_cols_cropped"] / 2,
                                              image_size[1] / 2 - Experiment_params["n_cols_cropped"] / 2)]

    for i in xrange(0, Experiment_params["n_rows_cropped"]):
        downsampled_rows.append(2 * i + 1)
        downsampled_rows.append(image_size[0] - 2 * i - 1)
    downsampled_rows.append(2 * Experiment_params["n_rows_cropped"] + 1)

    return downsampled_rows, downsampled_cols


def load_experiment_params(check_pt_id):
    params_file = dir_path + "/A3C-checkpoints/params_" + str(check_pt_id) + ".txt"

    if not os.path.exists(params_file):
        print "Error Loading Saved Model. Checkpointed Parameters do not exist !"
        sys.exit(0)

    with open(params_file, "rb") as fp:
        Experiment_params = pickle.load(fp)

    return Experiment_params



def get_next_action_value(sess, PongA3C_Network, curr_state, lstm_curr_state):

    assert len(lstm_curr_state) == 2
    feed_dict = {PongA3C_Network.input_layer: [curr_state],
                 PongA3C_Network.c_in: lstm_curr_state[0],
                 PongA3C_Network.h_in: lstm_curr_state[1]}

    prob_dist, v, lstm_state_out = sess.run([PongA3C_Network.action_probabilities, PongA3C_Network.output_value, PongA3C_Network.lstm_state_out],feed_dict=feed_dict)
    action = np.random.choice(np.arange(6), p=prob_dist[0])
    return action, v, lstm_state_out

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check_pt", dest="check_pt", help="Specific Checkpoint ID to load from")
    parser.add_argument("--render", dest="render", help="Render gameplay")
    parser.add_argument("--n_games", dest="n_games", help="n_games")
    parser.add_argument("--noise", dest="noise",help="adversarial noise to be added")
    parser.add_argument("--attack", dest="attack", help="whether to run the attack or not.")
    args = parser.parse_args()
    env = gym.make('PongNoFrameskip-v4')

    render = 0
    n_games = 20
    attack = 0
    noise = 0.1

    if args.check_pt:
        check_pt = int(args.check_pt)

    if args.render:
        render = int(args.render)

    if args.n_games:
        n_games = int(args.n_games)

    if args.noise :
        noise = float(args.noise)

    if args.attack :
        attack = int(args.attack)


    tf.reset_default_graph()

    Experiment_params = load_experiment_params(check_pt_id=check_pt)

    model_path = dir_path + "/A3C-checkpoints"
    worker = Worker(name=str(0), n_episodes_to_run=-1)
    worker.evaluate = True
    worker.n_games = n_games
    worker.render = render
    worker.attack = attack
    worker.noise = noise

    model_saver = tf.train.Saver()


    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    gpu_options.allow_growth=True

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options)) as sess:
        print "Loading Check pointed Model ..."
        ckpt = tf.train.get_checkpoint_state(model_path)
        model_saver.restore(sess, ckpt.model_checkpoint_path)

        print "Running model ..."
        worker.set_active_session(sess)
        worker.set_experiment_params(Experiment_params)
        worker.saver = model_saver
        worker.start()
        time.sleep(1.0)

        worker.join()




if __name__ == "__main__" :
    main()