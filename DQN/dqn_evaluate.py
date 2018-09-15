#Evaluating a stored checkpoint


import sys
import argparse
from dqn_train import Qfn_nn_architecture
from dqn_train import DQN_Network
from dqn_train import rgb2grayscale
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



dir_path = os.path.dirname(os.path.realpath(__file__))


def load_experiment_params(check_pt_id):

    params_file = dir_path + "/DQN-checkpoints/params_" + str(check_pt_id) + ".txt"

    if not os.path.exists(params_file):
        print "Error Loading Saved Model. Checkpointed Parameters do not exist !"
        sys.exit(0)

    with open(params_file, "rb") as fp:
        Experiment_params = pickle.load(fp)

    return Experiment_params

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


def get_best_action(sess, Q_Network, curr_state) :
    action,output_probs = sess.run([Q_Network.best_action,Q_Network.output_probs], feed_dict={Q_Network.input_layer: [curr_state]})
    #print output_probs
    #action = np.random.choice(np.arange(6), p=output_probs[0])
    return action


def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


def main() :

    parser = argparse.ArgumentParser()
    parser.add_argument("--check_pt", dest="check_pt", help="Specific Checkpoint ID to load from")
    parser.add_argument("--render", dest="render", help= "Render gameplay")
    parser.add_argument("--n_games", dest="n_games", help="n_games")
    parser.add_argument("--noise", dest="noise", help="adversarial noise to add")
    parser.add_argument("--attack", dest="attack", help="perform adversarial attacks with specified noise")
    args = parser.parse_args()
    env = gym.make('PongNoFrameskip-v4')




    render = 1
    n_games = 1
    noise = 0.1
    attack = 0

    if args.check_pt :
        check_pt = int(args.check_pt)

    if args.render:
        render = int(args.render)

    if args.n_games:
        n_games = int(args.n_games)

    if args.noise :
        noise = float(args.noise)

    if args.attack:
        attack = int(args.attack)


    Experiment_params = load_experiment_params(check_pt_id=check_pt)
    downsampled_rows, downsampled_cols = get_cropped_image_pixels(env,Experiment_params)
    model_path = dir_path + "/DQN-checkpoints"
    Q_Network = DQN_Network(Qfn_nn_architecture, env)
    model_saver = tf.train.Saver()

    game_stats = {}


    with tf.Session() as sess:
        print "Loading Check pointed Model ..."
        ckpt = tf.train.get_checkpoint_state(model_path)
        model_saver.restore(sess, ckpt.model_checkpoint_path)
        print "Running Model ..."

        n_games_played = 0
        while n_games_played < n_games :
            print "playing game: ", n_games_played + 1
            game_stats[n_games_played] = {}
            game_stats[n_games_played]["n_points_lost"] = 0
            game_stats[n_games_played]["n_points_won"] = 0
            game_stats[n_games_played]["duration"] = 0

            env.seed(int(time.time()))
            observation = env.reset()
            grayscale_img = rgb2grayscale(observation, downsampled_rows, downsampled_cols)

            curr_state = []


            random.seed(int(time.time()))
            np.random.seed(int(time.time()))
            for j in xrange(0, Experiment_params["state_rep_size"]):
                curr_state.append(grayscale_img)

            nops = 4
            for i in xrange(0,nops) :

                action = random.randint(0,Experiment_params["n_actions"]-1)
                observation,_,done,_ = env.step(action)
                grayscale_img = rgb2grayscale(observation, downsampled_rows, downsampled_cols)

                if len(curr_state) >=  Experiment_params["state_rep_size"] :
                    curr_state.pop(0)
                    curr_state.append(grayscale_img)
                else:
                    curr_state.append(grayscale_img)


            curr_state = np.dstack(curr_state)

            adv_grads_sign = sess.run(Q_Network.adv_grads_sign, feed_dict={Q_Network.input_layer:[curr_state]})
            new_state = curr_state + noise*adv_grads_sign[0][0]

            # plt.imshow(curr_state[:, :, 0], cmap='gray')
            # plt.show()
            # plt.imshow(new_state[:,:,0] - curr_state[:,:, 0],cmap='gray')
            # plt.show()
            # plt.imshow(new_state[:,:,0],cmap='gray')
            # plt.show()
            # print adv_grads_sign, adv_grads_sign[0][0].shape
            # sys.exit(0)

            done = False
            n_actions = 0

            while done == False and n_actions < 10000:


                if attack == 1 :
                    adv_grads_sign = sess.run(Q_Network.adv_grads_sign, feed_dict={Q_Network.input_layer: [curr_state]})
                    curr_state = curr_state + noise * adv_grads_sign[0][0]


                action = get_best_action(sess, Q_Network, curr_state)
                if render:
                    env.render()
                    time.sleep(0.01)

                n_actions += 1
                curr_state = []
                reward = 0
                for j in xrange(0, Experiment_params["state_rep_size"]) :
                    if done == False :
                        observation, reward, done, info = env.step(action)
                        if reward > 0:
                            reward = 1.0
                        if reward < 0:
                            reward = -1.0

                        if reward < 0:
                            game_stats[n_games_played]["n_points_lost"] = game_stats[n_games_played]["n_points_lost"] + 1
                        if reward > 0:
                            game_stats[n_games_played]["n_points_won"] = game_stats[n_games_played]["n_points_won"] + 1

                        if reward != 0 :
                            break

                        grayscale_img = rgb2grayscale(observation, downsampled_rows, downsampled_cols)
                        curr_state.append(grayscale_img)

                if reward != 0:
                    nops = 4
                    curr_state = []
                    for i in xrange(0, nops):

                        action = random.randint(0,Experiment_params["n_actions"]-1)
                        observation, _, done, _ = env.step(action)
                        grayscale_img = rgb2grayscale(observation, downsampled_rows, downsampled_cols)

                        if len(curr_state) >= Experiment_params["state_rep_size"] :
                            curr_state.pop(0)
                            curr_state.append(grayscale_img)
                        else:
                            curr_state.append(grayscale_img)

                curr_state = np.dstack(curr_state)
                game_stats[n_games_played]["duration"] = game_stats[n_games_played]["duration"] + 1

            n_games_played += 1


    if n_games > 0 :

        print "Game Stats: "
        pprint.pprint(game_stats)

        n_pts_won = []
        for game in game_stats :
            n_pts_won.append(game_stats[game]["n_points_won"])

        print "Mean number of points won: ", np.mean(n_pts_won)
        print "Stddev number of points won: ", np.std(n_pts_won)



    rewards = []
    for i in xrange(0,60000) :
            r = Experiment_params["rewards"][i]
            rewards.append(r)

    episodes = Experiment_params["n_processed_episodes"]
    mva_rewards = movingaverage(rewards, 100)

    matplotlib.rcParams.update({'font.size': 16})

    fig = plt.figure()
    plt.plot(mva_rewards[1:-1])

    plt.xlabel("Episode Number (Each episode is composed of 1000 frames)")
    plt.ylabel("Moving average of last 100 episode rewards")
    plt.title("DQN Training Performance")

    plt.show()


if __name__ == "__main__" :
    main()



