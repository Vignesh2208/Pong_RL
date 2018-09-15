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
from experience_replay_buffer import *
import pickle
import argparse
import pprint
import glob
import datetime
from datetime import datetime


dir_path = os.path.dirname(os.path.realpath(__file__))
env = gym.make('PongNoFrameskip-v4')



Experiment_params = {

    "n_max_episodes" : 60000,
    "n_max_timesteps_per_episode" : 250,
    "render" : False,
    "debug"  : False,
    "state_rep_size" : 4,
    "n_rows_cropped" : 10,
    "n_cols_cropped" : 4,
    "n_actions" : 6,
    "experience_replay_size" : 50000,
    "minibatch_size" : 32,
    "greedy_action_prob" : 1.0,
    "greedy_action_end_prob" : 0.1,
    "n_pre_train_steps" : 50000,
    "n_annealing_steps": 50000,
    "discount_factor" : 0.99,
    "model_path" : dir_path + "/DQN-checkpoints",
    "n_processed_episodes" : 0,
    "n_processed_time_steps" : 0,
    "rewards":  [],
}

Qfn_nn_architecture = {

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

    "optimization_params" : {
            "learning_rate": 0.001,
            "decay" : 0.9,
            "momentum": 0.0,
            "epsilon" : 1e-5,
    }
}



class DQN_Network() :
    def __init__(self,Qfn_architecture,env) :

        self.nn_components = {}

        with tf.device("/cpu:0") :
            self.input_layer = tf.placeholder(tf.float32,shape=Qfn_architecture["input_layer"]["shape"])


            #print self.input_layer.shape
            self.conv_layer_1 = tf.layers.conv2d(
                            inputs=self.input_layer,
                            filters = Qfn_architecture["conv_layer_1"]["n_filters"],
                            kernel_size = Qfn_architecture["conv_layer_1"]["filter_size"],
                            strides = Qfn_architecture["conv_layer_1"]["stride"],
                            activation=tf.nn.relu,
                            use_bias=False
                           )

            #print self.conv_layer_1.shape

            self.conv_layer_2 = tf.layers.conv2d(
                            inputs=self.conv_layer_1,
                            filters=Qfn_architecture["conv_layer_2"]["n_filters"],
                            kernel_size=Qfn_architecture["conv_layer_2"]["filter_size"],
                            strides=Qfn_architecture["conv_layer_2"]["stride"],
                            activation=tf.nn.relu,
                            use_bias=False
                           )


            #print self.conv_layer_2.shape
            self.flattened_layer = slim.flatten(self.conv_layer_2)

        
            self.dense_layer_1 = tf.contrib.layers.fully_connected(
                            inputs = self.flattened_layer,
                            #units = Qfn_architecture["dense_layer_1"]["n_units"],
                            num_outputs=Qfn_architecture["dense_layer_1"]["n_units"],
                            #activation=tf.nn.relu
                            activation_fn=tf.nn.relu
                            )

            self.output_layer = tf.contrib.layers.fully_connected(
                            inputs=self.dense_layer_1,
                            #units=Qfn_architecture["dense_layer_2"]["n_units"],
                            num_outputs=Qfn_architecture["dense_layer_2"]["n_units"],
                            activation_fn=None
                            )

            self.output_probs = tf.nn.softmax(self.output_layer)

            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, Experiment_params["n_actions"], dtype=tf.float32)
            self.best_action = tf.argmax(input=self.output_layer,axis=1)


            self.output_Q_value = tf.reduce_sum(tf.multiply(self.output_layer, self.actions_onehot), axis=1)
            self.expected_Q_value = tf.placeholder(tf.float32,shape=[None])

            self.loss = tf.reduce_mean(tf.square(self.expected_Q_value - self.output_Q_value))
            self.train_step = tf.train.AdamOptimizer(Qfn_architecture["optimization_params"]["learning_rate"]).minimize(self.loss)


            #Define loss for adversarial example generation
            self.best_pdf = tf.one_hot(self.best_action, Experiment_params["n_actions"], dtype=tf.float32)
            self.adv_loss = -1.0 * tf.reduce_sum(tf.multiply(self.best_pdf,tf.log(self.output_probs + 1e-100)))
            self.adv_grads_sign = tf.sign(tf.gradients(self.adv_loss, self.input_layer))





        self.nn_components = {

            "input_layer": self.input_layer,
            "conv_layer_1" : self.conv_layer_1,
            "conv_layer_2" : self.conv_layer_2,
            "dense_layer_1" : self.dense_layer_1,
            "all_Q_values" : self.output_layer,
            "best_action" : self.best_action,
            "output_Q_value" : self.output_Q_value,
            "expected_Q_value": self.expected_Q_value,
            "loss" : self.loss,
            "train_step": self.train_step,
        }



    def get_Q_network(self):
        return self.nn_components







# Convert to grayscale and crop image to 84x84 pixel size
def rgb2grayscale(image,downsampled_rows,downsampled_cols) :

    image = np.delete(image,downsampled_rows,axis=0)
    image = np.delete(image,downsampled_cols,axis=1)
    assert image[:,:,0].shape == image[:,:,1].shape
    assert image[:,:,0].shape == image[:,:,2].shape
    grayscale_img = float(1.0/3.0)*(image[:,:,0] + image[:,:,1] + image[:,:,2])
    assert grayscale_img[:,:].shape == image[:,:,0].shape
    return grayscale_img

def debug(t,grayscale_img,action, reward, observation) :
    if t % 20 == 0 and Experiment_params["debug"] == True:
        img = Image.fromarray(grayscale_img)
        img.show()
        print action, reward, observation.shape, grayscale_img.shape



def get_next_action(env,sess,Q_Network,curr_state,total_time_steps) :

    global Experiment_params

    n_annealing_steps =   Experiment_params["n_annealing_steps"]
    n_pre_train_steps = Experiment_params["n_pre_train_steps"]
    if np.random.uniform() <= Experiment_params["greedy_action_prob"]  or total_time_steps < n_pre_train_steps :

        if total_time_steps >= n_pre_train_steps :
            reduction_step_size = float(1.0 - Experiment_params["greedy_action_end_prob"]) / float(n_annealing_steps)

            if Experiment_params["greedy_action_prob"] > Experiment_params["greedy_action_end_prob"] :
                Experiment_params["greedy_action_prob"] = Experiment_params["greedy_action_prob"] - reduction_step_size

        action = env.action_space.sample()
    else:
        if Experiment_params["greedy_action_prob"] >  Experiment_params["greedy_action_end_prob"] :
            reduction_step_size = float(1.0 - Experiment_params["greedy_action_end_prob"])/float(n_annealing_steps)

            if Experiment_params["greedy_action_prob"] > Experiment_params["greedy_action_end_prob"] :
                Experiment_params["greedy_action_prob"] = Experiment_params["greedy_action_prob"] - reduction_step_size

        action = sess.run(Q_Network.best_action,feed_dict = {Q_Network.input_layer : [curr_state]})
    return action


def train_minibatch(sess,Q_Network, batch) :
    input_images = np.array(batch["images"])
    input_actions = np.array(batch["actions"])
    expected_Q_values = np.array(batch["q_values"])

    input_actions = input_actions.flatten()
    expected_Q_values = expected_Q_values.flatten()


    try:
        sess.run(Q_Network.train_step, feed_dict = { Q_Network.input_layer: input_images, Q_Network.actions: input_actions, Q_Network.expected_Q_value: expected_Q_values})
    except:
        print "Unexpected error in training step", sys.exc_info()[0]
        print "Input Actions:"
        pprint.pprint(input_actions)

        print "Expected Q Values:"
        pprint.pprint(expected_Q_values)


def get_cropped_image_pixels(env) :
    image_size = env.observation_space.shape
    downsampled_rows = [2 * i for i in xrange(0, image_size[0] / 2)]
    downsampled_cols = [2 * i for i in xrange(Experiment_params["n_cols_cropped"] / 2,
                                              image_size[1] / 2 - Experiment_params["n_cols_cropped"] / 2)]

    for i in xrange(0, Experiment_params["n_rows_cropped"]):
        downsampled_rows.append(2 * i + 1)
        downsampled_rows.append(image_size[0] - 2 * i - 1)
    downsampled_rows.append(2 * Experiment_params["n_rows_cropped"] + 1)

    return downsampled_rows, downsampled_cols


def load_save_experiment_params(operation, check_pt_id) :

    global Experiment_params

    file_name = Experiment_params["model_path"] + "/params_" + str(check_pt_id) + ".txt"
    if operation == "Save" :
        if not os.path.exists(Experiment_params["model_path"]):
            os.makedirs(Experiment_params["model_path"])


        with open(file_name,"wb") as fp:
            pickle.dump(Experiment_params,fp)
    else:
        if not os.path.exists(file_name):
            print "Error Loading Saved Model. Checkpointed Parameters do not exist !"
            sys.exit(0)

        n_max_episodes = Experiment_params["n_max_episodes"]
        model_path = Experiment_params["model_path"]
        greedy_action_prob = Experiment_params["greedy_action_prob"]


        with open(file_name,"rb") as fp:
            Experiment_params = pickle.load(fp)
            Experiment_params["n_max_episodes"] = n_max_episodes
            Experiment_params["model_path"] = model_path
            Experiment_params["greedy_action_prob"] = greedy_action_prob

def cleanup(files_to_remove) :
    for file_path in files_to_remove :
        if "checkpoint" not in file_path :
            #os.remove(Experiment_params["model_path"] + "/" + file_path)
            print "Removing File: ", file_path


def load_save_model(operation,model_saver, sess,replay_buffer,check_pt_id) :

    if check_pt_id == -1 :
        print "ERROR Loading/Saving Model. Checkpoint ID is negative !"
        sys.exit(0)

    dir_files_to_remove = os.listdir(Experiment_params["model_path"])



    if operation == "Save" :
        load_save_experiment_params("Save", check_pt_id)
        model_saver.save(sess, Experiment_params["model_path"] + '/model-' + str(check_pt_id) + '.ckpt')
        delete = True
        try:
            replay_buffer.save_exp_buffer(check_pt_id)
        except:
            print "ERROR During Storage of Replay Buffer ! Prev CheckPoint Not Deleted "
            delete = False
        if delete == True :
            cleanup(dir_files_to_remove)
    else:

        ckpt = tf.train.get_checkpoint_state(Experiment_params["model_path"])
        model_saver.restore(sess,ckpt.model_checkpoint_path)


        print "Loading Replay Buffer ..."
        replay_buffer.load_exp_buffer(check_pt_id)
        load_save_experiment_params("Load", check_pt_id)





def main() :

    global Experiment_params

    parser = argparse.ArgumentParser()
    parser.add_argument("--check_pt", dest="check_pt", help="Specific Checkpoint ID to load from")
    parser.add_argument("--periodic_save",dest="periodic_save",help="Should the model be saved periodically ?")
    parser.add_argument("--n_episodes_to_run", dest="n_episodes_to_run", help="Specifies number of episodes to train in current run")
    args = parser.parse_args()

    if args.check_pt :
        check_pt_id = int(args.check_pt)
        load_model = True
    else:
        load_model = False
        check_pt_id = -1

    if args.periodic_save :

        if int(args.periodic_save) > 0 :
            period = int(args.periodic_save)
            save_model = True
        else:
            period = 1
            save_model = False
    else:
        save_model = False

    if args.n_episodes_to_run :
        n_episodes_to_run = int(args.n_episodes_to_run)
    else:
        n_episodes_to_run = 1 # Default


    tf.reset_default_graph()
    if not os.path.exists(Experiment_params["model_path"]):
        os.makedirs(Experiment_params["model_path"])

    Q_Network = DQN_Network(Qfn_nn_architecture, env)
    init = tf.global_variables_initializer()
    model_saver = tf.train.Saver()
    experience_replay_buffer = Exp_Buffer(Experiment_params["experience_replay_size"],Experiment_params["model_path"])
    downsampled_rows, downsampled_cols = get_cropped_image_pixels(env)



    reward_per_episode = Experiment_params["rewards"]

    start_time = datetime.now()

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.allow_growth = True


    with tf.Session(config=config) as sess :

        #sess.run(init)
        if load_model == True :
            print('Loading Model ...')
            load_save_model("Load",model_saver,sess,experience_replay_buffer,check_pt_id)
            total_time_steps = Experiment_params["n_processed_time_steps"]
            n_processed_episodes = Experiment_params["n_processed_episodes"]

            print "Resuming from Episode Number: ", n_processed_episodes
            print "Loaded Replay Buffer Size: ", experience_replay_buffer.buffer_size
            print "Loaded Experiment Params:"
            pprint.pprint(Experiment_params)
        else:
            sess.run(init)
            total_time_steps = 0
            n_processed_episodes = 0


        limit = min(Experiment_params["n_max_episodes"],n_episodes_to_run + n_processed_episodes)
        while n_processed_episodes < limit :
            observation = env.reset()

            start_action = env.action_space.sample()
            grayscale_img = rgb2grayscale(observation, downsampled_rows, downsampled_cols)

            curr_state = []
            for j in xrange(0,Experiment_params["state_rep_size"]) :
                curr_state.append(grayscale_img)

            curr_state = np.dstack(curr_state)

            assert curr_state.shape == (84,84,4)

            episode_reward = 0.0
            for t in xrange(0,Experiment_params["n_max_timesteps_per_episode"]) :
                if Experiment_params["render"]:
                    env.render()

                action = get_next_action(env,sess,Q_Network, curr_state,total_time_steps)
                next_state = []
                done = False
                reward = 0.0
                for j in xrange(0, Experiment_params["state_rep_size"]) :
                    if done == False :
                        observation, reward, done, info = env.step(action)
                        if reward > 0:
                            reward = 1.0
                        if reward < 0:
                            reward = -1.0


                        grayscale_img = rgb2grayscale(observation, downsampled_rows, downsampled_cols)


                    next_state.append(grayscale_img)

                next_state = np.dstack(next_state)
                episode_reward += reward
                experience_replay_buffer.add_to_exp_replay_buffer(curr_state=curr_state,action=action,reward=reward,next_state=next_state,terminal=done)
                curr_state = next_state
                assert curr_state.shape == (84, 84, 4)

                if total_time_steps > Experiment_params["n_pre_train_steps"] :

                    experience_replay_samples = experience_replay_buffer.sample_from_exp_replay_buffer(Experiment_params["minibatch_size"])
                    batch = {
                        "images" : None,
                        "actions" : None,
                        "q_values" : None
                    }

                    expected_Q_values = []
                    batch_input_images = []
                    batch_input_actions = []

                    for sample in experience_replay_samples :

                        start_state = sample[0]
                        action = sample[1]
                        reward = sample[2]
                        end_state = [sample[3]]
                        is_terminal = sample[4]

                        assert action >= 0 and action < Experiment_params["n_actions"]

                        if is_terminal :
                            expected_Q_values.append(reward)
                        else:
                            Q_values = sess.run(
                                                Q_Network.output_layer,
                                                feed_dict = {Q_Network.input_layer : end_state}
                                                )


                            assert len(Q_values[0]) == Experiment_params["n_actions"]
                            Q_val = reward + Experiment_params["discount_factor"]*max(Q_values[0])
                            expected_Q_values.append(Q_val)

                        batch_input_images.append(start_state)
                        batch_input_actions.append(action)

                    batch["images"] = batch_input_images
                    batch["actions"] = batch_input_actions
                    batch["q_values"] = expected_Q_values

                    train_minibatch(sess,Q_Network,batch)

                if done:
                    print "Episode ", n_processed_episodes, " Finished after ", t , " Steps "
                    break

                total_time_steps += 1
                Experiment_params["n_processed_time_steps"] = total_time_steps


            reward_per_episode.append(episode_reward)
            n_processed_episodes += 1
            Experiment_params["n_processed_episodes"] = n_processed_episodes
            print "Episode Number : ", n_processed_episodes, " Episode Reward : ", episode_reward

            if (n_processed_episodes % period  == 0 or n_processed_episodes == limit) and save_model == True :
                Experiment_params["rewards"] = reward_per_episode

                print "Finish Time: ", str(datetime.now()), " Start Time: ", str(start_time)
                print "Saving Model at Episode: ", n_processed_episodes
                load_save_model("Save",model_saver,sess,experience_replay_buffer, n_processed_episodes)
                print "Model Saved ..."






if __name__ == "__main__" :
    main()


