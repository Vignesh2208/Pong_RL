from utils import *
import threading
import time
import math
import matplotlib
from matplotlib import pyplot as plt


from gym.envs.classic_control import rendering

def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0:
        if not err:
            print "Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l)
            err.append('logged')
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)







class Worker(threading.Thread) :


    def __init__(self,name,n_episodes_to_run):

        threading.Thread.__init__(self)

        print "Initializing Environment For Worker: " + str(name)
        self.env = gym.make('PongNoFrameskip-v4')
        self.viewer = rendering.SimpleImageViewer()
        self.worker_name = "worker-" + str(name)
        self.saver = None
        self.local_A3C_graph = A3C_Network(A3C_nn_architecture=A3C_nn_architecture,scope=self.worker_name)
        self.sess = None
        self.n_episodes_to_run = n_episodes_to_run
        self.episode_no = 0

        self.downsampled_rows, self.downsampled_cols = get_cropped_image_pixels(self.env)
        self.lstm_minibatch_start_state = None
        self.evaluate = False
        self.n_games = 0
        self.render = False
        self.attack = 0
        self.noise = 0.1


    def set_experiment_params(self,Experiment_params):
        self.Experiment_params = Experiment_params
        self.model_path = self.Experiment_params["model_path"]
        self.n_max_episodes = Experiment_params["n_max_episodes"]
        self.n_episodes_to_run = Experiment_params["n_processed_episodes"] + self.n_episodes_to_run
        self.episode_no = Experiment_params["n_processed_episodes"]
        self.n_timesteps_per_episode = Experiment_params["n_max_timesteps_per_episode"]
        self.rewards = Experiment_params["rewards"]
        self.total_processed_timesteps = Experiment_params["n_processed_episodes"] * self.n_timesteps_per_episode
        self.greedy_action_start_prob = Experiment_params["greedy_action_prob"]
        self.greedy_action_end_prob = Experiment_params["greedy_action_end_prob"]
        self.n_annealing_steps = Experiment_params["n_annealing_steps"]

        if self.n_annealing_steps == 0 :
            self.reduction_step_size = 0.0
        else:
            self.reduction_step_size = float(self.greedy_action_start_prob - self.greedy_action_end_prob) / float(self.n_annealing_steps)

        self.n_discount_steps = Experiment_params["n_discount_steps"]
        self.discount_factor = Experiment_params["discount_factor"]


    def set_active_session(self,sess):
        self.sess = sess


    def get_discounted_rewards(self,instantaneous_rewards,final_value):

        discounted_rewards = [0.0]*len(instantaneous_rewards)

        R = final_value
        i = len(instantaneous_rewards)-1
        while i >= 0 :
            R = instantaneous_rewards[i] + self.discount_factor*R
            discounted_rewards[i] = R
            i = i - 1

        return discounted_rewards


    def get_next_action_value(self, sess, curr_state, lstm_curr_state, total_time_steps):

        assert len(lstm_curr_state) == 2
        feed_dict = {self.local_A3C_graph.input_layer: [curr_state],
                     self.local_A3C_graph.c_in: lstm_curr_state[0],
                     self.local_A3C_graph.h_in: lstm_curr_state[1]}

        prob_dist, v, lstm_state_out = sess.run([self.local_A3C_graph.action_probabilities, self.local_A3C_graph.output_value, self.local_A3C_graph.lstm_state_out],feed_dict=feed_dict)


        if total_time_steps <= self.Experiment_params["n_pre_train_steps"] :
            a =  self.env.action_space.sample()
        else:

            if total_time_steps > self.Experiment_params["n_pre_train_steps"] + self.Experiment_params["n_annealing_steps"] :
                exploration_prob = self.Experiment_params["greedy_action_end_prob"]
            else:
                exploration_prob = self.Experiment_params["greedy_action_prob"] - self.reduction_step_size*(total_time_steps - self.Experiment_params["n_pre_train_steps"])


            if np.random.uniform() <= exploration_prob :
                a = self.env.action_space.sample()
            else:
                a = np.random.choice(np.arange(6), p=prob_dist[0])


        return a,v[0,0],lstm_state_out


    def update_global_gradients(self, sess, final_state, lstm_curr_state, episode_buffer):



        states = episode_buffer[:,0]
        states = np.array([states[i] for i in xrange(0,len(states))])
        next_states = episode_buffer[:,3]
        actions = episode_buffer[:,1].flatten()
        rewards = episode_buffer[:,2].flatten()
        value_fn_states = episode_buffer[:,4].flatten()
        is_terminal = episode_buffer[:,5].flatten()



        if is_terminal[-1] == True :
            final_state_value = 0.0
        else:
            final_state_value = sess.run(self.local_A3C_graph.output_value, feed_dict={self.local_A3C_graph.input_layer: [final_state],
                                                                                       self.local_A3C_graph.c_in: lstm_curr_state[0],
                                                                                       self.local_A3C_graph.h_in: lstm_curr_state[1]})[0, 0]

        discounted_rewards = self.get_discounted_rewards(instantaneous_rewards=rewards,final_value=final_state_value)
        values_extended = np.append(value_fn_states, final_state_value).flatten()
        input_advantages = rewards + self.discount_factor*values_extended[1:] - values_extended[:-1]
        expected_value_fn_states = discounted_rewards
        #This is some Generalized advantage estimation technique
        # i = len(input_advantages) - 2
        # while i >= 0 :
        #     input_advantages[i] = input_advantages[i] + self.discount_factor*input_advantages[i+1]
        #     i = i - 1

        i = len(input_advantages) - 1
        while i >= 0 :
            input_advantages[i] = math.pow(self.discount_factor,i)*input_advantages[i]
            i = i -1



        expected_value_fn_states = np.array(expected_value_fn_states).flatten()
        input_advantages = np.array(input_advantages).flatten()


        feed_dict = {self.local_A3C_graph.input_advantages: input_advantages,
                     self.local_A3C_graph.input_layer: states,
                     self.local_A3C_graph.chosen_actions: actions,
                     self.local_A3C_graph.expected_values: expected_value_fn_states,
                     self.local_A3C_graph.c_in: self.lstm_minibatch_start_state[0],
                     self.local_A3C_graph.h_in: self.lstm_minibatch_start_state[1]
                     }

        gradients,self.lstm_minibatch_start_state,grad_norm, _ = sess.run([self.local_A3C_graph.gradients, self.local_A3C_graph.lstm_state_out, self.local_A3C_graph.grad_norms, self.local_A3C_graph.loss_update],feed_dict=feed_dict)


        return gradients, grad_norm






    def save_model(self,sess):
        dir_files_to_remove = os.listdir(self.Experiment_params["model_path"])
        cleanup(dir_files_to_remove)
        print "Saving Model ..."

        #n_processed_episodes = min(self.Experiment_params["n_processed_episodes"] + self.n_episodes_to_run, self.Experiment_params["n_max_episodes"])
        n_processed_episodes = min(self.episode_no-1,self.Experiment_params["n_max_episodes"])
        self.saver.save(sess, self.model_path + '/model-' + str(n_processed_episodes) + '.ckpt')



    def runEpisode(self,sess):

        episode_no = self.episode_no
        self.episode_no += 1
        episode_grads = []

        if self.episode_no > self.n_episodes_to_run:
            self.save_model(sess)
            return -1



        total_processed_timesteps = (episode_no-1)*self.n_timesteps_per_episode
        episode_reward = 0
        done = False
        curr_state = []
        self.env.seed(int(time.time()))
        observation = self.env.reset()

        episode_buffer = []

        grayscale_img = rgb2grayscale(observation, self.downsampled_rows, self.downsampled_cols)
        for j in xrange(0, self.Experiment_params["state_rep_size"]):
            curr_state.append(grayscale_img)
        curr_state = np.dstack(curr_state)


        lstm_curr_state = self.lstm_minibatch_start_state
        for i in xrange(0,self.n_timesteps_per_episode) :
            if done == True :
                break

            if self.Experiment_params["render"] and self.worker_name == "worker-0":
                #self.env.render()
                rgb = self.env.render('rgb_array')
                upscaled = repeat_upsample(rgb, 4, 4)
                self.viewer.imshow(upscaled)

            action,value,lstm_curr_state = self.get_next_action_value(sess,  curr_state, lstm_curr_state, total_processed_timesteps)

            next_state = []
            reward = 0.0
            for j in xrange(0, self.Experiment_params["state_rep_size"]):
                if done == False:
                    observation, r, done, info = self.env.step(action)
                    if r > 0:
                        reward = 1.0
                    if r < 0:
                        reward = -1.0

                    grayscale_img = rgb2grayscale(observation, self.downsampled_rows, self.downsampled_cols)
                next_state.append(grayscale_img)

            next_state = np.dstack(next_state)
            episode_reward += reward

            if reward > 0 :
                print "+ve reward: ", reward
            if reward < -1 :
                print "-ve reward: ", reward, len(episode_buffer) + 1
            episode_buffer.append([curr_state,action,reward*abs(reward),next_state,value,done])
            curr_state = next_state
            total_processed_timesteps += 1

            if (len(episode_buffer) == self.n_discount_steps) and done != True and i != self.n_timesteps_per_episode - 1 :
                grads,grad_norms = self.update_global_gradients(sess, curr_state, lstm_curr_state, np.array(episode_buffer))
                episode_buffer = []
                episode_grads.append(grad_norms)


        self.rewards[episode_no] = episode_reward

        if len(episode_buffer) != 0 :
            grads, grad_norms = self.update_global_gradients(sess, curr_state, lstm_curr_state, np.array(episode_buffer))
            episode_buffer = []
            episode_grads.append(grad_norms)

        print self.worker_name + ": Episode No :", episode_no, " Episode Reward : ", episode_reward, " grad norm: ", \
        episode_grads[-1]

        return 0





    def run(self):

        print "Starting worker " + str(self.worker_name)
        assert self.sess != None

        if self.evaluate == True :
            self.evaluate_m()
            return

        with self.sess.as_default(), self.sess.graph.as_default() :
            ret = 0
            while ret == 0 :
                self.lstm_minibatch_start_state = self.local_A3C_graph.lstm_state_init
                ret = self.runEpisode(self.sess)

        print "Stopping worker " + str(self.worker_name)

    def get_next_action(self,sess,  curr_state, lstm_curr_state):

        assert len(lstm_curr_state) == 2
        feed_dict = {self.local_A3C_graph.input_layer: [curr_state],
                     self.local_A3C_graph.c_in: lstm_curr_state[0],
                     self.local_A3C_graph.h_in: lstm_curr_state[1]}

        prob_dist, v, lstm_state_out = sess.run(
            [self.local_A3C_graph.action_probabilities, self.local_A3C_graph.output_value, self.local_A3C_graph.lstm_state_out],
            feed_dict=feed_dict)

        np.random.seed(int(time.time()))

        action = np.argmax(prob_dist[0])
        return action, v, lstm_state_out

    def movingaverage(self,interval, window_size):
        window = np.ones(int(window_size)) / float(window_size)
        return np.convolve(interval, window, 'same')

    def evaluate_m(self):
        game_stats = {}
        n_games_played = 0
        n_games = self.n_games
        downsampled_rows, downsampled_cols = get_cropped_image_pixels(self.env)
        while n_games_played < n_games:
            print "playing game: ", n_games_played + 1
            game_stats[n_games_played] = {}
            game_stats[n_games_played]["n_points_lost"] = 0
            game_stats[n_games_played]["n_points_won"] = 0
            game_stats[n_games_played]["duration"] = 0


            np.random.seed(int(time.time()))
            random.seed(int(time.time()))
            observation = self.env.reset()
            self.env.seed(int(time.time()))

            lstm_curr_state = self.local_A3C_graph.lstm_state_init
            grayscale_img = rgb2grayscale(observation, downsampled_rows, downsampled_cols)
            curr_state = []
            for j in xrange(0, Experiment_params["state_rep_size"]):
                action = random.randint(0, self.Experiment_params["n_actions"] - 1)
                observation, _, done, _ = self.env.step(action)
                grayscale_img = rgb2grayscale(observation, downsampled_rows, downsampled_cols)

                if len(curr_state) >= Experiment_params["state_rep_size"]:
                    curr_state.pop(0)
                    curr_state.append(grayscale_img)
                else:
                    curr_state.append(grayscale_img)

            curr_state = np.dstack(curr_state)
            done = False
            n_actions = 0

            while done == False and n_actions < 10000:

                n_actions += 1

                if self.render == True:
                    #self.env.render()
                    rgb = self.env.render('rgb_array')
                    upscaled = repeat_upsample(rgb, 4, 4)
                    self.viewer.imshow(upscaled)
                    time.sleep(0.01)


                if self.attack == 1 :
                    adv_grads_sign = self.sess.run(self.local_A3C_graph.adv_grads_sign, feed_dict={self.local_A3C_graph.input_layer: [curr_state],
                                                                                                   self.local_A3C_graph.c_in:lstm_curr_state[0],
                                                                                                   self.local_A3C_graph.h_in:lstm_curr_state[1]
                                                                                                   })
                    curr_state = curr_state + self.noise * adv_grads_sign[0][0]


                action, v, lstm_curr_state = self.get_next_action(self.sess, curr_state, lstm_curr_state)
                curr_state = []
                for j in xrange(0, Experiment_params["state_rep_size"]):
                    if done == False:
                        observation, reward, done, info = self.env.step(action)
                        if reward > 0:
                            reward = 1.0
                        if reward < 0:
                            reward = -1.0

                        if reward != 0 :
                            break



                        grayscale_img = rgb2grayscale(observation, downsampled_rows, downsampled_cols)
                        curr_state.append(grayscale_img)



                if reward != 0:
                    nops = 4
                    for i in xrange(0, 4):
                        action = random.randint(0,self.Experiment_params["n_actions"]-1)
                        observation, _, done, _ = self.env.step(action)
                        grayscale_img = rgb2grayscale(observation, downsampled_rows, downsampled_cols)

                        if len(curr_state) >= Experiment_params["state_rep_size"] :
                            curr_state.pop(0)
                            curr_state.append(grayscale_img)
                        else:
                            curr_state.append(grayscale_img)

                curr_state = np.dstack(curr_state)
                if reward < 0:
                    game_stats[n_games_played]["n_points_lost"] = game_stats[n_games_played]["n_points_lost"] + 1
                if reward > 0:
                    game_stats[n_games_played]["n_points_won"] = game_stats[n_games_played]["n_points_won"] + 1

                game_stats[n_games_played]["duration"] = game_stats[n_games_played]["duration"] + 1

            n_games_played += 1

        print "Game Stats: "
        pprint.pprint(game_stats)

        n_pts_won = []
        for game in game_stats:
            n_pts_won.append(game_stats[game]["n_points_won"])

        print "Mean number of points won: ", np.mean(n_pts_won)
        print "Stddev number of points won: ", np.std(n_pts_won)

        episodes = self.Experiment_params["n_processed_episodes"]
        rewards = []
        for i in xrange(0, episodes):
            rewards.append(self.Experiment_params["rewards"][i])

        mva_rewards = self.movingaverage(rewards, 100)

        matplotlib.rcParams.update({'font.size': 16})
        fig = plt.figure()
        plt.plot(mva_rewards)

        plt.xlabel("Episode Number (Each episode is composed of 1000 frames)")
        plt.ylabel("Moving average of last 1000 episode rewards")
        plt.title("A3C Training Performance")

        plt.show()











