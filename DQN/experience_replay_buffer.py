import os
import numpy as np
import random
import pickle
import sys
import gzip


class Exp_Buffer() :

    def __init__(self,experience_replay_size,model_path):

        self.experience_replay_buffer = []
        self.buffer_size = experience_replay_size
        self.storage_path = model_path

    def add_to_exp_replay_buffer(self,curr_state,action,reward,next_state,terminal):


        if len(self.experience_replay_buffer) > self.buffer_size:
            self.experience_replay_buffer.pop(0)
            self.experience_replay_buffer.append([curr_state, action, reward, next_state, terminal])
        else:
            self.experience_replay_buffer.append([curr_state, action, reward, next_state, terminal])

    def sample_from_exp_replay_buffer(self,sample_size):
        if len(self.experience_replay_buffer) < sample_size:
            sample_size = len(self.experience_replay_buffer)

        samples = np.array(random.sample(self.experience_replay_buffer, sample_size))
        return samples

    def save_exp_buffer(self,check_pt_id):
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

        fp = gzip.open(self.storage_path + "/Exp_replay_buffer_" + str(check_pt_id) + ".pklz",'wb')
        pickle.dump(self.experience_replay_buffer,fp)
        fp.close()

    def load_exp_buffer(self, check_pt_id):
        if not os.path.exists(self.storage_path + "/Exp_replay_buffer_" + str(check_pt_id) + ".pklz")  :
            print "Failed to Load Exp Replay Buffer. Check point does not exist !"
            sys.exit(0)

        fp = gzip.open(self.storage_path + "/Exp_replay_buffer_" + str(check_pt_id) + ".pklz", 'rb')
        self.experience_replay_buffer = pickle.load(fp)
        fp.close()


