from model_params import *
import model_params as mp


def debug(t,grayscale_img,action, reward, observation) :
    if t % 20 == 0 and Experiment_params["debug"] == True:
        img = Image.fromarray(grayscale_img)
        img.show()
        print action, reward, observation.shape, grayscale_img.shape



# Convert to grayscale and crop image to 84x84 pixel size
def rgb2grayscale(image,downsampled_rows,downsampled_cols) :

    image = np.delete(image,downsampled_rows,axis=0)
    image = np.delete(image,downsampled_cols,axis=1)
    assert image[:,:,0].shape == image[:,:,1].shape
    assert image[:,:,0].shape == image[:,:,2].shape
    grayscale_img = float(1.0/3.0)*(image[:,:,0] + image[:,:,1] + image[:,:,2])
    assert grayscale_img[:,:].shape == image[:,:,0].shape
    return grayscale_img

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


def load_save_experiment_params(operation, check_pt_id,Experiment_params=None) :


    if operation == "Save" :
        if Experiment_params == None :
            print "ERROR Saving Experiment params ! - no params to save"
            sys.exit(0)


        file_name = Experiment_params["model_path"] + "/params_" + str(check_pt_id) + ".txt"
        if not os.path.exists(Experiment_params["model_path"]):
            os.makedirs(Experiment_params["model_path"])


        with open(file_name,"wb") as fp:
            pickle.dump(Experiment_params,fp)
    else:
        Experiment_params = mp.Experiment_params
        file_name = Experiment_params["model_path"] + "/params_" + str(check_pt_id) + ".txt"
        if not os.path.exists(file_name):
            print "Error Loading Saved Model. Checkpointed Parameters do not exist !"
            sys.exit(0)

        n_max_episodes = Experiment_params["n_max_episodes"]
        model_path = Experiment_params["model_path"]
        with open(file_name,"rb") as fp:
            Experiment_params = pickle.load(fp)
            Experiment_params["n_max_episodes"] = n_max_episodes
            Experiment_params["model_path"] = model_path

    return Experiment_params


def cleanup(files_to_remove) :
    for file_path in files_to_remove :
        if "checkpoint" not in file_path :
            os.remove(Experiment_params["model_path"] + "/" + file_path)
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

        replay_buffer.load_exp_buffer(check_pt_id)
        load_save_experiment_params("Load", check_pt_id)
        ckpt_file_path = Experiment_params["model_path"] + '/model-' + str(check_pt_id) + '.ckpt'
        model_saver.restore(sess, ckpt_file_path)




