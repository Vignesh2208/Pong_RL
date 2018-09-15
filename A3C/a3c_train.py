from worker import *
import multiprocessing
import pprint



def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--check_pt", dest="check_pt", help="Specific Checkpoint ID to load from")
    parser.add_argument("--n_episodes_to_run", dest="n_episodes_to_run",help="Specifies number of episodes to train in current run")

    args = parser.parse_args()


    if args.check_pt :
        check_pt_id = int(args.check_pt)
        if check_pt_id == -1 :
            load_model = False
        else:
            load_model = True
    else:
        load_model = False
        check_pt_id = -1


    if args.n_episodes_to_run :
        n_episodes_to_run = int(args.n_episodes_to_run)
    else:
        n_episodes_to_run = 1 # Default


    if load_model == True :
        Experiment_params = load_save_experiment_params("Load",check_pt_id)
    else:
        Experiment_params = mp.Experiment_params

    Experiment_params["greedy_action_end_prob"] = mp.Experiment_params["greedy_action_end_prob"]
    Experiment_params["n_max_episodes"] = mp.Experiment_params["n_max_episodes"]


    tf.reset_default_graph()
    if not os.path.exists(mp.Experiment_params["model_path"]):
        os.makedirs(Experiment_params["model_path"])



    worker = Worker(name=str(0), n_episodes_to_run=n_episodes_to_run)
    model_saver = tf.train.Saver()


    gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_opts )) as sess:

        if load_model == True :
            ckpt = tf.train.get_checkpoint_state(Experiment_params["model_path"])
            model_saver.restore(sess, ckpt.model_checkpoint_path)
            print "Loading Experiment Params ..."
            pprint.pprint(Experiment_params)

        else:
            sess.run(tf.global_variables_initializer())


        worker.set_active_session(sess)
        worker.set_experiment_params(Experiment_params)
        worker.saver = model_saver
        worker.start()
        time.sleep(1.0)



        worker.join()


        print "Training Finished "
        print "Saving Experiment Params ..."

        n_processed_episodes = min(Experiment_params["n_processed_episodes"] + n_episodes_to_run,Experiment_params["n_max_episodes"])
        if check_pt_id != -1 :
            Experiment_params["n_processed_episodes"] = n_processed_episodes
        else:
            Experiment_params["n_processed_episodes"] = n_episodes_to_run

        Experiment_params["n_processed_time_steps"] = n_processed_episodes*Experiment_params["n_max_timesteps_per_episode"]
        load_save_experiment_params("Save",n_processed_episodes,Experiment_params)




if __name__ == "__main__" :
    main()








