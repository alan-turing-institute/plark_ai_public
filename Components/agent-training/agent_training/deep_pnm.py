import PIL.Image
from IPython.display import display,clear_output,HTML
from IPython.display import Image as DisplayImage
import base64
import json
from io import StringIO
import ipywidgets as widgets
import sys
import time
import datetime
import imageio
import numpy as np
import io
import os
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import SubprocVecEnv
from plark_game import classes
#from gym_plark.envs import plark_env,plark_env_guided_reward,plark_env_top_left
from gym_plark.envs.plark_env_sparse import PlarkEnvSparse
from gym_plark.envs.plark_env import PlarkEnv
import datetime

from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv

from tensorboardX import SummaryWriter


import helper 

# To solve linear programs
import lemkelcp as lcp

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def envops(env, logdir):
    os.makedirs(logdir, exist_ok = True)  
    env = Monitor(env, logdir)
    return env

def compute_payoff_matrix(driving_agent,
                          model_type,
                          policy,
                          env,
                          player_filepaths,
                          opponent_filepaths):
    payoffs = np.zeros((len(player_filepaths), len(opponent_filepaths)))
    for i, model in enumerate(player_filepaths):
        for j, opponent_agent_filepath in enumerate(opponent_filepaths):
            # STILL TO CORRECT: NEED TO LOAD THE MODEL FROM THE FILEPATH
            model = helper.make_new_model(model_type, policy, env)
            # STILL TO CORRECT: NEED TO LOAD THE MODEL FROM THE FILEPATH
            opponent_model = helper.make_new_model(model_type, policy, env)
            if driving_agent == 'pelican':
                env.set_pelican(model)
                env.set_panther(opponent_model)
            else:
                env.set_panther(model)
                env.set_pelican(opponent_model)

            victory_count, avg_reward = helper.check_victory(model, env, trials = 10)
            payoffs[i,j] = avg_reward
    return payoffs

def train_agent_against_mixture(driving_agent,
                                policy,
                                model,
                                env,
                                tests,
                                mixture,
                                testing_interval,
                                max_steps,
                                model_type,
                                basicdate,tb_writer,
                                tb_log_name,
                                early_stopping = True,
                                previous_steps = 0):
    opponent_filepaths = np.random.choice(tests, size = max_steps // testing_interval, p = mixture)
    steps = 0
    for opponent_agent_filepath in opponent_filepaths:
        # STILL TO CORRECT: NEED TO LOAD THE MODEL FROM THE FILEPATH
        opponent_model = helper.make_new_model(model_type, policy, env)
        if driving_agent == 'pelican':
            env.set_panther(opponent_model)
        else:
            env.set_pelican(opponent_model)

        _, new_steps = train_agent(exp_path,
                               model,
                               env,
                               testing_interval,
                               testing_interval,
                               model_type,
                               basicdate,
                               tb_writer,
                               tb_log_name,
                               early_stopping = True,
                               previous_steps = steps,
                               save_model = False)
        steps += new_steps
    basicdate = basicdate + '_steps_' + str(previous_steps + steps)
    agent_filepath, _, _ = helper.save_model_with_env_settings(exp_path, model, model_type, env, basicdate)
    agent_filepath = os.path.dirname(agent_filepath)
    return agent_filepath, steps

def train_agent(exp_path,
                model,
                env,
                testing_interval,
                max_steps,
                model_type,
                basicdate,
                tb_writer,
                tb_log_name,
                early_stopping = True,
                previous_steps = 0,
                save_model = True):
    steps = 0
    logger.info("Beginning training for {} steps".format(max_steps))
    model.set_env(env)
    
    while steps < max_steps:
        logger.info("Training for {} steps".format(testing_interval))
        model.learn(testing_interval)    
        steps = steps + testing_interval
        if early_stopping:
            victory_count, avg_reward = helper.check_victory(model, env, trials = 10)
            if tb_writer is not None and tb_log_name is not None:
                tb_steps = steps + previous_steps
                logger.info("Writing to tensorboard for {} after {} steps".format(tb_log_name, tb_steps))
                tb_writer.add_scalar('{}_avg_reward'.format(tb_log_name), avg_reward, tb_steps)
                tb_writer.add_scalar('{}_victory_count'.format(tb_log_name), victory_count, tb_steps)
            if victory_count > 7:
                logger.info("Stopping training early")
                break #Stopping training as winning
    #Save agent
    logger.info('steps = '+ str(steps))
    agent_filepath = ''
    if save_model:
        basicdate = basicdate + '_steps_' + str(previous_steps + steps)
        agent_filepath ,_, _= helper.save_model_with_env_settings(exp_path, model, model_type, env, basicdate)
        agent_filepath = os.path.dirname(agent_filepath)
    return agent_filepath, steps

def run_deep_pnm(exp_name,
                 exp_path,
                 basicdate,
                 pelican_testing_interval = 100,
                 pelican_max_learning_steps = 10000,
                 panther_testing_interval = 100,
                 panther_max_learning_steps = 10000,
                 deep_pnm_iterations = 10000,
                 model_type = 'PPO2',
                 log_to_tb = False,
                 image_based = True,
                 num_parallel_envs = 1):

    pelican_training_steps = 0
    panther_training_steps = 0
    pelican_model_type = model_type
    panther_model_type = model_type
    pelican_tmp_exp_path = os.path.join(exp_path, 'pelican')
    os.makedirs(pelican_tmp_exp_path, exist_ok = True)
    panther_tmp_exp_path = os.path.join(exp_path, 'panther')
    os.makedirs(panther_tmp_exp_path, exist_ok = True)

    if log_to_tb:
        writer = SummaryWriter(exp_path)
        pelican_tb_log_name = 'pelican'
        panther_tb_log_name = 'panther'
    else:
        writer = None
        pelican_tb_log_name = None
        panther_tb_log_name = None    
            
    policy = 'CnnPolicy'
    if image_based is False:
        policy = 'MlpPolicy'

    parallel = False
    if model_type.lower() == 'ppo2':
        parallel = True
    
    log_dir_base = 'deep_pnm/'
    os.makedirs(log_dir_base, exist_ok = True)
    config_file_path = 'Components/plark-game/plark_game/game_config/10x10/balanced.json'

    #Train initial pelican vs rule based panther
    if parallel:
        pelican_env = envops(SubprocVecEnv([lambda:PlarkEnv(driving_agent = 'pelican',
                                                            config_file_path = config_file_path,
                                                            image_based = image_based,
                                                            random_panther_start_position = True,
                                                            max_illegal_moves_per_turn = 3) for _ in range(num_parallel_envs)]),
                             log_dir_base + '/pelican/')
    else:
        pelican_env = envops(PlarkEnv(driving_agent ='pelican',
                                      config_file_path = config_file_path,
                                      image_based = image_based,
                                      random_panther_start_position = True,
                                      max_illegal_moves_per_turn = 3),
                             log_dir_base + '/pelican/')
    
    pelican_model = helper.make_new_model(model_type, policy, pelican_env)
    logger.info('Training initial pelican')
    pelican_agent_filepath, steps = train_agent(parallel,
                                                num_parallel_envs,
                                                image_based,
                                                pelican_tmp_exp_path,
                                                pelican_model,
                                                pelican_env,
                                                pelican_testing_interval,
                                                pelican_max_learning_steps,
                                                pelican_model_type,
                                                basicdate,
                                                writer,
                                                pelican_tb_log_name)
    pelican_training_steps = pelican_training_steps + steps

    # Train initial panther agent vs rule based pelican
    if parallel:
        panther_env = envops(SubprocVecEnv([lambda:PlarkEnv(driving_agent = 'panther',
                                                            config_file_path = config_file_path,
                                                            image_based = image_based,
                                                            random_panther_start_position = True,
                                                            max_illegal_moves_per_turn = 3) for _ in range(num_parallel_envs)]),
                             log_dir_base + '/panther/')
    else:
        panther_env = envops(PlarkEnv(driving_agent = 'panther',
                                      config_file_path = config_file_path,
                                      image_based = image_based,
                                      random_panther_start_position = True,
                                      max_illegal_moves_per_turn = 3),
                             log_dir_base + '/panther/')
    panther_model = helper.make_new_model(model_type, policy, panther_env)        
    logger.info('Training initial panther')
    panther_agent_filepath, steps = train_agent(parallel,
                                                num_parallel_envs,
                                                image_based,
                                                panther_tmp_exp_path,
                                                panther_model,
                                                panther_env,
                                                panther_testing_interval,
                                                panther_max_learning_steps,
                                                panther_model_type,
                                                basicdate,
                                                writer,
                                                panther_tb_log_name)
    panther_training_steps = panther_training_steps + steps

    #Initialize the mixture of opponents
    pelican_models = [pelican_model]
    panther_models = [panther_model]
    W2 = [pelican_agent_filepath]
    W1 = [panther_agent_filepath]
    mixture1 = np.array([1.])
    mixture2 = np.array([1.])

    # Train agent vs agent
    logger.info('Deep Parallel Nash Memory')
    
    for i in range(deep_pnm_iterations):
        logger.info('Deep PNM iteration ' + str(i) + ' of ' + str(deep_pnm_iterations))
        logger.info('Training pelican')
        pelican_model = helper.make_new_model(model_type, policy, pelican_env)
        pelican_agent_filepath, steps = train_agent_against_mixture('pelican',
                                                                    policy,
                                                                    exp_path,
                                                                    pelican_model,
                                                                    pelican_env,
                                                                    W1,
                                                                    mixture1,
                                                                    pelican_testing_interval,
                                                                    pelican_max_learning_steps,
                                                                    pelican_model_type,
                                                                    basicdate,
                                                                    writer,
                                                                    pelican_tb_log_name,
                                                                    previous_steps = pelican_training_steps)
        pelican_training_steps = pelican_training_steps + steps

        logger.info('Training panther')
        panther_model = helper.make_new_model(model_type, policy, panther_env)
        panther_agent_filepath, steps = train_agent_against_mixture('pelican',
                                                                    policy,
                                                                    exp_path,
                                                                    panther_model,
                                                                    panther_env,
                                                                    W2,
                                                                    mixture2,
                                                                    panther_testing_interval,
                                                                    panther_max_learning_steps,
                                                                    panther_model_type,
                                                                    basicdate,
                                                                    writer,
                                                                    panther_tb_log_name,
                                                                    previous_steps = panther_training_steps)    
        panther_training_steps = panther_training_steps + steps
        
        #Adding new best responses to the tests
        W2.append(pelican_agent_filepath)
        W1.append(panther_agent_filepath)
        
        #Computing the payoff matrices and solving the corresponding LPs
        P1 = compute_payoff_matrix('pelican',
                                   model_type,
                                   policy,
                                   pelican_env,
                                   pelican_models,
                                   W2)
        P2 = compute_payoff_matrix('panther',
                                   model_type,
                                   policy,
                                   panther_env,
                                   panther_models,
                                   W1)
        mixture1,exit_code1,exit_string1 = lcp.lemkelcp(P1,np.zeros((len(W2),)))
        mixture2,exit_code2,exit_string2 = lcp.lemkelcp(P2,np.zeros((len(W1),)))
        if exit_code1 != 0 or exit_code2 != 0:
            print('Cannot solve the LPs...')
            break

    #Saving final version of the agents
    agent_filepath ,_, _= helper.save_model_with_env_settings(exp_path, pelican_model, pelican_model_type, pelican_env, basicdate)
    agent_filepath ,_, _= helper.save_model_with_env_settings(exp_path, panther_model, panther_model_type, panther_env, basicdate)

    logger.info('Training pelican total steps: ' + str(pelican_training_steps))
    logger.info('Training panther total steps: ' + str(panther_training_steps))
    # Make video 
    video_path =  os.path.join(exp_path, 'test_deep_pnm.mp4') 
    basewidth,hsize = helper.make_video(pelican_model, pelican_env, video_path)
    return video_path, basewidth, hsize

if __name__ == '__main__':
    basicdate = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    basepath = 'data/agents/models'
    exp_name = 'test_' + basicdate
    exp_path = os.path.join(basepath, exp_name)

    logger.info(exp_path)

    # run_deep_pnm(exp_name,exp_path,basicdate)
    run_deep_pnm(exp_name,
                 exp_path,
                 basicdate,
                  pelican_testing_interval = 1000,
                  pelican_max_learning_steps = 50000,
                  panther_testing_interval = 1000,
                  panther_max_learning_steps = 50000,
                  deep_pnm_iterations = 200,
                  model_type = 'PPO2',
                  log_to_tb = True,
                  image_based = False)
    # python deep_pnm.py 