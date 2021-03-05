import os
import sys
import datetime
import json
import logging
import imageio
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from plark_game import classes
from gym_plark.envs import plark_env
from gym_plark.envs.plark_env_sparse import PlarkEnvSparse
from gym_plark.envs.plark_env import PlarkEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.env_checker import check_env
from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv, VecEnv
from copy import deepcopy

# PyTorch Stable Baselines
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv as SubprocVecEnv_Torch

from plark_game import classes
from plark_game.classes.environment import Environment
from plark_game.classes.pantherAgent_load_agent import Panther_Agent_Load_Agent
from plark_game.classes.pelicanAgent_load_agent import Pelican_Agent_Load_Agent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DEFAULT_FPS = 3   # Originally was 10
BASEWIDTH = 512 # Originally was 512, increase/decrease for higher/lower resolution

def model_label(modeltype,basicdate,modelplayer):
    label = modeltype + "_" + str(basicdate) + "_" + modelplayer
    return label

def make_new_model(model_type,policy,env, n_steps=100, tensorboard_log=None):
    if model_type.lower() == 'dqn':
        model = DQN(policy,env,tensorboard_log=tensorboard_log)
    elif model_type.lower() == 'ppo2':
        model = PPO2(policy,env,tensorboard_log=tensorboard_log)
    elif model_type.lower() == 'ppo':
        model = PPO(policy,env, n_steps=n_steps)
    elif model_type.lower() == 'a2c':
        model = A2C(policy,env,tensorboard_log=tensorboard_log)
    elif model_type.lower() == 'acktr':
        model = ACKTR(policy,env,tensorboard_log=tensorboard_log)
    return model

def train_until(model, env, victory_threshold=0.8, victory_trials=10, max_seconds=120, testing_interval=200, tb_writer=None, tb_log_name=None):
    model.set_env(env)
    steps = 0
    max_victory_fraction = 0.0
    initial_time = datetime.datetime.now()
    current_time = datetime.datetime.now()
    elapsed_seconds = (current_time - initial_time).total_seconds()
    while elapsed_seconds < max_seconds:
        logger.info("Training for {} steps".format(testing_interval))
        before_learning = datetime.datetime.now()
        model.learn(testing_interval)
        after_learning = datetime.datetime.now()
        steps = steps + testing_interval

        logger.info("Learning took {:.2f} seconds".format((after_learning - before_learning).total_seconds()))
        
        logger.info("Checking victory")
        victory_count, avg_reward = check_victory(model, env, trials=victory_trials)
        after_check = datetime.datetime.now()
        logger.info("Victory check took {:.2f} seconds".format((after_check - after_learning).total_seconds()))
        victory_fraction = float(victory_count)/victory_trials
        logger.info("Won {} of {} evaluations ({:.2f})".format(victory_count, victory_trials, victory_fraction))
        max_victory_fraction = max(max_victory_fraction, victory_fraction)

        if tb_writer is not None and tb_log_name is not None:
            tb_writer.add_scalar('{}_avg_reward'.format(tb_log_name), avg_reward, steps)
            tb_writer.add_scalar('{}_victory_count'.format(tb_log_name), victory_count, steps)
            tb_writer.add_scalar('{}_victory_fraction'.format(tb_log_name), victory_fraction, steps)
        
        current_time = datetime.datetime.now()
        elapsed_seconds = (current_time - initial_time).total_seconds()
        
        if victory_fraction >= victory_threshold:
            logger.info("Achieved victory threshold after {} steps".format(steps))
            break
    logger.info("Achieved max victory fraction {:.2f} after {} seconds ({} steps)".format(max_victory_fraction, elapsed_seconds, steps))
    achieved_goal = max_victory_fraction >= victory_threshold
    return achieved_goal, steps, elapsed_seconds

def check_victory(model, env, trials):

    if isinstance(env, SubprocVecEnv_Torch):
        list_of_reward, n_steps, victories = evaluate_policy_torch(model, env, n_eval_episodes=trials, deterministic=False, render=False, callback=None, reward_threshold=None, return_episode_rewards=True)
    else: 
        list_of_reward, n_steps, victories = evaluate_policy(model, env, n_eval_episodes=trials, deterministic=False, render=False, callback=None, reward_threshold=None, return_episode_rewards=True)

    logger.info("===================================================")
    modelplayer = env.get_attr('driving_agent')[0]
    logger.info('In check_victory, driving_agent: %s' % modelplayer)

    avg_reward = float(sum(list_of_reward))/len(list_of_reward)
    victory_count = len([v for v in victories if v == True])
    victory_prop = float(victory_count)/len(victories) 
    logger.info('victory_prop: %.2f (%s out of %s); avg_reward: %.3f' % 
                                                      (victory_prop,
                                                       victory_count, 
                                                       len(victories),
                                                       avg_reward
                                                       ))
    logger.info("===================================================")

    return victory_prop, avg_reward, 

def evaluate_policy_torch(model, env, n_eval_episodes, deterministic=True, 
                                                       render=False, 
                                                       callback=None, 
                                                       reward_threshold=None, 
                                                       return_episode_rewards=False):
    """
    Modified from https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/common/evaluation.html#evaluate_policy
    to return additional info
    """
    logger.debug("Evaluating policy")
    episode_rewards, episode_lengths, victories = [], [], []
    obs = env.reset()
    episodes_reward = [0.0 for _ in range(env.num_envs)]
    episodes_len = [0.0 for _ in range(env.num_envs)]
    state = None

    logger.debug("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    logger.debug("In evaluate_policy_torch, n_eval_episodes: %s" % n_eval_episodes)
    logger.debug("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    while len(episode_rewards) < n_eval_episodes:
        action, state = model.predict(obs, state=state, deterministic=deterministic)
        obs, rewards, dones, _infos = env.step(action)
        for i in range(len(rewards)):
            episodes_reward[i] += rewards[i]
            episodes_len[i] += 1
        if render:
            env.render()
        if dones.any():
           for i, d in enumerate(dones):
               if d:
                   info = _infos[i]
                   victory = info['result'] == "WIN"
                   victories.append(victory)
                   episode_rewards.append(episodes_reward[i])
                   episode_lengths.append(episodes_len[i])
                   episodes_reward[i] = 0
                   episodes_len[i] = 0

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: {:.2f} < {:.2f}".format(mean_reward, reward_threshold)
    if return_episode_rewards:
        return episode_rewards, episode_lengths, victories
    return mean_reward, std_reward, victories

def evaluate_policy(model, env, n_eval_episodes=4, deterministic=True, render=False, callback=None, reward_threshold=None, return_episode_rewards=False):
    """
    Modified from https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/common/evaluation.html#evaluate_policy
    to return additional info
    """
    logger.debug("Evaluating policy")
    episode_rewards, episode_lengths, victories = [], [], []
    for ep in range(n_eval_episodes):
        logger.debug("Evaluating episode {} of {}".format(ep, n_eval_episodes))
        obs = env.reset()
        ep_done, state = False, None

        episode_length = 0
        episode_reward = 0.0
        if isinstance(env, VecEnv) or isinstance(env, SubprocVecEnv_Torch):
            episodes_reward = [0.0 for _ in range(env.num_envs)]
        else:
            episodes_reward = [0.0]

        victory = False

        while not ep_done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, rewards, dones, _infos = env.step(action)

            if not isinstance(env, VecEnv):
                rewards = [rewards]
                dones = np.array([dones])
                _infos = [_infos]

            episode_length += 1
            for i in range(len(rewards)):
                episodes_reward[i] += rewards[i]

            if callback is not None:
                callback(locals(), globals())

            if episode_length > 1000:
                logger.warning("Episode over 1000 steps.")
                
            if render:
                env.render()
            if any(dones):
                first_done_index = dones.tolist().index(True)
                info = _infos[first_done_index]
                victory = info['result'] == "WIN"
                episode_reward = episodes_reward[first_done_index]
                ep_done = True
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        victories.append(victory)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: {:.2f} < {:.2f}".format(mean_reward, reward_threshold)

    if return_episode_rewards:
        return episode_rewards, episode_lengths, victories
    return mean_reward, std_reward, victories

def get_env(driving_agent, 
            config_file_path, 
            opponent=None, 
            image_based=False, 
            random_panther_start_position=True,
            random_pelican_start_position=True,
            max_illegal_moves_per_turn = 3,
            sparse=False,
            normalise=False,
            is_in_vec_env=False):

    params = dict(driving_agent = driving_agent,
                  config_file_path = config_file_path,
                  image_based = image_based,
                  random_panther_start_position = random_panther_start_position,
                  random_pelican_start_position = random_pelican_start_position,
                  max_illegal_moves_per_turn = max_illegal_moves_per_turn,
                  normalise=normalise,
                  is_in_vec_env=is_in_vec_env)
    
    if opponent != None and driving_agent == 'pelican':
        params.update(panther_agent_filepath = opponent)
    elif opponent != None and driving_agent == 'panther':
        params.update(pelican_agent_filepath = opponent)
    if sparse:
        return PlarkEnvSparse(**params)
    else:
        return PlarkEnv(**params)

def get_envs(driving_agent, 
             config_file_path, 
             opponents=[], 
             num_envs=1,
             image_based=False, 
             random_panther_start_position=True,
             random_pelican_start_position=True,
             max_illegal_moves_per_turn=3,
             sparse=False,
             vecenv=True,
             mixture=None,
             normalise=False):

    params = dict(driving_agent = driving_agent,
                  config_file_path = config_file_path,
                  image_based = image_based,
                  random_panther_start_position = random_panther_start_position,
                  random_pelican_start_position = random_pelican_start_position,
                  max_illegal_moves_per_turn = max_illegal_moves_per_turn,
                  sparse = sparse,
                  normalise = normalise,
                  is_in_vec_env=vecenv)

    if len(opponents) == 1:
        params.update(opponent=opponents[0])
    if vecenv == False:
        return get_env(**params)
    elif len(opponents) < 2:
        return SubprocVecEnv_Torch([lambda:get_env(**params) for _ in range(num_envs)])
    elif len(opponents) >= 2:
        opponents = np.random.choice(opponents, size = num_envs, p = mixture)
        params_list = []
        for o in opponents:
            params.update(opponent=o)
            params_list.append(deepcopy(params))
        return SubprocVecEnv_Torch([lambda:get_env(**params) for params in params_list])

# Save model base on env
def save_model_with_env_settings(basepath,model,modeltype,env,basicdate=None):
    if basicdate is None:
        basicdate = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    if isinstance(env, VecEnv) or isinstance(env, SubprocVecEnv_Torch):
        modelplayer = env.get_attr('driving_agent')[0]
        render_height = env.get_attr('render_height')[0]
        render_width = env.get_attr('render_width')[0]
        image_based = env.get_attr('image_based')[0]
    else:
        modelplayer = env.driving_agent 
        render_height = env.render_height
        render_width = env.render_width
        image_based = env.image_based
    model_path,model_dir, modellabel = save_model(basepath,model,modeltype,modelplayer,render_height,render_width,image_based,basicdate)
    return model_path,model_dir, modellabel

# Saves model and metadata
def save_model(basepath,model,modeltype,modelplayer,render_height,render_width,image_based,basicdate=None):
    if basicdate is None:
        basicdate = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    modellabel = model_label(modeltype,basicdate,modelplayer)
    model_dir = os.path.join(basepath, modellabel)
    logger.info("Checking folder: " + model_dir)
    os.makedirs(model_dir, exist_ok=True)
    os.chmod(model_dir, 0o777)
    logger.info("Saving Metadata")
    print(model.env)
    if isinstance(model.env, VecEnv) or isinstance(model.env, SubprocVecEnv_Torch):
        normalise = model.env.get_attr('normalise')[0]
        domain_params_in_obs = model.env.get_attr('domain_params_in_obs')[0]
    else:
        normalise = model.env.normalise
        domain_params_in_obs = model.env.domain_params_in_obs
    save_model_metadata(model_dir,modeltype,modelplayer,basicdate,render_height,render_width,image_based, normalise, domain_params_in_obs)

    logger.info("Saving Model")
    model_path = os.path.join(model_dir, modellabel + ".zip")
    model.save(model_path)
    logger.info('model_dir: '+model_dir)  
    logger.info('model_path: '+model_path) 
    
    return model_path,model_dir, modellabel

## Used for generating the json header file which holds details regarding the model.
## This will be used when playing the game from the GUI.
def save_model_metadata(model_dir,modeltype,modelplayer,dateandtime,render_height,render_width,image_based, normalise, domain_params_in_obs):
    jsondata = {}
    jsondata['algorithm'] =  modeltype
    jsondata['date'] = str(dateandtime)
    jsondata['agentplayer'] = modelplayer
    jsondata['render_height'] = render_height
    jsondata['render_width'] = render_width
    jsondata['image_based'] = image_based
    jsondata['normalise'] = normalise
    jsondata['domain_params_in_obs'] = domain_params_in_obs
    json_path = os.path.join(model_dir, 'metadata.json')
    with open(json_path, 'w') as outfile:
        json.dump(jsondata, outfile)    

    logger.info('json saved to: '+json_path)


## Custom Model Evaluation Method for evaluating Plark games. 
## Does require changes to how data is passed back from environments. 
## Instead of using return ob, reward, done, {} use eturn ob, reward, done, {game.state}
def custom_eval(model, env, n_eval_episodes=10, deterministic=True,
                    render=False, callback=None, reward_threshold=None,
                    return_episode_rewards=False, player_type="PELICAN"):
    """
    Runs policy for `n_eval_episodes` episodes and returns average reward.
    This is made to work only with one env.

    :param model: (BaseRLModel) The RL agent you want to evaluate.
    :param env: (gym.Env or VecEnv) The gym environment. In the case of a `VecEnv`
        this must contain only one environment.
    :param n_eval_episodes: (int) Number of episode to evaluate the agent
    :param deterministic: (bool) Whether to use deterministic or stochastic actions
    :param render: (bool) Whether to render the environment or not
    :param callback: (callable) callback function to do additional checks,
        called after each step.
    :param reward_threshold: (float) Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: (bool) If True, a list of reward per episode
        will be returned instead of the mean.
    :return: (float, float) Mean reward per episode, std of reward per episode
        returns ([float], [int]) when `return_episode_rewards` is True
    """
    
    if player_type == "PELICAN":
        WINCONDITION = "PELICANWIN"
    if player_type == "PANTHER":
        WINCONDITION = "PANTHERWIN"
        
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"
    totalwin = 0
    episode_rewards, episode_lengths = [], []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, _info = env.step(action)
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
            if WINCONDITION in _info:
                totalwin = totalwin + 1
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    if reward_threshold is not None:
        assert mean_reward > reward_threshold, 'Mean reward below threshold: '\
                                         '{:.2f} < {:.2f}'.format(mean_reward, reward_threshold)
    if return_episode_rewards:
        return episode_rewards, episode_lengths, totalwin
    return mean_reward, std_reward, totalwin

def loadAgent(filepath, algorithm_type):
    try:
        if algorithm_type.lower() == 'dqn':
            model = DQN.load(filepath)
        elif algorithm_type.lower() == 'ppo2': 
            model = PPO2.load(filepath)
        elif algorithm_type.lower() == 'ppo': 
            model = PPO.load(filepath)
        elif algorithm_type.lower() == 'a2c':
            model = A2C.load(filepath)
        elif algorithm_type.lower() == 'acktr':
            model = ACKTR.load(filepath)
        return model
    except:
        raise ValueError('Error loading agent. File : "' + filepath + '" does not exsist' )

def og_load_driving_agent_make_video(pelican_agent_filepath, pelican_agent_name, panther_agent_filepath, panther_agent_name, config_file_path='/Components/plark-game/plark_game/game_config/10x10/balanced.json',video_path='/Components/plark_ai_flask/builtangularSite/dist/assets/videos'):
    """
    Method for loading and agent, making and environment, and making a video. Mainly used in notebooks. 
    """
    logger.info("Load driving agent make viedo - pelican agent filepast = " + pelican_agent_filepath)
    basicdate = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    video_file = basicdate+'.mp4' 
    video_file_path = os.path.join(video_path, video_file) 
    os.makedirs(video_path, exist_ok=True)
    files = os.listdir(pelican_agent_filepath)
    if len(files) > 0:
        for f in files:
            if '.zip' in f:
                # load model
                metadata_filepath = os.path.join(pelican_agent_filepath, 'metadata.json')
                agent_filepath = os.path.join(pelican_agent_filepath, f)
                

                with open(metadata_filepath) as f:
                    metadata = json.load(f)
                logger.info('Playing against:'+agent_filepath)  
                if metadata['agentplayer'] == 'pelican':        
                    pelican_agent = Pelican_Agent_Load_Agent(agent_filepath, metadata['algorithm'])
                    pelican_model = pelican_agent.model

                    env = plark_env.PlarkEnv(driving_agent='pelican',panther_agent_filepath=panther_agent_filepath, panther_agent_name=panther_agent_name, config_file_path=config_file_path)
                    basewidth,hsize = make_video(pelican_model,env,video_file_path)
                    logger.info("This is the environment variable " + str(env))

                elif metadata['agentplayer'] == 'panther':
                    raise ValueError('No Pelican agent found in ', pelican_agent_filepath) 
            
    else:
        raise ValueError('no agent found in ', files)

    return video_file, env.status,video_file_path

def load_driving_agent_make_video(pelican_agent_filepath, pelican_agent_name, panther_agent_filepath, panther_agent_name, config_file_path='/Components/plark-game/plark_game/game_config/10x10/balanced.json',video_path='/Components/plark_ai_flask/builtangularSite/dist/assets/videos',basic_agents_filepath='/Components/plark-game/plark_game/agents/basic',  renderWidth=None, renderHeight=None):
    """
    Method for loading and agent, making and environment, and making a video. Mainly used from flask server.
    """
    basicdate = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    video_file = basicdate+'.mp4' 
    video_file_path = os.path.join(video_path, video_file) 
    os.makedirs(video_path, exist_ok=True)

    kwargs = {
    'driving_agent': "pelican",
    'panther_agent_filepath': panther_agent_filepath,
    'panther_agent_name': panther_agent_name,
    }

    game_env = Environment()
    game_env.createNewGame(config_file_path, **kwargs)
    game = game_env.activeGames[len(game_env.activeGames)-1]
    
    agent = classes.load_agent(pelican_agent_filepath,pelican_agent_name,basic_agents_filepath,game,**kwargs)

    if renderHeight is None:
        renderHeight = game.pelican_parameters['render_height']
    if renderHeight is None:
        renderWidth = game.pelican_parameters['render_width']
    
    basewidth, hsize = new_make_video(agent, game, video_file_path, renderWidth, renderHeight)

    return video_file, game.gameState ,video_file_path

def make_video_VEC_ENV(model, env, video_file_path,n_steps = 100,fps=DEFAULT_FPS,deterministic=False,basewidth=BASEWIDTH,verbose=False):
    # Test the trained agent
    # This is when you have a stable baselines model and an gym env
    obs = env.reset()
    writer = imageio.get_writer(video_file_path, fps=fps) 
    hsize = None
    for step in range(n_steps):

        #######################################################################
        # Get image and comvert back to PIL.Image
        try:
            image = PIL.Image.fromarray(env.render(mode='rgb_array'))
        except:
            print("NOT WORKED TO CONVERT BACK TO PIL")
        #######################################################################

        action, _ = model.predict(obs, deterministic=deterministic)
    
        obs, reward, done, info = env.step(action)
        if verbose:
            logger.info("Step: "+str(step)+" Action: "+str(action)+' Reward:'+str(reward)+' Done:'+str(done))

        if hsize is None:
            wpercent = (basewidth/float(image.size[0]))
            hsize = int((float(image.size[1])*float(wpercent)))
        res_image = image.resize((basewidth,hsize), PIL.Image.ANTIALIAS)
        writer.append_data(np.copy(np.array(res_image)))
    writer.close()  
    return basewidth,hsize      


def make_video(model,env,video_file_path,n_steps = 10000,fps=DEFAULT_FPS,deterministic=False,basewidth=BASEWIDTH,verbose =False):
    # Test the trained agent
    # This is when you have a stable baselines model and an gym env
    obs = env.reset()
    writer = imageio.get_writer(video_file_path, fps=fps) 
    hsize = None
    for step in range(n_steps):
        image = env.render(view='ALL')
        action, _ = model.predict(obs, deterministic=deterministic)
    
        obs, reward, done, info = env.step(action)
        if verbose:
            logger.info("Step: "+str(step)+" Action: "+str(action)+' Reward:'+str(reward)+' Done:'+str(done))

        if hsize is None:
            wpercent = (basewidth/float(image.size[0]))
            hsize = int((float(image.size[1])*float(wpercent)))
        res_image = image.resize((basewidth,hsize), PIL.Image.ANTIALIAS)
        writer.append_data(np.copy(np.array(res_image)))
        if done:
            if verbose:
                logger.info("Goal reached:, reward="+ str(reward))
            break
    writer.close()  
    return basewidth,hsize      

def new_make_video(agent,game,video_file_path,renderWidth, renderHeight, n_steps = 10000,fps=DEFAULT_FPS,deterministic=False,basewidth=BASEWIDTH,verbose =False):
    # Test the trained agent
    # This is when you have a plark game agent and a plark game 
    game.reset_game()
    writer = imageio.get_writer(video_file_path, fps=fps) 
    hsize = None
    for step in range(n_steps):
        image = game.render(renderWidth, renderHeight, view='ALL')
        game_state_dict = game._state("PELICAN")
        action = agent.getAction(game_state_dict)
        game.game_step(action)
       
        if hsize is None:
            wpercent = (basewidth/float(image.size[0]))
            hsize = int((float(image.size[1])*float(wpercent)))
        res_image = image.resize((basewidth,hsize), PIL.Image.ANTIALIAS)
        writer.append_data(np.copy(np.array(res_image)))

        if game_state_dict['game_state'] == "PELICANWIN" or game_state_dict['game_state'] == "WINCHESTER" or game_state_dict['game_state'] == "ESCAPE": 
            break
    writer.close()  
    return basewidth,hsize  

def make_video_plark_env(agent, env, video_file_path, n_steps=10000, fps=DEFAULT_FPS, deterministic=False, basewidth=BASEWIDTH, verbose=False):

    print("Recording video...")

    # Test the trained agent
    # This is when you have a plark game agent and a plark env
    env.reset()
    writer = imageio.get_writer(video_file_path, fps=fps) 
    hsize = None

    #Render first image
    image = env.render(view='ALL')
    if hsize is None:
        wpercent = (basewidth/float(image.size[0]))
        hsize = int((float(image.size[1])*float(wpercent)))
    res_image = image.resize((basewidth,hsize), PIL.Image.ANTIALIAS)
    writer.append_data(np.copy(np.array(res_image)))

    obs = env._observation()
    for step in range(n_steps):
        action = agent.getAction(obs)
        obs, _, done, info = env.step(action)
        image = env.render(view='ALL')
       
        if hsize is None:
            wpercent = (basewidth/float(image.size[0]))
            hsize = int((float(image.size[1])*float(wpercent)))
        res_image = image.resize((basewidth,hsize), PIL.Image.ANTIALIAS)
        writer.append_data(np.copy(np.array(res_image)))

        if done:
            print(info['status'])
            break

    writer.close()  

    return basewidth,hsize  

def get_fig(df):
    fig, (ax1,ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True)
    df[['NE_Payoff', 'Pelican_BR_Payoff', 'Panther_BR_Payoff']].plot(ax = ax1, fontsize = 6)
    ax1.legend(loc = 'upper right',prop = {'size': 7})
    ax1.set_ylabel('Payoff to Pelican')
    df[['Pelican_supp_size', 'Panther_supp_size']].plot(kind = 'bar', ax = ax2, rot = 0)
    ax2.tick_params(axis = 'x', which = 'both', labelsize = 6)
    ax2.legend(loc = 'upper left', prop = {'size': 8})

def get_fig_with_exploit(df, exploit_df):
    # exploit_df should have the iteration as the index

    fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 7.5), sharex=True)
    cols = ['Pelican_BR_Payoff', 'NE_Payoff', 'Panther_BR_Payoff']
    df[cols].plot(ax=ax1,fontsize=10, color=['r', 'g', 'b'], linewidth=2)

    # might consider:
    # unstack and https://stackoverflow.com/questions/40256820/plotting-a-column-containing-lists-using-pandas
    tmp = exploit_df['panther_median'].reset_index()
    tmp.columns = ['x','y']
    tmp.plot(x='x', y='y', ax=ax1, color=['b'], kind='scatter')
    # exploit_df['pelican_median'].reset_index().plot(ax=ax1, color=['r'], kind='scatter')
    tmp = exploit_df['pelican_median'].reset_index()
    tmp.columns = ['x','y']
    tmp.plot(x='x', y='y', ax=ax1, color=['r'], kind='scatter')

    # ax1.legend(loc='lower right',prop={'size': 12})
    ax1.legend(loc='lower center',prop={'size': 12})
    ax1.set_ylabel('Payoff to Pelican', fontsize=14)
    df[['Pelican_supp_size', 'Panther_supp_size']].plot(kind='bar', ax=ax2, rot=0, color=['r', 'b'])
    ax2.tick_params(axis='x', which='both', labelsize=9)
    ax2.legend(loc='upper left',prop={'size': 14})
    ax2._get_lines.get_next_color()
    ax2.set_xlabel('PNM Iteration', fontsize=14)

