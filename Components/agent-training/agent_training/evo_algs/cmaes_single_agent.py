from gym_plark.envs.plark_env import PlarkEnv
from gym_plark.envs.plark_env_sparse import PlarkEnvSparse
from plark_game.agents.basic.panther_nn import PantherNN
from plark_game.agents.basic.pelican_nn import PelicanNN
from agent_training import helper
import evo_utils

from deap import creator, base, benchmarks, cma, tools, algorithms

import numpy as np
import copy
import time

def save_video(genome, agent, env, num_steps, file_name='evo_run.mp4'):

    if genome is not None:
        agent.set_weights(genome)
    video_path = '/' + file_name
    helper.make_video_plark_env(agent, env, video_path, n_steps=num_steps)

def evaluate(genome, config_file_path, driving_agent, normalise_obs, domain_params_in_obs,
             num_trials, num_inputs, num_hidden_layers, neurons_per_hidden_layer):

    #Instantiate the env
    env = PlarkEnvSparse(config_file_path=config_file_path, image_based=False,
                         driving_agent=driving_agent, normalise=normalise_obs,
                         domain_params_in_obs=domain_params_in_obs)

    if trained_agent == 'panther':
        agent = PantherNN(num_inputs=num_inputs, num_hidden_layers=num_hidden_layers,
                          neurons_per_hidden_layer=neurons_per_hidden_layer)
    else:
        agent = PelicanNN(num_inputs=num_inputs, num_hidden_layers=num_hidden_layers,
                          neurons_per_hidden_layer=neurons_per_hidden_layer)

    agent.set_weights(genome)

    reward = 0

    for i in range(num_trials):
        env.reset()

        obs = env._observation()
        trial_reward = 0
        step_num = 0
        while True:
            #print("Step num:", step_num)
            action = agent.getAction(obs)
            obs, r, done, info = env.step(action)
            trial_reward += r
            step_num += 1
            if done:
                step_num = 0
                #print(info['status'])
                break
        reward += trial_reward

    #Average trial reward
    reward /= num_trials

    #agent.save_agent(obs_normalise=normalise_obs, domain_params_in_obs=domain_params_in_obs)

    #print("Finished at step num:", step_num)
    #print("Reward:", reward)
    #print("Status:", info['status'])

    return [reward]

if __name__ == '__main__':

    #Env variables
    #config_file_path = '/Components/plark-game/plark_game/game_config/10x10/nn/nn_single_agent_balanced.json'
    config_file_path = '/Components/plark-game/plark_game/game_config/20x20/balanced.json'
    #config_file_path = '/Components/plark-game/plark_game/game_config/00_26x33_02_07_30_12.json'
    #trained_agent = 'panther'
    trained_agent = 'pelican'
    normalise_obs = True
    domain_params_in_obs = True
    stochastic_actions = True

    random_panther_start_position = True
    random_pelican_start_position = True

    num_trials = 5

    #Instantiate dummy env and dummy agent
    #I need to do this to ascertain the number of weights needed in the optimisation
    #procedure
    dummy_env = PlarkEnvSparse(config_file_path=config_file_path,
                               driving_agent=trained_agent, normalise=normalise_obs,
                               domain_params_in_obs=domain_params_in_obs,
                               random_panther_start_position=random_panther_start_position,
                               random_pelican_start_position=random_pelican_start_position)

    #Neural net variables
    num_inputs = len(dummy_env._observation())
    num_hidden_layers = 0
    neurons_per_hidden_layer = 0

    if trained_agent == 'panther':
        dummy_agent = PantherNN(num_inputs=num_inputs, num_hidden_layers=num_hidden_layers,
                                neurons_per_hidden_layer=neurons_per_hidden_layer,
                                stochastic_actions=stochastic_actions)
    else:
        dummy_agent = PelicanNN(num_inputs=num_inputs, num_hidden_layers=num_hidden_layers,
                                neurons_per_hidden_layer=neurons_per_hidden_layer,
                                stochastic_actions=stochastic_actions)

    num_weights = dummy_agent.get_num_weights()

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate, config_file_path=config_file_path,
                     driving_agent=trained_agent, normalise_obs=normalise_obs,
                     domain_params_in_obs=domain_params_in_obs, num_trials=num_trials,
                     num_inputs=num_inputs, num_hidden_layers=num_hidden_layers,
                     neurons_per_hidden_layer=neurons_per_hidden_layer)

    #np.random.seed(108)

    #Initial location of distribution centre
    centroid = [0.0] * num_weights
    #Initial standard deviation of the distribution
    init_sigma = 1.0
    #Number of children to produce at each generation
    #lambda_ = 20 * num_weights
    lambda_ = 100
    strategy = cma.Strategy(centroid=centroid, sigma=init_sigma, lambda_=lambda_)

    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    parallelise = True
    if parallelise:
        import multiprocessing

        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)

    num_genomes_in_hof = 1
    hof = evo_utils.HallOfFamePriorityYoungest(num_genomes_in_hof)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    num_gens = 1000
    dump_every = 25
    population, logbook = evo_utils.eaGenerateUpdate(toolbox, ngen=num_gens,
                                                     stats=stats, halloffame=hof,
                                                     dump_every=dump_every,
                                                     obs_normalise=normalise_obs,
                                                     domain_params_in_obs=domain_params_in_obs,
                                                     dummy_agent=dummy_agent)

    #Save video of best agent
    #save_video(hof[0], dummy_agent, dummy_env, num_steps=200, file_name='hof_best_agent.mp4')

    #Save best agent
    dummy_agent.set_weights(hof[0])
    dummy_agent.save_agent(obs_normalise=normalise_obs,
                           domain_params_in_obs=domain_params_in_obs)
