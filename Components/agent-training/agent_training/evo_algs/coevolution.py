from gym_plark.envs.plark_env_sparse import PlarkEnvSparse
from plark_game.agents.basic.panther_nn import PantherNN
from plark_game.agents.basic.pelican_nn import PelicanNN
from agent_training import helper

if __name__ == '__main__':

    #Env variables
    config_file_path = '/Components/plark-game/plark_game/game_config/10x10/nn/nn_coevolution_balanced.json'
    driving_agent = 'pelican'
    normalise_obs = True
    domain_params_in_obs = True
    stochastic_actions = False

    random_panther_start_position = True
    random_pelican_start_position = True

    #Instantiate dummy env and dummy agent
    #I need to do this to ascertain the number of weights needed in the optimisation
    #procedure
    dummy_env = PlarkEnvSparse(config_file_path=config_file_path,
                               driving_agent=driving_agent, normalise=normalise_obs,
                               domain_params_in_obs=domain_params_in_obs,
                               random_panther_start_position=random_panther_start_position,
                               random_pelican_start_position=random_pelican_start_position)
    game = dummy_env.env.activeGames[len(dummy_env.env.activeGames)-1] 

    #Neural net variables
    num_inputs = len(dummy_env._observation())
    num_hidden_layers = 0
    neurons_per_hidden_layer = 0

    #panther_dummy_agent = PantherNN(num_inputs=num_inputs, 
    #                                num_hidden_layers=num_hidden_layers, 
    #                                neurons_per_hidden_layer=neurons_per_hidden_layer)  

    pelican_dummy_agent = PelicanNN(file_dir_name='pelican_20210305_013925', game=game,
                                    driving_agent=True)  

    #Set agent
    #game.pelicanAgent = pelican_dummy_agent

    dummy_env.reset()

    '''
    max_num_steps = 1

    reward = 0
    obs = dummy_env._observation()
    for step_num in range(max_num_steps):
        action = panther_dummy_agent.getAction(obs)    
        obs, r, done, info = dummy_env.step(action)
        reward += r
        if done:
            break
    '''

    video_path = '/load_evo_non_driving_new.mp4'
    helper.make_video_plark_env(pelican_dummy_agent, dummy_env, video_path, n_steps=200)
