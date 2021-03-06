from gym_plark.envs.plark_env_sparse import PlarkEnvSparse
from plark_game.agents.basic.panther_nn import PantherNN
from plark_game.agents.basic.pelican_nn import PelicanNN
from agent_training import helper
from plark_game.classes.newgame import *
import json
import os

class RuleBasedGame(Newgame):

    def __init__(self, game_config, **kwargs):
        super().__init__(game_config, **kwargs)

        self.relative_basic_agents_filepath = '../agents/basic'
        self.import_agents()

        # Load agents
        relative_basic_agents_filepath = os.path.join(os.path.dirname(__file__), self.relative_basic_agents_filepath)
        relative_basic_agents_filepath = os.path.normpath(relative_basic_agents_filepath)


        self.pelicanAgent = load_agent(self.pelican_parameters['agent_filepath'], 
                                       self.pelican_parameters['agent_name'],
                                       relative_basic_agents_filepath,self)
        self.pantherAgent = load_agent(self.panther_parameters['agent_filepath'], 
                                       self.panther_parameters['agent_name'],
                                       relative_basic_agents_filepath,self)

        # Game state variables
        self.default_game_variables()

        # Create UI objects and render game. This must be the last thing in the __init__
        self.render_height = self.pelican_parameters['render_height']
        self.render_width = self.pelican_parameters['render_width']

        self.reset_game()
        self.render(self.render_width,self.render_height,"PANTHER")

if __name__ == '__main__':

    #Env variables
    config_file_path = '/Components/plark-game/plark_game/game_config/10x10/balanced.json'
    driving_agent = 'pelican'

    random_panther_start_position = True
    random_pelican_start_position = True

    with open(config_file_path) as json_file:
        game_config = json.load(json_file)
    game_config["render_settings"]["output_view_all"] = False
    game_config["game_settings"]["driving_agent"] = ""
    game = RuleBasedGame(game_config)

    game.reset_game()

    while True:
        game_state, _ = game.game_step(None)
        if game_state == "PELICANWIN" or game_state == "BINGO" or \
           game_state == "WINCHESTER" or game_state == "ESCAPE":
            break

    #video_path = '/load_evo_non_driving_new.mp4'
    #helper.make_video_plark_env(pelican_dummy_agent, dummy_env, video_path, n_steps=200)
