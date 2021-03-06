import os
import json
from plark_game.classes.newgame import *

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

def create_rule_based_game(config_file_path):
    with open(config_file_path) as json_file:
        game_config = json.load(json_file)
    game_config["render_settings"]["output_view_all"] = False
    game_config["game_settings"]["driving_agent"] = ""
    return RuleBasedGame(game_config)
