from plark_game.classes.pantherAgent import *
from plark_game.agents.basic.nn_agent import *
import numpy as np
import torch

class PantherNN(NNAgent):

    def __init__(self, num_inputs=None, num_hidden_layers=0, neurons_per_hidden_layer=0,
                 file_dir_name=None, game=None, stochastic_actions=False,
                 driving_agent=True, in_tournament=False):

        #If a file directory name is given, check file name is for a Panther
        if file_dir_name is not None:
            self.check_file_name(file_dir_name)

        super().__init__(num_inputs, len(ACTION_LOOKUP),
                         num_hidden_layers, neurons_per_hidden_layer,
                         file_dir_name, 'panther', game, stochastic_actions,
                         driving_agent, in_tournament)

    def action_lookup(self, action):
        return ACTION_LOOKUP[action]

    def save_agent(self, obs_normalise, domain_params_in_obs, file_name_suffix=''):
        self._save_agent_to_file('panther', obs_normalise, domain_params_in_obs,
                                 file_name_suffix)

    def check_file_name(self, model_dir):

        #The model directory should have 'pelican' in it
        if 'panther' not in model_dir:
            print("Model directory should have \'panther\' in it, are you sure this is a"
                  "panther model?")
            print("Model directory:", model_dir)
            exit()

