from plark_game.classes.agent import Agent
from plark_game.classes.observation import Observation

import numpy as np
import torch

import datetime
import os
import json
import csv

#One should never instantiate an NNAgent - one should always instantiate one of its
#subclasses

class NNAgent(Agent):

    def __init__(self, num_inputs, num_outputs,
                 num_hidden_layers=0, neurons_per_hidden_layer=0,
                 file_dir_name=None, agent_type=None, game=None,
                 stochastic_actions=False, driving_agent=True,
                 in_tournament=False):

        self.agent_type = agent_type
        self.stochastic_actions = stochastic_actions
        self.driving_agent = driving_agent

        self.in_tournament = in_tournament

        #For reading and writing models
        self.base_path = '/data/agents/evo_models/'

        #Number of outputs is always given via the subclass call to this constructor
        self.num_outputs = num_outputs

        #If file directory name is given, read from file
        if file_dir_name is not None:
            metadata, genotype = self._read_agent_from_file(file_dir_name)

            self.num_inputs = metadata['num_inputs']
            self.num_hidden_layers = metadata['num_hidden_layers']
            self.neurons_per_hidden_layer = metadata['neurons_per_hidden_layer']
            self.stochastic_actions = metadata['stochastic_actions']

            obs_kwargs = {}
            obs_kwargs['driving_agent'] = self.agent_type
            obs_kwargs['normalise'] = metadata['normalise']
            obs_kwargs['domain_params_in_obs'] = metadata['domain_params_in_obs']
            self.obs_kwargs = obs_kwargs

            if not self.in_tournament:
                assert game is not None, "Need to hand NewGame object to NNAgent " \
                    "constructor in order to build the Observation class"

                self.observation = Observation(game, **obs_kwargs)

            #Build neural net
            self._build_nn()

            #Set read genotype as weights
            self.set_weights(genotype)

        else:
            #Check that num_inputs is not None
            if num_inputs is None:
                print('One needs to give either a number of inputs or a file directory'
                      ' name to build an NNAgent')
                exit()

            self.num_inputs = num_inputs
            self.num_hidden_layers = num_hidden_layers
            self.neurons_per_hidden_layer = neurons_per_hidden_layer

            self.observation = None

            #Build neural net
            self._build_nn()


    def _build_nn(self):

        layers = []
        if self.num_hidden_layers == 0:
            layers.append(torch.nn.Linear(self.num_inputs, self.num_outputs))

        else:
            layers.append(torch.nn.Linear(self.num_inputs, self.neurons_per_hidden_layer))
            #Hidden layers have ReLU activation
            layers.append(torch.nn.ReLU())

            for i in range(self.num_hidden_layers-1):
                layers.append(torch.nn.Linear(self.neurons_per_hidden_layer,
                                              self.neurons_per_hidden_layer))
                layers.append(torch.ReLU())

            layers.append(torch.nn.Linear(self.neurons_per_hidden_layer, self.num_outputs))

        #Final layer goes through a softmax
        layers.append(torch.nn.Softmax(dim=0))

        self.nn = torch.nn.Sequential(*layers).double()

    #Takes a list, passes through the network and returns a list
    def _forward_pass(self, x):
        x = torch.tensor(x, dtype=torch.float64)
        net_out = self.nn.forward(x)
        return net_out.tolist()

    #Randomly sample action from network output probability distribution
    def _sample_action(self, net_out):
        action_nums = list(range(len(net_out)))
        return np.random.choice(action_nums, p=net_out)

    #Get the most probable action from the network output probability distribution
    def _get_most_probable_action(self, net_out):
        return np.argmax(net_out)

    def getAction(self, state):

        #If state dictionary comes through, convert to numpy array.
        #This will happen when the NNAgent is the non-driving agent.
        if not self.driving_agent:
            state = self.observation.get_observation(state)

        assert len(state) == self.num_inputs, "State length: {}, num inputs: {}" \
            .format(len(state), self.num_inputs)

        #Push state through network
        net_out = self._forward_pass(state)

        #Get action from nework output
        if self.stochastic_actions:
            action = self._sample_action(net_out)
        else:
            action = self._get_most_probable_action(net_out)

        if (not self.driving_agent) or self.in_tournament:
            action = self.action_lookup(action)

        return action

    #Stitch together state from tournament
    def getTournamentAction(self, obs, obs_normalised, domain_parameters,
                            domain_parameters_normalised, state):

        if self.obs_kwargs['normalise']:
            stitched_obs = obs_normalised
        else:
            stitched_obs = obs

        if self.obs_kwargs['domain_params_in_obs']:
            if self.obs_kwargs['normalise']:
                stitched_obs = np.concatenate((stitched_obs, domain_parameters_normalised))
            else:
                stitched_obs = np.concatenate((stitched_obs, domain_parameters))

        return self.getAction(stitched_obs)

    #Returns the number of weights
    def get_num_weights(self):
        num_weights = 0
        for layer in self.nn:
            for params in layer.parameters():
                num_weights += params.numel()
        return num_weights

    def print_weights(self):
        for layer in self.nn:
            for params in layer.parameters():
                print(params)

    def _set_weights_err_msg(self, weights_len, num_weights_required):
        return "Trying to set {} weights to an NN that requires {} weights" \
            .format(weights_len, num_weights_required)

    #Sets a list of weights
    def set_weights(self, new_weights):

        #Check new weights is of correct size
        num_weights_required = self.get_num_weights()
        assert num_weights_required == len(new_weights), \
                                       self._set_weights_err_msg(len(new_weights), \
                                                                 num_weights_required)

        weight_index = 0
        for layer in self.nn:
            for params in layer.parameters():

                #Slice out new weights
                p_weights = new_weights[weight_index : weight_index + params.numel()]
                weight_index += params.numel()

                #Resize and set new weights
                params.data = torch.tensor(np.reshape(p_weights, params.size()), \
                                           dtype=torch.float64)

    #Return weights as a 1d list
    def get_weights(self):
        weights = []
        for layer in self.nn:
            for params in layer.parameters():
                weights += params.flatten().tolist()
        return weights

    def _save_metadata(self, dir_path, player_type, obs_normalise, domain_params_in_obs):
        metadata = {}

        metadata['playertype'] = player_type
        metadata['normalise'] = obs_normalise
        metadata['domain_params_in_obs'] = domain_params_in_obs
        metadata['stochastic_actions'] = self.stochastic_actions

        metadata['num_inputs'] = self.num_inputs
        metadata['num_hidden_layers'] = self.num_hidden_layers
        metadata['neurons_per_hidden_layer'] = self.neurons_per_hidden_layer

        file_path = dir_path + '/metadata.json'

        with open(file_path, 'w') as outfile:
            json.dump(metadata, outfile)

    def _save_genotype(self, dir_path):
        #Save genotype as a csv - it is just a list
        file_path = dir_path + '/genotype.csv'

        with open(file_path, 'w') as outfile:
            csv_writer = csv.writer(outfile)
            csv_writer.writerow(self.get_weights())

    def _save_agent_to_file(self, player_type, obs_normalise, domain_params_in_obs):

        #Construct full directory path
        date_time = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        dir_name = player_type + '_' + date_time
        dir_path = self.base_path + dir_name

        #Create directory for model
        os.makedirs(dir_path, exist_ok=True)
        os.chmod(dir_path, 0o777)

        #Save metadata
        self._save_metadata(dir_path, player_type, obs_normalise, domain_params_in_obs)

        #Save genotype
        self._save_genotype(dir_path)

    def _read_metadata(self, metadata_filepath):
        with open(metadata_filepath, 'r') as metadata_file:
            metadata = json.load(metadata_file)
        return metadata

    def _read_genotype(self, genotype_filepath):
        with open(genotype_filepath, 'r') as genotype_file:
            reader = csv.reader(genotype_file)
            genotype = list(map(float, list(reader)[0]))
        return genotype

    def _read_agent_from_file(self, dir_name):
        dir_path = self.base_path + dir_name + '/'

        if self.in_tournament:
            dir_path = "/plark_ai_public" + dir_path

        #Read metadata
        metadata = self._read_metadata(dir_path + 'metadata.json')

        #Read genotype
        genotype = self._read_genotype(dir_path + 'genotype.csv')

        return metadata, genotype
