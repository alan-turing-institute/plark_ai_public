# +
# #!pip install pycddlib
# #!pip install stable-baselines3
# The following is needed on the DGX:
# #!pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

import sys
sys.path.insert(1, '/Components/')

import datetime
import numpy as np
import pandas as pd
import os
import glob
from agent_training import helper
from agent_training import lp_solve
import matplotlib.pyplot as plt
import time
import itertools
# -

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PNM():

    # Python constructor to initialise the players within the gamespace.
    # These are subsequently used throughout the game.
    def __init__(self, **kwargs):

        # ######################################################################
        # PARAMS
        # ######################################################################

        # Path related args
        config_file_path                = kwargs.get('config_file_path', '/Components/plark-game/plark_game/game_config/10x10/balanced.json')
        self.basicdate                  = kwargs.get('basicdate', str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
        basepath                        = kwargs.get('basepath', '/data/agents/models')
        
        # Training, evaluation steps, opponents, etc:
        self.training_steps             = kwargs.get('training_steps', 100) # N training steps per PNM iteration for each agent
        self.payoff_matrix_trials       = kwargs.get('payoff_matrix_trials', 25) # N eval steps per pairing
        self.max_n_opponents_to_sample  = kwargs.get('max_n_opponents_to_sample', 30) # so 28 max for 7 parallel envs
        self.retraining_prob            = kwargs.get('retraining_prob', 0.8) # Probability with which a policy is bootstrapped.
        self.max_pnm_iterations         = kwargs.get('max_pnm_iterations', 100) # N PNM iterations
        self.stopping_eps               = kwargs.get('stopping_eps', 0.001) # required quality of RB-NE
        
        self.testing_interval           = kwargs.get('testing_interval', 10) # test exploitability of mixture every n intervals
        self.exploit_steps              = kwargs.get('exploit_steps', 10000) # Steps for testing exploitabilty
        self.exploit_trials             = kwargs.get('exploit_trials', 50) # N eval steps for RBBR in exploit step
        self.exploit_n_rbbrs            = kwargs.get('exploit_n_rbbrs', 5) # N different RBBRs combputed 

        # Model training params:
        normalise                       = kwargs.get('normalise', True) # Normalise observation vector.
        self.num_parallel_envs          = kwargs.get('num_parallel_envs', 7) # Used determine envs in VecEnv
        self.model_type                 = kwargs.get('model_type', 'PPO') # 'PPO' instead of 'PPO2' since we are using torch version
        self.policy                     = kwargs.get('policy', 'MlpPolicy') # Feature extractors
        self.parallel                   = kwargs.get('parallel', True) # Keep it true while working with PPO        
        self.initial_pelicans           = kwargs.get('initial_pelicans', []) # Specify paths to existing agents if available.
        self.initial_panthers           = kwargs.get('initial_panthers', []) # '' ''
        sparse                          = kwargs.get('sparse', False) # Set to true for sparse rewards.

        # Game specific:
        self.max_illegal_moves_per_turn = kwargs.get('max_illegal_moves_per_turn', 2)
                
        # Video Args:
        self.video_flag                 = kwargs.get('video_flag', True) # whether to periodically create videos or not
        self.video_steps                = kwargs.get('video_steps', 100) # N steps used to create videos        
        self.basewidth                  = kwargs.get('basewidth', 1024) # Increase/decrease for high/low resolution videos.
        self.fps                        = kwargs.get('fps', 2) # Videos with lower values are easier to interpret.

        # Path to experiment folder
        exp_name = self.basicdate
        self.exp_path = os.path.join(basepath, exp_name)
        logger.info(self.exp_path)

        # Models are saved to:
        self.pelicans_tmp_exp_path = os.path.join(self.exp_path, 'pelicans_tmp')
        os.makedirs(self.pelicans_tmp_exp_path, exist_ok = True)
        self.panthers_tmp_exp_path = os.path.join(self.exp_path, 'panthers_tmp')
        os.makedirs(self.panthers_tmp_exp_path, exist_ok = True)
        # Logs are saved to:
        self.pnm_logs_exp_path = '/data/pnm_logs/test_' + self.basicdate
        os.makedirs(self.pnm_logs_exp_path, exist_ok = True)

        # Initialise sets
        self.pelicans = []
        self.panthers = []

        # Initial models set to None
        self.panther_model = None
        self.pelican_model = None

        self.pnm_iteration = 0
        #self.pelican_training_steps = 0
        #self.panther_training_steps = 0

        # Initialize the payoffs
        self.payoffs = np.zeros((1, 1))

        # Creating pelican env
        self.pelican_env = helper.get_envs('pelican',
                                           config_file_path,
                                           num_envs = self.num_parallel_envs,
                                           random_panther_start_position = True,
                                           random_pelican_start_position = True,
                                           max_illegal_moves_per_turn = self.max_illegal_moves_per_turn,
                                           sparse = sparse,
                                           vecenv = self.parallel,
                                           normalise = normalise,
                                           ui_on=False )

        # Creating panther env
        self.panther_env = helper.get_envs('panther',
                                           config_file_path,
                                           num_envs = self.num_parallel_envs,
                                           random_panther_start_position = True,
                                           random_pelican_start_position = True,
                                           max_illegal_moves_per_turn = self.max_illegal_moves_per_turn,
                                           sparse = sparse,
                                           vecenv = self.parallel,
                                           normalise = normalise,
                                           ui_on=False )


    def compute_initial_payoffs(self):
        # If I appended multiple entries all together
        if len(self.initial_pelicans) > 0:
            self.pelicans = self.pelicans[0]
        if len(self.initial_panthers) > 0:
            self.panthers = self.panthers[0]
        # If it is the first iteration and we are starting with initial models we need to build the corresponding payoff
        # Left out the last one for each (added in the normal cycle flow)
        # As we may start with a different number of agents per set, we need to deal with this
        for j, (pelican, panther) in enumerate(itertools.zip_longest(self.pelicans[:-1], self.panthers[:-1])):
            if pelican is not None:
                path = glob.glob(pelican + "/*.zip")[0]
                self.pelican_model = helper.loadAgent(path, self.model_type)
            else:
                self.pelican_model = None
            if panther is not None:
                path = glob.glob(panther + "/*.zip")[0]
                self.panther_model = helper.loadAgent(path, self.model_type)
            else:
                self.panther_model = None
            self.compute_payoff_matrix(self.pelicans[:min(j + 1, len(self.pelicans))],
                                       self.panthers[:min(j + 1, len(self.panthers))])

              
    def compute_payoff_matrix(self, pelicans, panthers):
        """
        - Pelican strategies are rows; panthers are columns
        - Payoffs are all to the pelican
        """

        # Resizing the payoff matrix for new strategies
        self.payoffs = np.pad(self.payoffs,
                             [(0, len(pelicans) - self.payoffs.shape[0]),
                             (0, len(panthers) - self.payoffs.shape[1])],
                             mode = 'constant')

        # Adding payoff for the last row strategy
        if self.pelican_model is not None:
            for i, opponent in enumerate(panthers):
                self.pelican_env.env_method('set_panther_using_path', opponent)
                victory_prop, avg_reward = helper.check_victory(self.pelican_model,
                                                                self.pelican_env,
                                                                trials = self.payoff_matrix_trials)
                self.payoffs[-1, i] = victory_prop

        # Adding payoff for the last column strategy
        if self.panther_model is not None:
            for i, opponent in enumerate(pelicans):
                self.panther_env.env_method('set_pelican_using_path', opponent)
                victory_prop, avg_reward = helper.check_victory(self.panther_model,
                                                                self.panther_env,
                                                                trials = self.payoff_matrix_trials)
                self.payoffs[i, -1] = 1 - victory_prop # do in terms of pelican

    def iter_train_against_mixture(self,
                                   n_rbbrs, # Number of resource bounded best responses
                                   exp_path,
                                   driving_agent, # agent that we train
                                   env, # Can either be a single env or subvecproc
                                   protagonist_filepaths, # Filepaths to existing models
                                   protagonist_mixture, # mixture for bootstrapping
                                   opponent_policy_fpaths, # policies of opponent of driving agent
                                   opponent_mixture): 
        
        win_percentages = []
        filepaths = []
        for i in range(n_rbbrs):
        
            model = self.bootstrap(protagonist_filepaths, env, protagonist_mixture)
            
            filepaths.append(self.train_agent_against_mixture(  driving_agent,
                                                                exp_path,
                                                                model,
                                                                env,
                                                                opponent_policy_fpaths,
                                                                opponent_mixture,
                                                                self.exploit_steps,
                                                                filepath_addon='_exploitability_model_%d' % i))
        
            win_percentages.append(self.eval_agent_against_mixture( driving_agent,
                                                                    exp_path,
                                                                    model,
                                                                    env,
                                                                    opponent_policy_fpaths,
                                                                    opponent_mixture,
                                                                    self.exploit_trials))

        return filepaths, win_percentages
            

    def bootstrap(self, model_paths, env, mixture):
        if np.random.rand(1) < self.retraining_prob:
            path = np.random.choice(model_paths, 1, p = mixture)[0]
            path = glob.glob(path + "/*.zip")[0]
            return helper.loadAgent(path, self.model_type)
        else:
            return helper.make_new_model(self.model_type,
                                         self.policy,
                                         env,
                                         n_steps=self.training_steps)
                       
    def eval_agent_against_mixture(self,
                                    exp_path,
                                    driving_agent, # agent that we train
                                    model,
                                    env, # Can either be a single env or subvecproc
                                    opponent_policy_fpaths, # policies of opponent of driving agent
                                    opponent_mixture, # mixture of opponent of driving agent
                                    n_eps): # number of eval eps

        ################################################################
        # Heuristic to compute number of opponents to sample as mixture
        ################################################################
        # Min positive probability
        min_prob = min([pr for pr in opponent_mixture if pr > 0])
        target_n_opponents = self.num_parallel_envs * int(1.0 / min_prob)
        n_opponents = min(target_n_opponents, self.max_n_opponents_to_sample)

        if self.parallel:
            # Ensure that n_opponents is a multiple of
            n_opponents = self.num_parallel_envs * round(n_opponents / self.num_parallel_envs)

        logger.info("=============================================")
        logger.info("Sampling %d opponents" % n_opponents)
        logger.info("=============================================")

        # Sample n_opponents
        opponents = np.random.choice(opponent_policy_fpaths,
                                     size = n_opponents,
                                     p = opponent_mixture)

        logger.info("=============================================")
        logger.info("Opponents has %d elements" % len(opponents))
        logger.info("=============================================")

        victories = []
        avg_rewards = []
        # If we use parallel envs, we run all the training against different sampled opponents in parallel
        if self.parallel:
            # Method to load new opponents via filepath
            setter = 'set_panther_using_path' if driving_agent == 'pelican' else 'set_pelican_using_path'
            for i, opponent in enumerate(opponents):
                # Stick this in the right slot, looping back after self.num_parallel_envs
                env.env_method(setter, opponent, indices = [i % self.num_parallel_envs])
                # When we have filled all self.num_parallel_envs, then train
                if i > 0 and (i + 1) % self.num_parallel_envs == 0:
                    logger.info("Beginning parallel eval for {} steps".format(self.training_steps))
                    model.set_env(env) 

                    victory_prop, avg_reward = helper.check_victory(model, env, trials = n_eps)

                    victories.append(victory_prop)
                    avg_rewards.append(avg_reward)
        
        # Otherwise we sample different opponents and we train against each of them separately
        else:
            for opponent in opponents:
                if driving_agent == 'pelican':
                    env.set_panther_using_path(opponent)
                else:
                    env.set_pelican_using_path(opponent)
                logger.info("Beginning sequential eval for {} steps".format(self.training_steps))
                model.set_env(env)
                victory_prop, avg_reward = helper.check_victory(model, env, trials = n_eps)
                victories.append(victory_prop)
                avg_rewards.append(avg_reward)

        return np.mean(victories)#, np.mean(avg_rewards)
            
                
    def train_agent_against_mixture(self,
                                    driving_agent, # agent that we train
                                    exp_path,
                                    model,
                                    env, # Can either be a single env or subvecproc
                                    opponent_policy_fpaths, # policies of opponent of driving agent
                                    opponent_mixture,
                                    training_steps,
                                    filepath_addon=''): # mixture of opponent of driving agent
                                    
        ################################################################
        # Heuristic to compute number of opponents to sample as mixture
        ################################################################
        # Min positive probability
        min_prob = min([pr for pr in opponent_mixture if pr > 0])
        target_n_opponents = self.num_parallel_envs * int(1.0 / min_prob)
        n_opponents = min(target_n_opponents, self.max_n_opponents_to_sample)

        if self.parallel:
            # Ensure that n_opponents is a multiple of
            n_opponents = self.num_parallel_envs * round(n_opponents / self.num_parallel_envs)

        logger.info("=============================================")
        logger.info("Sampling %d opponents" % n_opponents)
        logger.info("=============================================")

        # Sample n_opponents
        opponents = np.random.choice(opponent_policy_fpaths,
                                     size = n_opponents,
                                     p = opponent_mixture)

        logger.info("=============================================")
        logger.info("Opponents has %d elements" % len(opponents))
        logger.info("=============================================")

        # If we use parallel envs, we run all the training against different sampled opponents in parallel
        if self.parallel:
            # Method to load new opponents via filepath
            setter = 'set_panther_using_path' if driving_agent == 'pelican' else 'set_pelican_using_path'
            for i, opponent in enumerate(opponents):
                # Stick this in the right slot, looping back after self.num_parallel_envs
                env.env_method(setter, opponent, indices = [i % self.num_parallel_envs])
                # When we have filled all self.num_parallel_envs, then train
                if i > 0 and (i + 1) % self.num_parallel_envs == 0:
                    logger.info("Beginning parallel training for {} steps".format(self.training_steps))
                    model.set_env(env)
                    model.learn(training_steps)

        # Otherwise we sample different opponents and we train against each of them separately
        else:
            for opponent in opponents:
                if driving_agent == 'pelican':
                    env.set_panther_using_path(opponent)
                else:
                    env.set_pelican_using_path(opponent)
                logger.info("Beginning sequential training for {} steps".format(self.training_steps))
                model.set_env(env)
                model.learn(self.training_steps)

        # Save agent
        logger.info('Finished train agent')
        savepath = self.basicdate + '_pnm_iteration_'+ str(self.pnm_iteration) + filepath_addon 
        agent_filepath, _, _= helper.save_model_with_env_settings(exp_path, model, self.model_type, env, savepath)
        agent_filepath = os.path.dirname(agent_filepath)
        return agent_filepath

    def train_agent(self,
                    exp_path, # Path for saving the agent
                    model,
                    env):     # Can be either single env or vec env
                     
        logger.info("Beginning individual training for {} steps".format(self.training_steps))
        model.set_env(env)
        model.learn(self.training_steps)

        logger.info('Finished train agent')
        savepath = self.basicdate + '_pnm_iteration_' + str(self.pnm_iteration)
        agent_filepath ,_, _= helper.save_model_with_env_settings(exp_path, model, self.model_type, env, savepath)
        agent_filepath = os.path.dirname(agent_filepath)
        return agent_filepath


    def initialAgents(self):
        # If no initial pelican agent is given, we train one from fresh
        if len(self.initial_pelicans) == 0:
            # Train initial pelican vs default panther
            self.pelican_model = helper.make_new_model(self.model_type,
                                                       self.policy,
                                                       self.pelican_env,
                                                       n_steps=self.training_steps)
            logger.info('Training initial pelican')
            pelican_agent_filepath = self.train_agent(self.pelicans_tmp_exp_path,
                                                      self.pelican_model,
                                                      self.pelican_env)
        else:
            logger.info('Initial set of %d pelicans found' % (len(self.initial_pelicans)))
            pelican_agent_filepath = self.initial_pelicans


        # If no initial panther agent is given, we train one from fresh
        if len(self.initial_panthers) == 0:
            # Train initial panther agent vs default pelican
            self.panther_model = helper.make_new_model(self.model_type,
                                                       self.policy,
                                                       self.panther_env,
                                                       n_steps=self.training_steps)
            logger.info('Training initial panther')
            panther_agent_filepath  = self.train_agent(self.panthers_tmp_exp_path,
                                                       self.panther_model,
                                                       self.panther_env)
        else:
            logger.info('Initial set of %d panthers found' % (len(self.initial_panthers)))
            panther_agent_filepath = self.initial_panthers

        return panther_agent_filepath, pelican_agent_filepath

    def run_pnm(self):

        panther_agent_filepath, pelican_agent_filepath = self.initialAgents()

        # Initialize old NE stuff for stopping criterion
        value_to_pelican = 0.
        mixture_pelicans = np.array([1.])
        mixture_panthers = np.array([1.])

        # Create DataFrames for plotting purposes
        df_cols = ["NE_Payoff", "Pelican_BR_Payoff", "Panther_BR_Payoff", "Pelican_supp_size", "Panther_supp_size"]
        df = pd.DataFrame(columns = df_cols)
        # second df for period rigorous exploitability checks
        exploit_df_cols = ["iter",  "NE_Payoff", "Pelican_BR_Payoffs", "Panther_BR_Payoffs"]
        exploit_df = pd.DataFrame(columns = exploit_df_cols)

        # Train best responses until Nash equilibrium is found or max_iterations are reached
        logger.info('Parallel Nash Memory (PNM)')
        for self.pnm_iteration in range(self.max_pnm_iterations):
            start = time.time()

            logger.info("*********************************************************")
            logger.info('PNM iteration ' + str(self.pnm_iteration + 1) + ' of ' + str(self.max_pnm_iterations))
            logger.info("*********************************************************")

            self.pelicans.append(pelican_agent_filepath)
            self.panthers.append(panther_agent_filepath)

            if self.pnm_iteration == 0:
                self.compute_initial_payoffs()

            # Computing the payoff matrices and solving the corresponding LPs
            # Only compute for pelican in the sparse env, that of panther is the negative traspose (game is zero-sum)
            logger.info('Computing payoffs and mixtures')
            self.compute_payoff_matrix(self.pelicans, self.panthers)
            logger.info("=================================================")
            logger.info("New matrix game:")
            logger.info("As numpy array:")
            logger.info('\n' + str(self.payoffs))
            logger.info("As dataframe:")
            tmp_df = pd.DataFrame(self.payoffs).rename_axis('Pelican', axis = 0).rename_axis('Panther', axis = 1)
            logger.info('\n' + str(tmp_df))

            # save payoff matrix
            np.save('%s/payoffs_%d.npy' % (self.pnm_logs_exp_path, self.pnm_iteration), self.payoffs)

            def get_support_size(mixture):
                # return size of the support of mixed strategy mixture
                return sum([1 if m > 0 else 0 for m in mixture])

            # Check if we found a stable NE, in that case we are done (and fitting DF)
            if self.pnm_iteration > 0:
                # Both BR payoffs (from against last time's NE) in terms of pelican payoff
                br_value_pelican = np.dot(mixture_panthers, self.payoffs[-1, :-1])
                br_value_panther = np.dot(mixture_pelicans, self.payoffs[:-1, -1])

                ssize_pelican = get_support_size(mixture_pelicans)
                ssize_panther = get_support_size(mixture_panthers)

                logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                logger.info("\n\
                             Pelican BR payoff: %.3f,\n\
                             Value of Game: %.3f,\n\
                             Panther BR payoff: %.3f,\n\
                             Pelican Supp Size: %d,\n\
                             Panther Supp Size: %d,\n" % (
                                                          br_value_pelican,
                                                          value_to_pelican,
                                                          br_value_panther,
                                                          ssize_pelican,
                                                          ssize_panther
                                                          ))
                logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                values = dict(zip(df_cols, [value_to_pelican, br_value_pelican,
                                                              br_value_panther,
                                                              ssize_pelican,
                                                              ssize_panther]))
                df = df.append(values, ignore_index = True)

                # Write to csv file
                df_path =  os.path.join(self.exp_path, 'values_iter_%02d.csv' % self.pnm_iteration)
                df.to_csv(df_path, index = False)
                helper.get_fig(df)
                fig_path = os.path.join(self.exp_path, 'values_iter_%02d.pdf' % self.pnm_iteration)
                plt.savefig(fig_path)
                print("==========================================")
                print("WRITTEN VALUES DF TO CSV: %s" % df_path)
                print("==========================================")

                # here value_to_pelican is from the last time the subgame was solved
                if abs(br_value_pelican - value_to_pelican) < self.stopping_eps and\
                   abs(br_value_panther - value_to_pelican) < self.stopping_eps:

                    print('Stable Nash Equilibrium found')
                    break

            logger.info("SOLVING NEW GAME:")
            # solve game for pelican
            (mixture_pelicans, value_to_pelican) = lp_solve.solve_zero_sum_game(self.payoffs)
            # with np.printoptions(precision=3):
            logger.info(mixture_pelicans)
            mixture_pelicans /= np.sum(mixture_pelicans)
            # with np.printoptions(precision=3):
            logger.info("After normalisation:")
            logger.info(mixture_pelicans)
            np.save('%s/mixture_pelicans_%d.npy' % (self.pnm_logs_exp_path, self.pnm_iteration), mixture_pelicans)

            # solve game for panther
            (mixture_panthers, value_panthers) = lp_solve.solve_zero_sum_game(-self.payoffs.transpose())
            # with np.printoptions(precision=3):
            logger.info(mixture_panthers)
            mixture_panthers /= np.sum(mixture_panthers)
            # with np.printoptions(precision=3):
            logger.info("After normalisation:")
            logger.info(mixture_panthers)
            np.save('%s/mixture_panthers_%d.npy' % (self.pnm_logs_exp_path, self.pnm_iteration), mixture_panthers)

            # end of logging matrix game and solution
            logger.info("=================================================")

            # Train from skratch or retrain an existing model for pelican
            logger.info('Training pelican')
            
            self.pelican_model = self.bootstrap(self.pelicans, self.pelican_env, mixture_pelicans)
                
            pelican_agent_filepath = self.train_agent_against_mixture('pelican',
                                                                      self.pelicans_tmp_exp_path,
                                                                      self.pelican_model,
                                                                      self.pelican_env,
                                                                      self.panthers,
                                                                      mixture_panthers,
                                                                      self.training_steps)

            
            
            
            # Train from scratch or retrain an existing model for panther
            logger.info('Training panther')
            
            self.panther_model = self.bootstrap(self.panthers, self.panther_env, mixture_panthers)
            
            panther_agent_filepath = self.train_agent_against_mixture('panther',
                                                                     self.panthers_tmp_exp_path,
                                                                     self.panther_model,
                                                                     self.panther_env,
                                                                     self.pelicans,
                                                                     mixture_pelicans,
                                                                     self.training_steps)

            logger.info("PNM iteration lasted: %d seconds" % (time.time() - start))

            if self.pnm_iteration  > 0 and self.pnm_iteration  % self.testing_interval == 0:
                # Find best pelican (protagonist) against panther (opponent) mixture
                candidate_pelican_rbbr_fpaths, candidate_pelican_rbbr_win_percentages = self.iter_train_against_mixture(
                                                self.exploit_n_rbbrs, # Number of resource bounded best responses
                                                self.pelicans_tmp_exp_path,
                                                self.pelican_model, # driving_agent, # agent that we train
                                                self.pelican_env, # env, # Can either be a single env or subvecproc
                                                self.pelicans, # Filepaths to existing models
                                                mixture_pelicans, # mixture for bootstrapping
                                                self.panthers, # opponent_policy_fpaths, # policies of opponent of driving agent
                                                mixture_panthers) # opponent_mixture)

                logger.info("################################################")
                logger.info('candidate_pelican_rbbr_win_percentages: %s' %  np.round(candidate_pelican_rbbr_win_percentages,2))
                logger.info("################################################")
                br_values_pelican = np.round(candidate_pelican_rbbr_win_percentages,2).tolist()

                candidate_panther_rbbr_fpaths, candidate_panther_rbbr_win_percentages = self.iter_train_against_mixture(
                                                self.exploit_n_rbbrs, # Number of resource bounded best responses
                                                self.panthers_tmp_exp_path,
                                                self.panther_model, # driving_agent, # agent that we train
                                                self.panther_env, # env, # Can either be a single env or subvecproc
                                                self.panthers, # Filepaths to existing models
                                                mixture_panthers, # mixture for bootstrapping
                                                self.pelicans, # opponent_policy_fpaths, # policies of opponent of driving agent
                                                mixture_pelicans) # opponent_mixture)

                logger.info("################################################")
                logger.info('candidate_panther_rbbr_win_percentages: %s' % np.round(candidate_panther_rbbr_win_percentages,2))
                logger.info("################################################")
                br_values_panther = [1-p for p in np.round(candidate_panther_rbbr_win_percentages,2)]

                values = dict(zip(exploit_df_cols, [self.pnm_iteration,
                                                    value_to_pelican, 
                                                    br_values_pelican,
                                                    br_values_panther]))
                exploit_df = exploit_df.append(values, ignore_index = True)

                # add medians
                exploit_df['pelican_median'] = exploit_df['Pelican_BR_Payoffs'].apply(np.median)
                exploit_df['panther_median'] = exploit_df['Panther_BR_Payoffs'].apply(np.median)

                # Write to csv file
                df_path =  os.path.join(self.exp_path, 'exploit_iter_%02d.csv' % self.pnm_iteration)

                tmp_df = exploit_df.set_index('iter')
                tmp_df.to_csv(df_path, index = True)

                helper.get_fig_with_exploit(df, tmp_df)
                fig_path = os.path.join(self.exp_path, 'values_with_exploit_iter_%02d.pdf' % self.pnm_iteration)
                plt.savefig(fig_path)
                print("==========================================")
                print("WRITTEN EXPLOIT DF TO CSV: %s" % df_path)
                print("==========================================")

                if self.video_flag:
                    # occasionally ouput useful things along the way
                    # Make videos
                    verbose = False
                    video_path =  os.path.join(self.exp_path, 'pelican_pnm_iter_%02d.mp4' % self.pnm_iteration)
                    basewidth,hsize = helper.make_video_VEC_ENV(self.pelican_model, 
                                                                self.pelican_env, 
                                                                video_path,
                                                                fps=self.fps,
                                                                basewidth=self.basewidth,
                                                                n_steps=self.video_steps,
                                                                verbose=verbose)
                                                                
                    video_path =  os.path.join(self.exp_path, 'panther_pnm_iter_%02d.mp4' % self.pnm_iteration)
                    basewidth,hsize = helper.make_video_VEC_ENV(self.panther_model, 
                                                                self.panther_env, 
                                                                video_path,
                                                                fps=self.fps,
                                                                basewidth=self.basewidth,
                                                                n_steps=self.video_steps,
                                                                verbose=verbose)


        # Saving final mixture and corresponding agents
        logger.info("################################################")
        logger.info("Saving final pelican mixtures and agents:")
        support_pelicans = np.nonzero(mixture_pelicans)[0]
        mixture_pelicans = mixture_pelicans[support_pelicans]
        np.save(self.exp_path + '/final_mixture_pelicans.npy', mixture_pelicans)
        logger.info("Final pelican mixture saved to: %s" % self.exp_path + '/final_mixture_pelicans.npy')
        print("mixture:")
        print(mixture_pelicans)
        for i, idx in enumerate(mixture_pelicans):
            self.pelican_model = helper.loadAgent(glob.glob(self.pelicans[i]+ "/*.zip")[0], self.model_type)
            self.pelican_model.set_env(self.pelican_env) 
            agent_filepath ,_, _= helper.save_model_with_env_settings(self.pelicans_tmp_exp_path,
                                                                      self.pelican_model,
                                                                      self.model_type,
                                                                      self.pelican_env,
                                                                      self.basicdate + "_ps_" + str(i))
            logger.info("Saving  pelican %d to %s" % (i, agent_filepath))
        support_panthers = np.nonzero(mixture_panthers)[0]
        mixture_panthers = mixture_panthers[support_panthers]
        np.save(self.exp_path + '/final_mixture_panthers.npy', mixture_panthers)
        logger.info("Final panther mixture saved to: %s" % self.exp_path + '/final_mixture_panthers.npy')
        for i, idx in enumerate(mixture_panthers):
            self.panther_model = helper.loadAgent(glob.glob(self.panthers[i]+ "/*.zip")[0], self.model_type)
            self.panther_model.set_env(self.panther_env) 
            agent_filepath ,_, _= helper.save_model_with_env_settings(self.panthers_tmp_exp_path,
                                                                      self.panther_model,
                                                                      self.model_type,
                                                                      self.panther_env,
                                                                      self.basicdate + "_ps_" + str(i))

            logger.info("Saving  panther %d to %s" % (i, agent_filepath))
if __name__ == '__main__':

    #examples_dir = '/data/examples'
    # Initial sets of opponents is automatically loaded from dir
    #pelicans_start_opponents = [p.path for p in os.scandir(examples_dir + '/pelicans') if p.is_dir()]
    #panthers_start_opponents = [p.path for p in os.scandir(examples_dir + '/panthers') if p.is_dir()]
    #pelicans_start_opponents = ["data/examples/pelicans_tmp/PPO_20210220_152444_steps_" + str(i * 100) + "_pelican" for i in range(1, 7)]
    #panthers_start_opponents = ["data/examples/panthers_tmp/PPO_20210220_152444_steps_" + str(i * 100) + "_panther" for i in range(1, 7)]
    pelicans_start_opponents = []
    panthers_start_opponents = []

    pnm = PNM(initial_pelicans = pelicans_start_opponents,
              initial_panthers = panthers_start_opponents)

    pnm.run_pnm()
