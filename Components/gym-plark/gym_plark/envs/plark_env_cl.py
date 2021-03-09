import os
import subprocess
import time
import signal
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

import sys
from plark_game import classes

from gym_plark.envs.plark_env import PlarkEnv

import logging
logger = logging.getLogger(__name__)

from plark_game.classes.environment import Environment
from plark_game.classes.observation import Observation
import random
import math

class PlarkEnvCurriculum(PlarkEnv):

        def __init__(self,config_file_path=None, **kwargs):
                super(PlarkEnvCurriculum, self).__init__(config_file_path, **kwargs)
                self.min_distance           = kwargs.get('min_distance', 6)            # Minimum init distance between panther and pelican
                self.initial_distance       = kwargs.get('initial_distance', 6)        # First thing to try out with CL, keeping low initial distance
                self.outcome_eval_threshold = kwargs.get('outcome_eval_threshold', 10) # Call progression function every n steps
                self.increase_thresh        = kwargs.get('increase_thresh', 0.8)       # Threshold at which we make things more challenging
                self.decrease_thresh        = kwargs.get('increase_thresh', 0.2)       # Threshold at which we make things easier
                self.search_range           = kwargs.get('search_range', 4)            # Torpedo search range
                self.speed                  = kwargs.get('speed', [4])                   # Torpedo search range

                self.outcome_tracker        = []                                       # To keep track of outcomes

        def progression(self):
                wins = sum(self.outcome_tracker)/float(self.outcome_eval_threshold)
                if wins >= self.increase_thresh:
                    self.initial_distance+=1
                elif wins <= self.decrease_thresh:
                    self.initial_distance = max(self.min_distance, self.initial_distance-1)
                self.outcome_tracker = [] # Reset
                logger.info("New Init Distance: %d, Wins: %f " % (self.initial_distance, wins))

        def step(self, action):
                action = self.ACTION_LOOKUP[action]
                if self.verbose:
                        logger.info('Action:'+action)
                gameState, uioutput = self.env.activeGames[len(
                        self.env.activeGames)-1].game_step(action)
                self.status = gameState
                self.uioutput = uioutput

                ob = self._observation()
                #print(ob)
                reward = 0
                done = False
                _info = {}

                _info['status'] = self.status

                #  PELICANWIN = Pelican has won
                #  ESCAPE     = Panther has won
                #  BINGO      = Panther has won, Pelican has reached it's turn limit and run out of fuel
                #  WINCHESTER = Panther has won, All torpedoes dropped and stopped running. Panther can't be stopped
                if self.status in ["ESCAPE", "BINGO", "WINCHESTER", "PELICANWIN"]:
                        #print("Ending status:", self.status)
                        done = True
                        if self.verbose:
                                logger.info("GAME STATE IS " + self.status)
                        if self.status in ["ESCAPE", "BINGO", "WINCHESTER"]:
                                if self.driving_agent == 'pelican':
                                        reward = -1
                                        _info['result'] = "LOSE"
                                        self.outcome_tracker.append(0)
                                        logger.info("Len outcome tracker: %d, win percentage %f" % (len(self.outcome_tracker), sum(self.outcome_tracker)/float(len(self.outcome_tracker))))
                                elif self.driving_agent == 'panther':
                                        reward = 1
                                        _info['result'] = "WIN"
                                else:
                                        raise ValueError('driving_agent not set correctly')
                        if self.status == "PELICANWIN":
                                if self.driving_agent == 'pelican':
                                        reward = 1
                                        _info['result'] = "WIN"
                                        self.outcome_tracker.append(1)
                                        logger.info("Len outcome tracker: %d, win percentage %f" % (len(self.outcome_tracker), sum(self.outcome_tracker)/float(len(self.outcome_tracker))))
                                elif self.driving_agent == 'panther':
                                        reward = -1
                                        _info['result'] = "LOSE"
                                else:
                                        raise ValueError('driving_agent not set correctly')
                if len(self.outcome_tracker) == self.outcome_eval_threshold:
                    self.progression()
                return ob, reward, done, _info

        def reset(self):

                #If a game doesn't already exist
                if len(self.env.activeGames) == 0:
                        if self.config_file_path:
                                logger.info('config filepath: ' +str(self.config_file_path))
                                self.env.createNewGame(config_file_path=self.config_file_path, **self.kwargs)
                        else:
                                self.env.createNewGame(**self.kwargs)   

                # Potentially modify torped settings. Fixed for now.
                # Choosing lower values so that a shot in the dark will be unlikely to hit the target
                self.env.activeGames[len(self.env.activeGames)-1].torpedo_parameters['speed'] = self.speed
                self.env.activeGames[len(self.env.activeGames)-1].torpedo_parameters['turn_limit'] = 1
                self.env.activeGames[len(self.env.activeGames)-1].torpedo_parameters['search_range'] = self.search_range

                #Force a reset with CL distances...
                #On reset randomly places panther in a different location.
                map_width = self.env.activeGames[len(self.env.activeGames)-1].map_width
                map_height = self.env.activeGames[len(self.env.activeGames)-1].map_height
                #Taken out of domain_parameters.py (Simon's bounds)
                panther_start_col_lb = int(math.floor(0.33 * map_width))
                panther_start_col_ub = int(math.floor(0.66 * map_width))
                panther_start_row_lb = int(math.floor(0.8 * map_height))
                panther_start_row_ub = map_height-1

                panther_start_col = random.choice(range(panther_start_col_lb,
                                                        panther_start_col_ub+1))
                panther_start_row = random.choice(range(panther_start_row_lb,
                                                        panther_start_row_ub+1)) 
                self.env.activeGames[len(self.env.activeGames)-1].panther_start_col = panther_start_col
                self.env.activeGames[len(self.env.activeGames)-1].panther_start_row = panther_start_row

                pelican_start_col_lb = max(panther_start_col - self.initial_distance, 0)
                pelican_start_col_ub = min(panther_start_col + self.initial_distance, map_width-1)
                pelican_start_row_lb = max(panther_start_row - self.initial_distance, 0)
                pelican_start_row_ub = min(panther_start_row, map_height-1) # Does not make sense for the pelican to start below the panther

                pelican_start_col = random.choice(range(pelican_start_col_lb,
                                                        pelican_start_col_ub+1))
                pelican_start_row = random.choice(range(pelican_start_row_lb,
                                                        pelican_start_row_ub+1))
                self.env.activeGames[len(self.env.activeGames)-1].pelican_start_col = pelican_start_col
                self.env.activeGames[len(self.env.activeGames)-1].pelican_start_row = pelican_start_row

                self.env.activeGames[len(self.env.activeGames)-1].reset_game()

                self.game = self.env.activeGames[len(self.env.activeGames)-1]           

                #logger.info("Pelican start xy: %d %d" % (pelican_start_col, pelican_start_row))
                #logger.info("Panther start xy: %d %d" % (panther_start_col, panther_start_row))

                if self.driving_agent == 'pelican':
                        self.render_width = self.game.pelican_parameters['render_width']
                        self.render_height = self.game.pelican_parameters['render_height']

                elif self.driving_agent == 'panther':
                        self.render_width = self.game.panther_parameters['render_width']
                        self.render_height = self.game.panther_parameters['render_height']

                return self._observation()

