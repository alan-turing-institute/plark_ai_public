import os
from random import seed
from random import randint

from plark_game.classes.pantherAgent import Panther_Agent
from plark_game import game_helper
import jsonpickle

# TODO: Implement this to check all columns for a free path
# Retrieve Information about map - specifically the location of sonobouys
# If panther_col +- sb_range columns contains no sb then actionValue = actions[0]
#global_sb, a list containing the locations of all sonobuoys within the map
#global_torps, a list containing the locations of all the torpedos within the map.


class PantherAgentRandOpportunistic(Panther_Agent):

    def __init__(self):
        pass

    def getAction(self, state):
        map_width = state['map_width']
        panther_col = state['panther_col']

        #grid = jsonpickle.decode(state['mapFile'])  # get map in order to get sb_locations data
        # Get cells containing sb
        #deployed_sb_cells = [cell for row in grid for cell in row if "SONOBUOY" in cell['objects']]
        #deployed_sb_cols = [cell["coordinate"][0] for cell in deployed_sb_cells]
        deployed_sb_cols = [buoy.col for buoy in state['deployed_sonobuoys'] if buoy.state == 'HOT']
        #print(deployed_sb_cols)

        # sb_range = list(filter(lambda item: (item.type == 'SONOBUOY'), state['pelican_loadout']))[0].range

        # Removed 0 here as the panther will check if it can go straight ahead or not already
        actions = []
        # If Panther is not in sb+-1 col go straight up - can add here +- sb_range

        if panther_col not in (deployed_sb_cols +
                               [item-1 for item in deployed_sb_cols] +
                               [item+1 for item in deployed_sb_cols]):
            return self.action_lookup(0)
        if panther_col != 0:  # If it is not on the left add move left action
            actions.append(5)
        if panther_col < map_width:  # If it is not on the right add move right action
            actions.append(1)

        seed()
        actionValue = actions[randint(0, len(actions) - 1)]

        return self.action_lookup(actionValue)

    def getTournamentAction(self, obs, obs_norm, d_params, d_params_norm, state):
        map_width = state['map_width']
        panther_col = state['panther_col']

        #grid = jsonpickle.decode(state['mapFile'])  # get map in order to get sb_locations data
        # Get cells containing sb
        #deployed_sb_cells = [cell for row in grid for cell in row if "SONOBUOY" in cell['objects']]
        #deployed_sb_cols = [cell["coordinate"][0] for cell in deployed_sb_cells]
        deployed_sb_cols = [buoy.col for buoy in state['deployed_sonobuoys'] if buoy.state == 'HOT']
        # sb_range = list(filter(lambda item: (item.type == 'SONOBUOY'), state['pelican_loadout']))[0].range

        # Removed 0 here as the panther will check if it can go straight ahead or not already
        actions = []
        # If Panther is not in sb+-1 col go straight up - can add here +- sb_range
        if panther_col not in (deployed_sb_cols +
                               [item-1 for item in deployed_sb_cols] +
                               [item+1 for item in deployed_sb_cols]):
            return self.action_lookup(0)
        if panther_col != 0:  # If it is not on the left add move left action
            actions.append(5)
        if panther_col < map_width:  # If it is not on the right add move right action
            actions.append(1)

        seed()
        actionValue = actions[randint(0, len(actions) - 1)]

        return self.action_lookup(actionValue)

# ACTION_LOOKUP = {
#     0 : '1',  # Up
#     1 : '2',  # Up right
#     2 : '3',  # Down right
#     3 : '4',  # Down
#     4 : '5',  # Down left
#     5 : '6',  # Up left
# 	6 : 'end'
# }
