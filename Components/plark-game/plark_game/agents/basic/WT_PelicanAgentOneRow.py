from plark_game.classes.WT_PelicanAgentThreeTorpedo import PelicanAgentThreeTorpedo
import jsonpickle
import logging

from random import seed
from random import randint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Potentially include an extra variable here to define how far down the grid to place is, as a %


class PelicanAgentOneRow(PelicanAgentThreeTorpedo):

    def __init__(self):
        super(PelicanAgentOneRow, self).__init__()

    def getAction(self, state):
        if len(self.sb_locations) == 0:
            self.sb_locations = self.generate_sb_locations(state)

        return super(PelicanAgentOneRow, self).getAction(state)

    def generate_sb_locations(self, state):
        #grid = jsonpickle.decode(state['mapFile'])
        #num_cols = len(grid)
        #num_rows = len(grid[0])
        num_rows = state['map_width']
        num_cols = state['map_height']
        sb_locations = []
        no_of_sbs = len(list(filter(lambda item: (item.type == 'SONOBUOY'), state['pelican_loadout'])))
        sb_range = list(filter(lambda item: (item.type == 'SONOBUOY'), state['pelican_loadout']))[0].range


        # Start some distance down the grid
        # Behaviour needs to be included to drop a buoy/ torpedo above and below an active sb
        # Maybe by changing the fixed sb class and basing this class on it
        seed()
        #sb_locations.append({'col': sb_range, 'row': [randint(1+sb_range, num_rows - sb_range - 1)]})
        sb_locations.append({'col': sb_range, 'row': randint(1+sb_range, num_rows - sb_range - 1)})

        for sb in range(no_of_sbs):
            new_sb = {
                'col': sb_locations[-1]['col'] + (sb_range * 2) + 2,
                'row': sb_locations[-1]['row']
            }
            if new_sb['col'] >= num_cols:
                break

            if new_sb['row'] >= num_rows:
                break

            sb_locations.append(new_sb)
        logger.info('sb_locations: ' + str(sb_locations))
        return sb_locations

