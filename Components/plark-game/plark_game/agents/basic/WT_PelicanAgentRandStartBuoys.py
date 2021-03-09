from plark_game.classes.pelicanAgentFixedSBs import PelicanAgentFixedSBs
import jsonpickle
import logging

from random import seed
from random import randint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PelicanAgentRandStartBuoys(PelicanAgentFixedSBs):

    def __init__(self):
        super(PelicanAgentRandStartBuoys, self).__init__()

    def getAction(self, state):
        if len(self.sb_locations) == 0:
            self.sb_locations = self.generate_sb_locations(state)

        return super(PelicanAgentRandStartBuoys, self).getAction(state)

    def generate_sb_locations(self, state):
        grid = jsonpickle.decode(state['mapFile'])
        num_cols = len(grid)
        num_rows = len(grid[0])
        sb_locations = []
        no_of_sbs = len(list(filter(lambda item: (item.type == 'SONOBUOY'), state['pelican_loadout'])))
        sb_range = list(filter(lambda item: (item.type == 'SONOBUOY'), state['pelican_loadout']))[0].range


        # Start some distance down the grid
        seed()
        sb_locations.append({'col': sb_range, 'row': [randint(1, num_rows - sb_range - 2)]})

        for sb in range(no_of_sbs):
            new_sb = {
                'col': sb_locations[-1]['col'] + (sb_range * 2) + 2,
                'row': sb_locations[-1]['row']
            }
            if new_sb['col'] >= num_cols:
                new_sb = {
                    'col': sb_locations[0]['col'],
                    'row': sb_locations[-1]['row'] + (sb_range * 2) + 2
                }

            if new_sb['row'] >= num_rows:
                break

            sb_locations.append(new_sb)
        logger.info('sb_locations: ' + str(sb_locations))
        return sb_locations

