from plark_game.classes.pelicanAgent import Pelican_Agent

from random import seed
from random import randint

import jsonpickle
from plark_game import game_helper


class PelicanAgentFixedSBs(Pelican_Agent):
    """
    Simple Pelican agent which drops sonobuoys in fixed locations.
    If sonobuoys activate, drops torpedo nearby.
    """

    # Really, we should make sb_locations a constructor argument rather than subclassing,
    # but this way imposes minimal changes on any calling code
    def __init__(self):
        self.target_col = None
        self.target_row = None
        self.sb_locations = []

    def getAction(self, state):

        num_rows = state['map_width']
        num_cols = state['map_height']

        pelican_row = state['pelican_row']
        pelican_col = state['pelican_col']

        deployed_sbs = [{'state':buoy.state,
                         'col':buoy.col,
                         'row':buoy.row} for buoy in state['deployed_sonobuoys']]

        for sb in deployed_sbs:
            # active sb means panther nearby
            if sb["state"] == "HOT":
                # If pelican is outside sb range fly to sb
                if (
                    game_helper.distance(pelican_col, pelican_row, sb["col"], sb["row"])
                    > state["sb_range"]
                ):
                    path = game_helper.get_path(
                        pelican_col,
                        pelican_row,
                        sb["col"],
                        sb["row"],
                        num_cols,
                        num_rows,
                    )
                    return self.action_lookup(
                        self.cell_to_action(path[0], pelican_col, pelican_row)
                    )
                # If pelican is within sb range, drop torpedo if there isn't one nearby already
                else:

                    if (game_helper.searchRadius_nongrid(state,
                                                         pelican_col,
                                                         pelican_row,
                                                         1,
                                                         "TORPEDO") == []):
                        return self.action_lookup(7)  # torp
                    else:
                        return self.action_lookup(8)  # end

        for sb in self.sb_locations:
            # If there are still sonobuoys in payload to deploy
            if list(
                filter(lambda item: (item.type == "SONOBUOY"), state["pelican_loadout"])
            ):
                # If there isn't already a sonobuoy at this target location
                sb_already_there = False
                for ds in deployed_sbs:
                    if sb['col']==ds['col'] and sb['row']==ds['row']:
                        sb_already_there = True
                        break
                if not sb_already_there:
                    # If not already in target location, move towards it
                    if (pelican_col != sb["col"]) or (pelican_row != sb["row"]):
                        path = game_helper.get_path(
                            pelican_col,
                            pelican_row,
                            sb["col"],
                            sb["row"],
                            num_cols,
                            num_rows,
                        )
                        return self.action_lookup(
                            self.cell_to_action(path[0], pelican_col, pelican_row)
                        )
                    # If in target location, drop sb
                    else:
                        return self.action_lookup(6)  # drop_buoy

        # If we haven't already dropped a torp or sb, and don't have any sbs left to deploy, pick a random sb location and fly towards it
        if self.target_col == None:
            seed()
            random_sb = randint(0, len(self.sb_locations) - 1)
            self.target_col = self.sb_locations[random_sb]["col"]
            self.target_row = self.sb_locations[random_sb]["row"]
        else:
            if (pelican_col == self.target_col) and (pelican_row == self.target_row):
                self.target_col = None
                self.target_row = None
                return self.action_lookup(8)  # end

            path = game_helper.get_path(
                pelican_col,
                pelican_row,
                self.target_col,
                self.target_row,
                num_cols,
                num_rows,
            )

            return self.action_lookup(
                self.cell_to_action(path[0], pelican_col, pelican_row)
            )
