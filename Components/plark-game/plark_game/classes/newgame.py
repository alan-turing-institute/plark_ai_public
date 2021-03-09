from .newgamebase import NewgameBase
from .pelican import Pelican
from .panther import Panther
from .torpedo import Torpedo
from .sonobuoy import Sonobuoy
from .map import Map
from .pil_ui import PIL_UI
from .move import Move
from .pelicanAgent import Pelican_Agent
from .pantherAgent import Panther_Agent
from .pantherAgent_load_agent import Panther_Agent_Load_Agent
from .pelicanAgent_load_agent import Pelican_Agent_Load_Agent
from .pantherAgent_set_agent import Panther_Agent_Set_Agent
from .pelicanAgent_set_agent import Pelican_Agent_Set_Agent
from .observation import Observation
from .explosion import Explosion
import numpy as np
import os
import json
import sys
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import torch
import gc
import glob

class Newgame(NewgameBase):

        # Python constructor to initialise the players within the gamespace.
        # These are subsequently used throughout the game.
        def __init__(self, game_config, **kwargs):
                # call the base class constructor
                super().__init__(game_config, **kwargs)
                self.trained_agents_filepath = '/data/models/'
                self.relative_basic_agents_filepath = '../agents/basic'
                self.import_agents()

                # Load agents
                relative_basic_agents_filepath = os.path.join(os.path.dirname(__file__), self.relative_basic_agents_filepath)
                relative_basic_agents_filepath = os.path.normpath(relative_basic_agents_filepath)

                if self.driving_agent == 'panther':
                        if not self.output_view_all:
                                self.gamePlayerTurn = "PANTHER"

                        self.pelicanAgent = load_agent(self.pelican_parameters['agent_filepath'], self.pelican_parameters['agent_name'],relative_basic_agents_filepath,self)

                else:
                        if not self.output_view_all:
                                self.gamePlayerTurn = "PELICAN"

                        self.pantherAgent = load_agent(self.panther_parameters['agent_filepath'], self.panther_parameters['agent_name'],relative_basic_agents_filepath,self)


                # Game state variables
                self.default_game_variables()

                # Create UI objects and render game. This must be the last thing in the __init__
                if self.driving_agent == 'pelican':
                        self.render_height = self.pelican_parameters['render_height']
                        self.render_width = self.pelican_parameters['render_width']
                else:
                        self.render_height = self.panther_parameters['render_height']
                        self.render_width = self.panther_parameters['render_width']

                self.reset_game()
                if self.ui_on:
                	self.render(self.render_width,self.render_height,self.gamePlayerTurn)


        def set_pelican(self, pelican):
            kwargs = {}
            kwargs['driving_agent'] = 'pelican'
            self.pelicanAgent =  Pelican_Agent_Set_Agent(pelican, Observation(self, **kwargs))

        def set_panther(self, panther):
            kwargs = {}
            kwargs['driving_agent'] = 'panther'
            self.pantherAgent =  Panther_Agent_Set_Agent(panther, Observation(self, **kwargs))

        def pelicanPhase(self):
                # This is what is called if the panther is the playable agent.
                self.pelicanMove = Move()
                while True:
                        pelican_action = self.pelicanAgent.getAction(self._state("PELICAN"))

                        self.perform_pelican_action(pelican_action)

                        if self.pelican_move_in_turn >= self.pelican_parameters['move_limit'] or pelican_action == 'end':
                                break

        def pantherPhase(self):
                # This is what is called if the pelican is the playable agent.
                self.pantherMove = Move()
                while True:
                        panther_action = self.pantherAgent.getAction(self._state("PANTHER"))

                        self.perform_panther_action(panther_action)
                        if self.gameState == 'ESCAPE' or self.panther_move_in_turn >= self.panther_parameters['move_limit'] or panther_action == 'end':
                                break

        def import_agents(self):
                #This loads the agents from a default relative_basic_agents_filepath path which is inside the pip module.
                relative_basic_agents_filepath = os.path.join(os.path.dirname(__file__), self.relative_basic_agents_filepath)
                relative_basic_agents_filepath = os.path.normpath(relative_basic_agents_filepath)

                for agent in os.listdir(relative_basic_agents_filepath):
                        if os.path.splitext(agent)[1] == ".py":
                                # look only in the modpath directory when importing
                                oldpath, sys.path[:] = sys.path[:], [relative_basic_agents_filepath]

                                try:
                                        #logger.info('Opening agent from:'+relative_basic_agents_filepath+'/'+str(agent))
                                        module = __import__(agent[:-3])

                                except ImportError as err:
                                        raise ValueError("Couldn't import", agent, ' - ', err )
                                        continue
                                finally:    # always restore the real path
                                        sys.path[:] = oldpath


        def load_pelican_using_path(self, file_path, observation=None, image_based=False, in_tournament=False, **kwargs):
            metadata_filepath = os.path.join(file_path, 'metadata.json')
            agent_filepath = glob.glob(file_path+"/*.zip")[0]
            with open(metadata_filepath) as f:
                metadata = json.load(f)
            #logger.info('Playing against:'+agent_filepath)
            kwargs = {}
            kwargs['driving_agent'] = metadata['agentplayer']
            kwargs['normalise'] = metadata['normalise']
            kwargs['domain_params_in_obs'] = metadata['domain_params_in_obs']
            if image_based == False:
                observation = Observation(self,**kwargs)
            self.pelicanAgent = Pelican_Agent_Load_Agent(
                agent_filepath,
                metadata['algorithm'],
                observation,
                image_based,
                in_tournament=in_tournament,
                **kwargs
            )


        def load_panther_using_path(self, file_path, observation=None, image_based=False, in_tournament=False, **kwargs):
            metadata_filepath = os.path.join(file_path, 'metadata.json')
            agent_filepath = glob.glob(file_path+"/*.zip")[0]
            with open(metadata_filepath) as f:
                metadata = json.load(f)
            #logger.info('Playing against:'+agent_filepath)
            kwargs = {}
            kwargs['driving_agent'] = metadata['agentplayer']
            kwargs['normalise'] = metadata['normalise']
            kwargs['domain_params_in_obs'] = metadata['domain_params_in_obs']
            if image_based == False:
                observation = Observation(self,**kwargs)
            self.pantherAgent = Panther_Agent_Load_Agent(
                agent_filepath,
                metadata['algorithm'],
                observation,
                image_based,
                in_tournament=in_tournament,
                **kwargs)


def load_agent(file_path, agent_name, basic_agents_filepath, game, in_tournament=False, **kwargs):
        if '.py' in file_path: # This is the case for an agent in a non standard location or selected throguh web ui.
                        # load python
                        file_path = os.path.join(basic_agents_filepath, file_path)
                        oldpath, sys.path[:] = sys.path[:], [basic_agents_filepath]
                        try:
                                module = __import__(file_path.split('/')[-1][:-3])
                        except ImportError as err:
                                raise ValueError("Couldn't import " + file_path, ' - ', err)
                        cls = getattr(module, agent_name)
                        #logger.info('Playing against:'+agent_name)
                        # always restore the real path
                        sys.path[:] = oldpath
                        return cls()
        else:
                files = os.listdir(file_path)
                for f in files:
                        if '.zip' not in f:
                                # ignore non agent files
                                pass

                        elif '.zip' in f:
                                # load model
                                metadata_filepath = os.path.join(file_path, 'metadata.json')
                                agent_filepath = os.path.join(file_path, f)

                                with open(metadata_filepath) as f:
                                        metadata = json.load(f)
                                #logger.info('Playing against:'+agent_filepath)

                                observation = None
                                image_based = True
                                if 'image_based' in metadata and metadata['image_based'] == False: # If the image_based flag is not present asume image based, if it is there and set to false the set to false.
                                        image_based = False

                                        if kwargs is None:
                                            kwargs = {}
                                        kwargs['driving_agent'] = metadata['agentplayer']
                                        kwargs['normalise'] = metadata['normalise']
                                        kwargs['domain_params_in_obs'] = metadata['domain_params_in_obs']
                                        if not in_tournament:
                                            observation = Observation(game,**kwargs)

                                print("Filepath of the agent being loaded is: " + agent_filepath)


                                if metadata['agentplayer'] == 'pelican':
                                        return Pelican_Agent_Load_Agent(
                                                agent_filepath,
                                                metadata['algorithm'],
                                                observation,
                                                image_based,
                                                in_tournament=in_tournament,
                                                **kwargs
                                        )
                                elif metadata['agentplayer'] == 'panther':
                                        return Panther_Agent_Load_Agent(
                                                agent_filepath,
                                                metadata['algorithm'],
                                                observation,
                                                image_based,
                                                in_tournament=in_tournament,
                                                **kwargs
                                        )

                        else:
                                raise ValueError('no agent found in ', file_path)
