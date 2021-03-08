import os
import sys
import json
import logging
import time
import pika
import uuid

from pika.adapters.utils.connection_workflow import (
    AMQPConnectorSocketConnectError,
)

from plark_game.classes.newgame import load_agent
from plark_game.classes.pantherAgent_load_agent import Panther_Agent_Load_Agent
from plark_game.classes.pelicanAgent_load_agent import Pelican_Agent_Load_Agent

from schema import deserialize_state

##########################################################
# set AGENTS_PATH to point to the directory containg your agents, relative
# to the base `plark_ai_public` directory (i.e. keep "/plark_ai_public" at
# the start of the path below - this is the directory as it will be seen
# in the docker image).
#
# This directory should have sub-directories 'panther' and/or 'pelican', and
# those should have a sub-directory containing your agent.
# Agent can be either a .zip file and metadata json file, or a .py file.


AGENTS_PATH = os.path.join(
    os.path.abspath(
        os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
        ),
    ),
    "data",
    "agents",
    "test"
)
##########################################################
# You shouldn't need to change BASIC_AGENTS_PATH, unless you
# alter the directory structure of this package.

BASIC_AGENTS_PATH = os.path.normpath(
    os.path.join(
        "/plark_ai_public",
        "Components",
        "plark-game",
        "plark_game",
        "agents",
        "basic",
    )
)



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_combatant(
    agent_path, agent_name, basic_agents_path, game=None, **kwargs
):
    """
    Loads agent as a combatant.
        agent_path:
        agent_name:
        basic_agents_path:
        game:
        kwargs:
    Returns:
        agent
    """

    if ".py" in agent_path:
        return load_agent(

            agent_path, agent_name, basic_agents_path, game, in_tournament=False,**kwargs
        )
    #Load an agent trained via an Evolutionary Algorithm - they use the NNAgent class
    elif "evo_models" in agent_path:

        from plark_game.agents.basic.panther_nn import PantherNN
        from plark_game.agents.basic.pelican_nn import PelicanNN

        if 'pelican' in agent_name:
            return PelicanNN(file_dir_name='pelican', game=game, in_tournament=True)
        else:
            return PantherNN(file_dir_name='panther', game=game, in_tournament=True)

    else:

        files = os.listdir(agent_path)

        for f in files:
            if ".zip" not in f:
                # ignore non agent files
                pass

            elif ".zip" in f:
                # load model
                metadata_filepath = os.path.join(agent_path, "metadata.json")
                agent_filepath = os.path.join(agent_path, f)

                with open(metadata_filepath) as f:
                    metadata = json.load(f)

                # NON-IMAGE-BASED
                if (
                    "image_based" in metadata
                    and metadata["image_based"] is False
                ):
                    return load_agent(
                        agent_path,
                        agent_name,
                        basic_agents_path,
                        game,
                        in_tournament=True,
                        **kwargs
                    )

                # IMAGE-BASED - NOT SUPPORTED FOR DSG TOURNAMENT
                observation = None
                image_based = True
                algorithm = metadata["algorithm"]
                print("algorithm: ", algorithm)

                if metadata["agentplayer"] == "pelican":
                    return Pelican_Agent_Load_Agent(
                        agent_filepath,
                        algorithm,
                        observation,
                        image_based,
                        in_tournament=True
                    )
                elif metadata["agentplayer"] == "panther":
                    return Panther_Agent_Load_Agent(
                        agent_filepath,
                        algorithm,
                        observation,
                        image_based,
                        in_tournament=True
                    )

    return None


class Combatant:
    def __init__(
        self, agent_path, agent_name, basic_agents_path, game=None, **kwargs
    ):
        self.agent = load_combatant(
            agent_path, agent_name, basic_agents_path, game=None, **kwargs
        )
        self.ready = False

    def get_action(self, ch, method, props, body):
        """
        Participants: you may want to modify this function, if you're not using
        a stable_baselines agent
        """
        message = json.loads(body)
        # 'state' is the state information that can be known by the agent
        # (i.e. the panther position is hidden from pelican agents, etc.)
        state = message["state"]
        # convert json objects back into e.g. Torpedo instances
        deserialized_state = deserialize_state(state)

        # 'obs' is a list of numbers, representing the state in the format that
        # is expected by stable_baselines
        obs = message["obs"]
        # 'obs_normalised' is the same as `obs`, but normalised such that values
        # lie between 0 and 1
        obs_normalised = message["obs_normalised"]
        # 'domain_parameters' is the parameters of the domain
        domain_parameters = message["domain_parameters"]
        # 'domain_parameters_normalised' is as above, but normalised
        domain_parameters_normalised = message["domain_parameters_normalised"]

        # ask the agent for the action, given this observation ( or state, ...)
        # below is an example using a stable_baselines agent, that just expects
        # the observation.
        # Modify this function call if you have a different
        # type of agent that expects different information.
        response = self.agent.getTournamentAction(
            obs,
            obs_normalised,
            domain_parameters,
            domain_parameters_normalised,
            deserialized_state
        )

        # send the action back to the battle
        ch.basic_publish(
            exchange="",
            routing_key=props.reply_to,
            properties=pika.BasicProperties(
                correlation_id=props.correlation_id
            ),
            body=str(response),
        )

        ch.basic_ack(delivery_tag=method.delivery_tag)

    def send_ready(self, agent_type, conn, channel, queue):
        """
        Send a message to the battleground indicating we are ready.
        """
        channel.basic_publish(
            exchange="",
            routing_key=queue,
            properties=pika.BasicProperties(
            ),
            body="{}_READY".format(agent_type.upper()),
        )
        self.ready = True
        return True


def run_combatant(agent_type, agent_path, agent_name, basic_agents_path):
    """
    Run a combatant agent in the tournament environment.
    Load the agent from the agent path, and subscribe to a RabbitMQ queue.
    """

    agent = Combatant(agent_path, agent_name, basic_agents_path)

    agent_queue = "rpc_queue_{}".format(agent_type)

    ready_queue = "rpc_queue_ready"

    if "RABBITMQ_HOST" in os.environ.keys():

        hostname = os.environ["RABBITMQ_HOST"]
    else:
        hostname = "localhost"

    connected = False
    while not connected:
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=hostname)
            )
            connected = True
        except (
            pika.exceptions.AMQPConnectionError,
            AMQPConnectorSocketConnectError,
        ):
            logger.info("Waiting for connection...")
            time.sleep(2)
    channel = connection.channel()

    channel.queue_declare(queue=agent_queue)
    channel.queue_declare(queue=ready_queue)

    channel.basic_qos(prefetch_count=1)

    channel.basic_consume(
        queue=agent_queue, on_message_callback=agent.get_action
    )

    agent.send_ready(agent_type, connection, channel, ready_queue)
    print(" [x] {} Awaiting RPC requests".format(agent_type.upper()))
    channel.start_consuming()


if __name__ == "__main__":
    """"""

    agent_type = sys.argv[1].lower()

    if agent_type not in ["pelican", "panther"]:
        raise RuntimeError(
            "First argument to combatant.py must be 'pelican' or 'panther'"
        )

    agent_path = os.path.join(AGENTS_PATH, agent_type)
    subdirs = os.listdir(agent_path)
    for subdir in subdirs:
        if os.path.isdir(os.path.join(agent_path, subdir)):
            agent_path = os.path.join(agent_path, subdir)
            break

    agent_name = "comb_%s" % (agent_type)

    run_combatant(agent_type, agent_path, agent_name, BASIC_AGENTS_PATH)
