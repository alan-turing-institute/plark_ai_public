#!/usr/bin/env python

import os
import sys
import json
from combatant import load_combatant
from schema import deserialize_state
import numpy as np

from combatant import AGENTS_PATH

ALLOWED_ACTIONS = {
    "PANTHER": [
        "1",  # Up
        "2",  # Up right
        "3",  # Down right
        "4",  # Down
        "5",  # Down left
        "6",  # Up left
        "end",
    ],
    "PELICAN": [
        "1",  # Up
        "2",  # Up right
        "3",  # Down right
        "4",  # Down
        "5",  # Down left
        "6",  # Up left
        "drop_buoy",
        "drop_torpedo",
        "end",
    ],
}

#AGENT_NAME = ""

BASIC_AGENTS_PATH = os.path.join(
    os.path.abspath(
        os.path.join(
            os.path.abspath(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
            ),
            os.pardir,
        )
    ),
    "Components",
    "plark-game",
    "plark_game",
    "agents",
    "basic"
)

def main():

    agent_type = sys.argv[1].upper()
    AGENT_NAME = "comb_%s" % (agent_type)

    if agent_type not in ["PELICAN", "PANTHER"]:
        raise Exception("Agent type must PELICAN or PANTHER: %s" % (agent_type))
    
    if not os.path.exists(AGENTS_PATH):
        raise Exception("Given agent path doesn't exist: %s" % (AGENTS_PATH))

    if agent_type == "PELICAN":
        agent_path = os.path.join(AGENTS_PATH, "pelican")
        test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pelican")
    else:
        agent_path = os.path.join(AGENTS_PATH, "panther")
        test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "panther")

    state = deserialize_state(json.load(open(os.path.join(test_path, "10x10_state.json"))))

    obs = np.loadtxt(os.path.join(test_path, "10x10_obs.txt"))
    obs_norm = np.loadtxt(os.path.join(test_path, "10x10_obs_norm.txt"))
    d_params = np.loadtxt(os.path.join(test_path, "10x10_domain_params.txt"))
    d_params_norm = np.loadtxt(os.path.join(test_path, "10x10_domain_params_norm.txt"))

    agent = load_combatant(agent_path, AGENT_NAME, BASIC_AGENTS_PATH)

    action = agent.getTournamentAction(obs, obs_norm, d_params, d_params_norm, state)

    if action not in ALLOWED_ACTIONS[agent_type]:
        raise RuntimeError("NO!")
    else:
        print("Test successful")


if __name__ == "__main__":
    
    main()
