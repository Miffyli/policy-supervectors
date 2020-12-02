from collections import namedtuple

# Minimal agent class that can be created to work
# with collect_trajectories
SimpleAgentClass = namedtuple("SimpleAgent", ["get_action"])
