import random
import numpy as np
import gymnasium as gym

OTHER_SPEED_RANGE_LOW = 30  # [m/s]
OTHER_SPEED_RANGE_HIGH = 31  # [m/s]
OTHER_SPEED_INTERVAL = 1  # [m/s]

EGO_SPEED_RANGE_LOW = 20  # [m/s]
EGO_SPEED_RANGE_HIGH = 40  # [m/s]
EGO_SPEED_INTERVAL = 1  # [m/s]

DURATION = 40  # [s]

DESIRED_DISTANCE = 30  # [m] Desired distance between ego and other vehicle

# [m] Minimum distance between ego and other vehicle in initial state
MIN_DIST = 10
# [m] Maximum distance between ego and other vehicle in initial state
MAX_DIST = 11
D_CRASH = 5  # [m] Distance at which crash occurs in simulation

# set this to any value n>0 if you want to sample n elements for each transition type (e.g. SLOWER->FASTER) to be included in the demo.json
SAMPLES_NUMBER_PER_TRANSITION = 5


_other_speed_num_points = (
    int(OTHER_SPEED_RANGE_HIGH - OTHER_SPEED_RANGE_LOW) // OTHER_SPEED_INTERVAL + 1
)
_other_speeds = np.linspace(
    OTHER_SPEED_RANGE_LOW, OTHER_SPEED_RANGE_HIGH, _other_speed_num_points
)

_ego_speed_num_points = (
    int(EGO_SPEED_RANGE_HIGH - EGO_SPEED_RANGE_LOW) // EGO_SPEED_INTERVAL + 1
)
_ego_speeds = np.linspace(
    EGO_SPEED_RANGE_LOW, EGO_SPEED_RANGE_HIGH, _ego_speed_num_points
)

_dist_interval = 1
_dist_num_points = int(MAX_DIST - MIN_DIST) // _dist_interval + 1
# space between ego and other is vehicle_distance = 30 / vehicle_density
vehicle_densities_choices = np.linspace(
    30 / MAX_DIST, 30 / MIN_DIST, _dist_num_points)

config = {
    "lanes_count": 1,
    "duration": DURATION,
    "policy_frequency": 10,
    "simulation_frequency": 40,
    "screen_width": 1500,
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 2,
        "features": ["x", "vx", "presence"],
        "absolute": True,
        "normalize": False,
    },
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": _ego_speeds,  # Speed range of ego vehicle
    },
    "vehicles_count": 1,
    "other_vehicles_type": "highway_one_lane_simulator.MyVehicle",
}

def simulate(policy, render=False):
    """Simulate a policy on a random start state."""
    env = gym.make("highway-v0", render_mode="rgb_array" if render else None)
    env.configure(config)
    env.config["vehicles_density"] = random.choice(vehicle_densities_choices)
    env.reset()
    obs, reward, done, truncated, info = env.step(env.action_space.sample())
    while not done:
        obs, reward, done, truncated, info = env.step(env.action_space.sample())
        print("obs: {}".format(obs))
        if render:
            env.render()
    env.close()

simulate(None, render=True)
