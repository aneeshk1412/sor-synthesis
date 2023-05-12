import json

import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt

from highway_env.vehicle.behavior import IDMVehicle
from highway_env.road.road import Road, Route
from highway_env.utils import Vector
import random


class LDIPSState(object):
    def __init__(self, state) -> None:
        self.state = state

    def get(self, name):
        return self.state[name]["value"]


class MyVehicle(IDMVehicle):
    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        target_lane_index: int = None,
        target_speed: float = None,
        route: Route = None,
        enable_lane_change: bool = True,
        timer: float = None,
    ):
        speed = random.choice(
            _other_speeds
        )  ## Change velocity of car in front using this variable
        target_speed = speed  ## TODO: Not working yet for above 30
        super().__init__(
            road, position, heading, speed, target_lane_index, target_speed, route
        )


def dimensionless_template(name, value):
    return {
        name: {
            "dim": [0, 0, 0],
            "type": "NUM",
            "name": name,
            "value": value,
        }
    }


def distance_template(name, value):
    return {
        name: {
            "dim": [1, 0, 0],
            "type": "NUM",
            "name": name,
            "value": value,
        }
    }


def speed_template(name, value):
    return {
        name: {
            "dim": [1, -1, 0],
            "type": "NUM",
            "name": name,
            "value": value,
        }
    }


def start_template(value):
    return {
        "start": {"dim": [0, 0, 0], "type": "STATE", "name": "start", "value": value}
    }


def output_template(value):
    return {
        "output": {"dim": [0, 0, 0], "type": "STATE", "name": "output", "value": value}
    }


def compute_state(obs):
    """Compute the state in LDIPS format (json object) from the given observation"""
    return {
        **dimensionless_template("x_diff", float(obs[1][0] - obs[0][0])),
        **dimensionless_template("v_diff", float(obs[1][1] - obs[0][1])),
    }


def compute_ldips_state(obs, prev_action):
    """Compute the LDIPS state in LDIPS format (json object) from the given observation"""
    return LDIPSState(
        {
            **start_template(prev_action),
            **compute_state(obs),
        }
    )


def compute_ldips_sample(obs, prev_action, action):
    """Compute the Full LDIPS sample in LDIPS format (json object) from the given observation"""
    return LDIPSState(
        {
            **start_template(prev_action),
            **compute_state(obs),
            **output_template(action),
        }
    )


def print_debug(reward, done, truncated, state, info):
    print("Reward: {}, Done: {}, Truncated: {}".format(reward, done, truncated))
    print(f"State: {state.get('x_diff') = } {state.get('v_diff') = }")
    print(info)
    print()


def save_trace_to_json(trace, filename="demo.json"):
    trace_json = json.dumps([s.sample for s in trace])
    with open(filename, "w") as f:
        f.write(trace_json)


OTHER_SPEED_RANGE_LOW = 10  # [m/s]
OTHER_SPEED_RANGE_HIGH = 11  # [m/s]
OTHER_SPEED_INTERVAL = 1  # [m/s]

EGO_SPEED_RANGE_LOW = 10  # [m/s]
EGO_SPEED_RANGE_HIGH = 30  # [m/s]
EGO_SPEED_INTERVAL = 1  # [m/s]

DURATION = 50  # [s]

DESIRED_DISTANCE = 30  # [m] Desired distance between ego and other vehicle

MIN_DIST = 10  # [m] Minimum distance between ego and other vehicle in initial state
MAX_DIST = 70  # [m] Maximum distance between ego and other vehicle in initial state
D_CRASH = 5  # [m] Distance at which crash occurs in simulation

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
## space between ego and other is vehicle_distance = 30 / vehicle_density
vehicle_densities_choices = np.linspace(30 / MAX_DIST, 30 / MIN_DIST, _dist_num_points)

config = {
    "lanes_count": 1,
    "duration": DURATION,
    "policy_frequency": 20,
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
        "target_speeds": _ego_speeds,  ## Speed range of ego vehicle
    },
    "vehicles_count": 1,
    "other_vehicles_type": "highway_one_lane_simulator.MyVehicle",
}


def init_condition(state):
    v_diff = state.get("v_diff")
    x_diff = state.get("x_diff")
    return v_diff**2 / 2 - x_diff > D_CRASH


def run_simulation(policy, spec, show=False):
    env = gym.make("highway-v0", render_mode="rgb_array")
    env.configure(config)
    env.config["vehicles_density"] = random.choice(vehicle_densities_choices)
    env.reset()
    trace = []
    sat = True

    action = "SLOWER"
    action_idx = env.action_type.actions_indexes[action]
    count = 0
    while True:
        count += 1
        obs, reward, done, truncated, info = env.step(action_idx)
        prev_action = action

        state = compute_ldips_state(obs, prev_action)
        action = policy(state)
        action_idx = env.action_type.actions_indexes[action]
        assert env.action_space.contains(action_idx)

        sample = compute_ldips_sample(obs, prev_action, action)
        trace.append(sample)

        if show:
            env.render()
        if done:
            break
        if truncated or state.get("x_diff") < 0:
            # Corner case when vehicle in front goes out of view
            # remove last element from history
            trace.pop()
            break
    if not spec(trace):
        sat = False
    if show:
        for s in trace:
            print(s.state)
            print()
        plt.imshow(env.render())
    return sat, trace


def verify_policy(policy, spec, show=False, num_runs=20):
    sat = True
    i = 0
    while i < num_runs:
        print("ITERATION", i, "of", num_runs, "runs")
        sat, trace = run_simulation(policy, spec, show=show)
        if sat is None:
            continue
        if sat is False:
            return sat, trace
        i += 1
    return True, []


def policy_ground_truth(state):
    """Ground truth policy."""
    x_diff = state.get("x_diff")
    v_diff = state.get("v_diff")

    pre = state.get("start")
    fast_to_slow = ((v_diff**2) / 2 - x_diff > -DESIRED_DISTANCE) and v_diff < 0
    fast_to_fast = ((v_diff**2) / 2 + x_diff > DESIRED_DISTANCE) and v_diff > 0
    slow_to_fast = ((v_diff**2) / 2 + x_diff > DESIRED_DISTANCE) and v_diff > 0
    slow_to_slow = ((v_diff**2) / 2 - x_diff > -DESIRED_DISTANCE) and v_diff < 0

    if pre == "SLOWER":
        if slow_to_fast:
            post = "FASTER"
        elif slow_to_slow:
            post = "SLOWER"
        else:
            post = "SLOWER"
    elif pre == "FASTER":
        if fast_to_slow:
            post = "SLOWER"
        elif fast_to_fast:
            post = "FASTER"
        else:
            post = "FASTER"
    return post


def spec_1(trace):
    last_state = trace[-1]
    if abs(last_state.get("x_diff") - DESIRED_DISTANCE) > 1:
        return False
    else:
        return True


if __name__ == "__main__":
    sat, trace = run_simulation(policy_ground_truth, spec_1, show=True)
    print(sat, len(trace))
