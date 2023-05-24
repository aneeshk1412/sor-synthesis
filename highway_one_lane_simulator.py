import json
import sys
import os

import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from highway_env.vehicle.behavior import IDMVehicle
from highway_env.road.road import Road, Route
from highway_env.utils import Vector
import random

from pips.learned_policy import policy_ldips


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
        )  # Change velocity of car in front using this variable
        target_speed = speed  # TODO: Not working yet for above 30
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


def acceleration_template(name, value):
    return {
        name: {
            "dim": [1, -2, 0],
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
        **distance_template("x_diff", float(obs[1][0] - obs[0][0])),
        **speed_template("v_diff", float(obs[1][1] - obs[0][1])),
        **acceleration_template("acc", float(2.0)),
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
    trace_json = json.dumps([s.state for s in trace])
    with open(filename, "w") as f:
        f.write(trace_json)


def plot_series(policy, trace_1, trace_2):
    # the initial states in both experiments must be the same
    assert trace_1[0].state['x_diff']['value'] == trace_2[0].state['x_diff']['value']
    assert trace_1[0].state['v_diff']['value'] == trace_2[0].state['v_diff']['value']
    init_v_diff = trace_1[0].state['v_diff']['value']
    init_dist = trace_1[0].state['x_diff']['value']
    directory = 'plots/' + str(policy.__name__) + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # plot x diff
    plt.clf()
    diff_series_1 = [x.state['x_diff']['value'] for x in trace_1]

    # Create x-axis values ranging from 0 to the length of the data
    x1 = range(len(diff_series_1))

    actions = [x.state['output']['value'] for x in trace_1]
    # Plot the sorted data as a line chart

    if trace_2:
        diff_series_2 = [x.state['x_diff']['value'] for x in trace_2]
        x2 = range(len(diff_series_2))
        plt.plot(x1, diff_series_1, label='ldips')
        plt.plot(x2, diff_series_2, label='gt')
        print(f'{trace_2[0].get("x_diff")}', f'{trace_1[0].get("x_diff")}')
        print(f'{trace_2[0].get("v_diff")}', f'{trace_1[0].get("v_diff")}')
    else:
        # if the second trace is not given then this is a simulation of only gt
        plt.plot(x1, diff_series_1, label='GT')

    # Add the legend
    plt.legend(loc='upper right')

    # Add labels and title to the chart
    plt.xlabel('Time (100 ms)')
    plt.ylabel("Distance (m)")
    plt.title(f'Distance Between Cars vs. Time\n{init_dist=}\n{init_v_diff=}')
    plt.grid(True)
    # Set the minimum values of the x and y axes to 0
    plt.xlim(0, None)
    # plt.ylim(0, None)
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle='--', alpha=0.4)

    # Iterate over each action
    for i, action in enumerate(actions):
        # Determine the x-coordinate range for the rectangle
        start = i
        end = i + 1
        # Determine the color based on the action
        if action == 'FASTER':
            color = 'green'
        elif action == 'SLOWER':
            color = 'red'
        else:
            color = 'blue'
        # Add the colored rectangle
        # Adjust ymin and ymax values for rectangle height
        plt.axvspan(start, end, ymin=0, ymax=0.05, facecolor=color, alpha=0.8)
    # Save the chart as an image file
    plt.savefig(directory+'distance.png')


def pretty_str_state(state, iter):
    pre_action = state.state['start']['value']
    post_action = state.state['output']['value']
    distance = state.state['x_diff']['value']
    # v_self = state.state['v_self']['value']
    # v_front = state.state['v_front']['value']
    v_diff = state.state['v_diff']['value']
    if iter:
        result = '(' + str(iter) + ') '
    else:
        result = ''
    result += (pre_action + ' -> ' + post_action + ':\n')
    tab = '   '
    result += tab + 'distance: ' + str(distance) + '\n'
    # result += tab + 'v_self: ' + str(v_self) + '\n'
    # result += tab + 'v_front: ' + str(v_front) + '\n'
    result += tab + 'v_diff: ' + str(v_diff)
    return result


OTHER_SPEED_RANGE_LOW = 30  # [m/s]
OTHER_SPEED_RANGE_HIGH = 38  # [m/s]
OTHER_SPEED_INTERVAL = 1  # [m/s]

EGO_SPEED_RANGE_LOW = 28  # [m/s]
EGO_SPEED_RANGE_HIGH = 40  # [m/s]
EGO_SPEED_INTERVAL = 1  # [m/s]

DURATION = 60  # [s]

DESIRED_DISTANCE = 30  # [m] Desired distance between ego and other vehicle

# [m] Minimum distance between ego and other vehicle in initial state
MIN_DIST = 10
# [m] Maximum distance between ego and other vehicle in initial state
MAX_DIST = 10
D_CRASH = 5  # [m] Distance at which crash occurs in simulation

# set this to any value n>0 if you want to sample n elements for each transition type (e.g. SLOWER->FASTER) to be included in the demo.json
SAMPLES_NUMBER_PER_TRANSITION = 10


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
    "vehicles_density": random.choice(vehicle_densities_choices)
}


def run_simulation(policy, spec, show=False, env=None, init_obs=None):
    print(policy, env)
    sat = True
    
    count = 0
    stable_cnt = 0
    assert init_obs
    action = 'FASTER'  # let's assume the first action is always 'FASTER'
    init_state = compute_ldips_sample(init_obs[0], None, action)
    action_idx = env.action_type.actions_indexes[action]
    trace = [init_state]
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

        # check if stability has been achieved
        if state.get("x_diff") < 32 and state.get("x_diff") > 28:
            stable_cnt += 1
        else:
            stable_cnt = 0

        if show:
            env.render()
        if done:
            break
        if truncated or state.get("x_diff") < 0 or stable_cnt > 100:
            # Corner case when vehicle in front goes out of view
            # remove last element from history
            trace.pop()
            break
    if not spec(trace):
        sat = False
    if show:
        plt.imshow(env.render())
    return sat, trace


def policy_ground_truth(state):
    """Ground truth policy."""
    x_diff = state.get("x_diff")
    v_diff = state.get("v_diff")

    pre = state.get("start")
    fast_to_slow = ((v_diff**2) / 2 - x_diff > -
                    DESIRED_DISTANCE) and v_diff < 0
    fast_to_fast = ((v_diff**2) / 2 + x_diff > DESIRED_DISTANCE) and v_diff > 0
    slow_to_fast = ((v_diff**2) / 2 + x_diff > DESIRED_DISTANCE) and v_diff > 0
    slow_to_slow = ((v_diff**2) / 2 - x_diff > -
                    DESIRED_DISTANCE) and v_diff < 0

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


def analyze_trace(trace):
    result = {('SLOWER', 'SLOWER'): [], ('SLOWER', 'FASTER'): [],
              ('FASTER', 'FASTER'): [], ('FASTER', 'SLOWER'): []}
    iter = 0
    for s in trace:
        # print(pretty_str_state(s, iter))
        iter += 1
        # print('-'*50)
        pre_action = s.state['start']['value']
        post_action = s.state['output']['value']
        result[(pre_action, post_action)].append(s)
    return result


COUNT = int(DURATION * 0.85) * config['policy_frequency']
DELTA_DISTANCE_MAX = 3  # how much above the desired allowed to go


if __name__ == "__main__":
    # set the desired policy

    if len(sys.argv) != 2:
        print("Usage: python highway_one_lane_simulator.py <policy>")
        sys.exit(1)

    if sys.argv[1] == 'gt':
        env = gym.make("highway-v0", render_mode="rgb_array")
        env.configure(config)
        init_obs=env.reset()
        sat, trace = run_simulation(
            policy_ground_truth, spec_1, show=True, env=env, init_obs=init_obs)
        plot_series(policy=policy_ground_truth, trace_1=trace, trace_2=None)
        # we don't need the first element other than for plotting purposes
        trace.pop()

        # if you want to have a fix and same number of samples for each type of transition
        if SAMPLES_NUMBER_PER_TRANSITION > 0:
            sampled_trace = []
            samples_map = analyze_trace(trace=trace)
            for k in samples_map.keys():
                if len(samples_map[k]) >= SAMPLES_NUMBER_PER_TRANSITION:
                    for s in random.sample(samples_map[k], SAMPLES_NUMBER_PER_TRANSITION):
                        sampled_trace.append(s)
                else:
                    for s in samples_map[k]:
                        sampled_trace.append(s)
            save_trace_to_json(trace=sampled_trace,
                               filename='demos/sampled_demo.json')
            save_trace_to_json(trace=trace, filename='demos/full_demo.json')
    ########
    elif sys.argv[1] == 'ldips':
        env = gym.make("highway-v0", render_mode="rgb_array")
        env.configure(config)
        init_obs = env.reset()
        sat, trace_ldips = run_simulation(
            policy_ldips, spec_1, show=True, env=env, init_obs=init_obs)
        env.reset()
        _, trace_gt = run_simulation(
            policy_ground_truth, spec_1, show=True, env=env, init_obs=init_obs)

        plot_series(policy=policy_ldips, trace_1=trace_ldips, trace_2=trace_gt)
        # save_trace_to_json(trace=trace, filename='demos/full_demo.json')
        
        # we don't need the first element other than for plotting purposes
        trace_ldips.pop()

        # HACKY CHEATY repair using GT
        violation_found = False
        repaired_samples_json = []
        cex_cnt = 0
        fast_to_slow_repair = 0
        slow_to_fast_repair = 0
        trace_ldips_popped = trace_ldips[1:]
        random.shuffle(trace_ldips_popped)
        for i, s in enumerate(trace_ldips_popped):
            gt_action = policy_ground_truth(s)
            if s.state['output']['value'] != gt_action:
                if gt_action == 'FASTER':
                    slow_to_fast_repair += 1
                else:
                    fast_to_slow_repair += 1

                cex_cnt += 1
                violation_found = True
                # print('State BEFORE repair:')
                # print(pretty_str_state(state=s, iter=i))
                # print("GT prediction: ", gt_action)
                s.state['output']['value'] = gt_action
                repaired_samples_json.append(s.state)
                # print('State AFTER repair:')
                # print(pretty_str_state(state=s, iter=i))
                # only one state should be repaired per iteration
                if cex_cnt >= 30:
                    break

        print('-'*110)
        if violation_found:
            # print('All repaired samples:\n')
            # print(repaired_samples_json)
            print('Repaired sample stats:',
                  f'{fast_to_slow_repair=}', f'{slow_to_fast_repair=}')
            # write the repaiered samples into a file
            with open('demos/repaired_samples.json', "w") as f:
                f.write(json.dumps(repaired_samples_json))
        else:
            print('No violation was found!')
    else:
        raise Exception('policy should be either gt or ldips')
