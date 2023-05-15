import json
import os

import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt

from highway_env.vehicle.behavior import IDMVehicle
from highway_env.road.road import Road, Route
from highway_env.utils import Vector
import random

from learned_policy import policy_ldips


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
        **dimensionless_template("v_self", float(obs[0][1])),
        **dimensionless_template("v_front", float(obs[1][1])),
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


def plot_series(policy, trace):
    directory = 'plots/' + str(policy.__name__) + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # plot x diff
    plt.clf()
    x_diff_series = [x.state['x_diff']['value'] for x in trace]
    # Create x-axis values ranging from 0 to the length of the data
    x = range(len(x_diff_series))
    actions = [x.state['output']['value'] for x in trace]
    # Plot the sorted data as a line chart
    plt.plot(x, x_diff_series)

    # Add labels and title to the chart
    plt.xlabel('Time (100 ms)')
    plt.ylabel("Distance (m)")
    plt.title('Distance Between Cars vs. Time')
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

    # plot velocities
    plt.clf()
    v_self_series = [x.state['v_self']['value'] for x in trace]
    v_front_series = [x.state['v_front']['value'] for x in trace]
    # Create x-axis values ranging from 0 to the length of the data
    x = range(len(v_self_series))
    # Plot the sorted data as a line chart
    plt.plot(x, v_self_series)
    plt.plot(x, v_front_series)
    # Add labels and title to the chart
    plt.xlabel('Time (s)')
    plt.ylabel("Velocity (m/s)")
    plt.title('Velocities of the Cars vs. Time')
    plt.grid(True)
    # Set the minimum values of the x and y axes to 0
    plt.xlim(0, None)
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle='--', alpha=0.4)

    # plt.ylim(0, None)
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
    plt.savefig(directory+'velocity.png')


def pretty_str_state(state, iter):
    pre_action = state.state['start']['value']
    post_action = state.state['output']['value']
    distance = state.state['x_diff']['value']
    v_self = state.state['v_self']['value']
    v_front = state.state['v_front']['value']
    v_diff = state.state['v_diff']['value']
    result = '(' + str(iter) + ') '
    result += (pre_action + ' -> ' + post_action + ':\n')
    tab = '   '
    result += tab + 'distance: ' + str(distance) + '\n'
    result += tab + 'v_self: ' + str(v_self) + '\n'
    result += tab + 'v_front: ' + str(v_front) + '\n'
    result += tab + 'v_diff: ' + str(v_diff)
    return result


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
        print(pretty_str_state(s, iter))
        iter += 1
        print('-'*50)
        pre_action = s.state['start']['value']
        post_action = s.state['output']['value']
        result[(pre_action, post_action)].append(s)
    return result

COUNT = 20 * config['simulation_frequency']
DELTA_DISTANCE = 1 # [m]
DELTA_DISTANCE_MAX = 3 # how much above the desired allowed to go

def find_spec_1_breakpoint(trace):
    """ Distance always greater than D_CRASH """
    for i, s in enumerate(trace):
        if s.get('x_diff') <= D_CRASH:
            while i-1 >= 0:
                if trace[i-1].get('output') == 'FASTER':
                    return i-1, False
                i -= 1
            break
    return None, True

def find_spec_2_breakpoint(trace):
    """ Distance always less than DESIRED_DISTANCE + DELTA_DISTANCE """
    for i, s in enumerate(trace):
        if s.get('x_diff') >= DESIRED_DISTANCE + DELTA_DISTANCE_MAX:
            while i-1 >= 0:
                if trace[i-1].get('output') == 'SLOWER':
                    return i-1, False
                i -= 1
            break
    return None, True

def find_spec_3_breakpoint(trace):
    """ Distance between DESIRED_DISTANCE + DELTA_DISTANCE and DESIRED_DISTANCE - DELTA_DISTANCE after COUNT seconds """
    for i, s in enumerate(trace):
        if i < COUNT:
            continue
        if s.get('x_diff') >= DESIRED_DISTANCE + DELTA_DISTANCE:
            while i-1 >= 0:
                if trace[i-1].get('output') == 'SLOWER':
                    return i-1, False
                i -= 1
            break
        if s.get('x_diff') <= DESIRED_DISTANCE - DELTA_DISTANCE:
            while i-1 >= 0:
                if trace[i-1].get('output') == 'FASTER':
                    return i-1, False
                i -= 1
            break
    return None, True


if __name__ == "__main__":
    # set the desired policy here
    POLICY = policy_ldips
    #POLICY = policy_ground_truth

    sat, trace = run_simulation(POLICY, spec_1, show=True)
    plot_series(policy=POLICY, trace=trace)
    print(f'{sat=}, {len(trace)=}')

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

        save_trace_to_json(trace=sampled_trace, filename='sampled_demo.json')
    save_trace_to_json(trace=trace, filename='full_demo.json')

    ## automatic repair
    if POLICY == policy_ldips:
        i, sat = find_spec_1_breakpoint(trace)
        if not sat:
            print("Found broken spec 1:\n")
            sample = trace[i]
            print ('state before repair:')
            print (pretty_str_state(state=sample,iter=i))
            print()

            if sample.state['output']['value'] == "SLOWER":
               sample.state['output']['value'] = "FASTER"
            if sample.state['output']['value'] == "FASTER":
                sample.state['output']['value'] = "SLOWER"
            else:
                raise Exception("Invalid action")
            print(json.dumps(sample.state))
        else:
            i, sat = find_spec_2_breakpoint(trace)
            if not sat:
                print("Found broken spec 2:\n")
                sample = trace[i]
                print ('state before repair:')
                print (pretty_str_state(state=sample,iter=i))
                print()

                if sample.state['output']['value'] == "SLOWER":
                    sample.state['output']['value'] = "FASTER"
                elif sample.state['output']['value'] == "FASTER":
                    sample.state['output']['value'] = "SLOWER"
                else:
                    raise Exception("Invalid action")
                print(json.dumps(sample.state))
            else:
                i, sat = find_spec_3_breakpoint(trace)
                if not sat:
                    print("Found broken spec 3:\n")
                    sample = trace[i]
                    print ('state before repair:')
                    print (pretty_str_state(state=sample,iter=i))
                    print()
                    if sample.state['output']['value'] == "SLOWER":
                        sample.state['output']['value'] = "FASTER"
                    elif sample.state['output']['value'] == "FASTER":
                        sample.state['output']['value'] = "SLOWER"
                    else:
                        raise Exception("Invalid action")
                    print(json.dumps(sample.state))
                else:
                    print("No broken specs found")

    ## manual repair
    # if POLICY == policy_ldips:
    #     while(True):
    #         print (f'There are {len(trace)} transitions in the trace of the learned policy. Enter the index of the transition to repair:')
    #         idx = int(input())
    #         print('Here is the chosen transition sample:')
    #         sample = trace[idx]
    #         print (pretty_str_state(sample, idx))
    #         print ('continue?')
    #         cont = input()
    #         if cont in {'y','Y','yes'}:
    #             break
    #     print ('Enter correct post action:')
    #     repaired_post_action = input()
    #     assert repaired_post_action in {'SLOWER', 'FASTER'}
    #     print ('New repaired sample:')
    #     json_sample = sample.state
    #     json_sample['output']['value'] = repaired_post_action
    #     print (json.dumps(json_sample))




