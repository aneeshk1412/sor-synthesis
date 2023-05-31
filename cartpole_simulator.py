import json
import gymnasium as gym
from ldips_utils import *


def cartpole_state_fn(obs):
    return {
        **template("x", [1, 0, 0], "NUM", float(obs[0])),
        **template("x_dot", [1, -1, 0], "NUM", float(obs[1])),
        **template("theta", [0, 0, 0], "NUM", float(obs[2])),
        **template("theta_dot", [0, -1, 0], "NUM", float(obs[3])),
    }


def expert_policy(state: LDIPSSample):
    if state.get("theta") > 0:
        return "1"
    else:
        return "0"


def simulate(policy, max_steps=10000, render=False):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    env.reset()
    trace = []
    action = env.action_space.sample()
    steps = 0
    sat = False
    while True:
        obs, reward, done, truncated, info = env.step(action)
        prev_action = action
        state = LDIPSSample(obs, cartpole_state_fn, prev_action, None)
        action = int(policy(state))
        sample = LDIPSSample(obs, cartpole_state_fn, str(int(prev_action)), str(int(action)))
        trace.append(sample)
        steps += 1
        if done:
            break
        if steps > max_steps:
            sat = True
            break
    env.close()
    return sat, trace


def construct_condition_bitmaps(traces, conditions):
    ## Construct State -> Condition Bitvector
    state_to_condition_bitmap = dict()
    for trace in traces:
        for sample in trace:
            for i, condition in enumerate(conditions):
                if condition(sample):
                    if sample not in state_to_condition_bitmap:
                        state_to_condition_bitmap[sample] = [0] * len(conditions)
                    state_to_condition_bitmap[sample][i] = 1

    ## Construct Condition Bitvector -> List of States
    condition_bitmap_to_states = dict()
    for sample in state_to_condition_bitmap:
        condition_bitmap = tuple(state_to_condition_bitmap[sample])
        if condition_bitmap not in condition_bitmap_to_states:
            condition_bitmap_to_states[condition_bitmap] = []
        condition_bitmap_to_states[condition_bitmap].append(sample)

    return state_to_condition_bitmap, condition_bitmap_to_states


def construct_condition_bitmap_graph(traces, state_to_condition_bitmap):
    """ Graph where abstract states are the condition bitmaps """
    ## dict of dict s -> a -> set of s'
    graph_adjacency_list = dict()

    for trace in traces:
        for s1, s2 in zip(trace, trace[1:]):
            s1_bitmap = tuple(state_to_condition_bitmap[s1])
            s2_bitmap = tuple(state_to_condition_bitmap[s2])
            if s1_bitmap not in graph_adjacency_list:
                graph_adjacency_list[s1_bitmap] = dict()
            action = s1.get("output")
            if action not in graph_adjacency_list[s1_bitmap]:
                graph_adjacency_list[s1_bitmap][action] = set()
            graph_adjacency_list[s1_bitmap][action].add(s2_bitmap)

    return graph_adjacency_list


if __name__ == "__main__":
    ## Simulate the expert policy and collect initial samples.
    traces = []
    for i in range(5):
        _, trace = simulate(expert_policy, render=False)
        traces.append(trace)

    with open("cartpole_samples.json", "w") as f:
        json.dump([s.sample for t in traces for s in t], f, indent=4)

    ## Conditions in the expert policy and specification
    conditions = [
        lambda s: s.get("theta") > 0,
        ## below are the specification conditions
        lambda s: s.get("theta") < 0.2,
        lambda s: s.get("theta") > -0.2,
    ]

    ## Construct State -> Condition Bitvector and Condition Bitvector -> List of States
    state_to_condition_bitmap, condition_bitmap_to_states = construct_condition_bitmaps(traces, conditions)
    # print("State -> Condition Bitvector")
    # print(state_to_condition_bitmap)
    # print()
    # print("Condition Bitvector -> List of States")
    # print(condition_bitmap_to_states)

    ## Construct Graph where abstract states are the condition bitmaps
    graph_adjacency_list = construct_condition_bitmap_graph(traces, state_to_condition_bitmap)
    # print(graph_adjacency_list)

    ## Bad States are those that violate the specification
    bad_states = set(bv for bv in condition_bitmap_to_states if bv[1] == 0 or bv[2] == 0)
    # print("Bad States")
    # print(bad_states)

    ## Simple algorithm
    ## Find the transitions in traces that go from good to bad states
    for trace in traces:
        for s1, s2 in zip(trace, trace[1:]):
            s1_bitmap = tuple(state_to_condition_bitmap[s1])
            s2_bitmap = tuple(state_to_condition_bitmap[s2])
            if s1_bitmap not in bad_states and s2_bitmap in bad_states:
                print(s1.sample)
                ''' Next Steps
                   The concrete states within this good abstract state (equivalently the condition bitmap)
                   have to be split into two sets, one for which we will change the action and other where the action will not be changed
                   Ideas:
                   - Our old heuristic was only to change the concrete state whose action made us enter the bad state (penultimate state)
                   - Split the abstract states into two halves based on some sorting order (how to sort??) and split the concrete states according to this (then update all the traces accordingly and feed it back to LDIPS)
                   - Synthesize a condition to split the abstract states based on the concrete states in it and in the bad state (research question?)
                '''

