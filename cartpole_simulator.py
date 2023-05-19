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
        return 1
    else:
        return 0


def simulate(policy, render=False):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    env.reset()
    trace = []
    action = env.action_space.sample()
    while True:
        obs, reward, done, truncated, info = env.step(action)
        prev_action = action
        state = LDIPSSample(obs, cartpole_state_fn, prev_action, None)
        action = policy(state)
        sample = LDIPSSample(obs, cartpole_state_fn, int(prev_action), int(action))
        trace.append(sample.sample)
        if done:
            break
    env.close()
    return trace


if __name__ == "__main__":
    ## Simulate the expert policy and collect initial samples.
    traces = []
    for i in range(5):
        traces.extend(simulate(expert_policy))

    with open("cartpole_samples.json", "w") as f:
        json.dump(traces, f, indent=4)

    ## Run LDIPS on the samples to get policy1.
