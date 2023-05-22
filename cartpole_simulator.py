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
        trace.append(sample.sample)
        steps += 1
        if done:
            break
        if steps > max_steps:
            sat = True
            break
    env.close()
    return sat, trace


if __name__ == "__main__":
    ## Simulate the expert policy and collect initial samples.
    traces = []
    for i in range(5):
        _, trace = simulate(expert_policy)
        traces.append(trace)

    with open("cartpole_samples.json", "w") as f:
        json.dump([s for t in traces for s in t], f, indent=4)

    ## Run LDIPS on the samples to get policy1.
