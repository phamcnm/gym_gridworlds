import gymnasium
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")
n_states = env.observation_space.n
n_actions = env.action_space.n

R = np.zeros((n_states, n_actions))
P = np.zeros((n_states, n_actions, n_states))
T = np.zeros((n_states, n_actions))

env.reset()
for s in range(n_states):
    for a in range(n_actions):
        env.unwrapped.set_state(s)
        s_next, r, terminated, _, _ = env.step(a)
        R[s, a] = r
        P[s, a, s_next] = 1.0
        T[s, a] = terminated

P = P * (1.0 - T[..., None])  # next state probability for terminal transitions is 0

def bellman_q(pi, gamma):
    I = np.eye(n_states * n_actions)
    P_under_pi = (
        P[..., None] * pi[None, None]
    ).reshape(n_states * n_actions, n_states * n_actions)
    return (
        R.ravel() * np.linalg.inv(I - gamma * P_under_pi)
    ).sum(-1).reshape(n_states, n_actions)

def calc_error(pi, gamma, Q):
    return np.sum(np.abs(bellman_q(pi, gamma) - Q))

def episode(env, Q, eps, seed):
    data = dict()
    data["s"] = []
    data["a"] = []
    data["r"] = []
    s, _ = env.reset(seed=int(seed))
    done = False
    while not done:
        a = eps_greedy_action(Q, s, eps)
        s_next, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        data["s"].append(s)
        data["a"].append(a)
        data["r"].append(r)
        s = s_next
    return data

def eps_greedy_probs(Q, eps):
    # get eps-greedy pi from Q and eps
    pi = np.full((n_states, n_actions), eps / n_actions)
    for i in range(n_states):
        max_action = np.argmax(Q[i])
        pi[i, max_action] = 1 - eps + (eps / n_actions)
    
    return pi

def eps_greedy_probs_state(Q, s, eps):
    # get eps-greedy pi of only one state from Q and eps
    pi_s = np.full((n_actions), eps / n_actions)
    max_action = np.argmax(Q[s])
    pi_s[max_action] = 1 - eps + (eps / n_actions)
    return pi_s

def eps_greedy_action(Q, s, eps):
    # choose action with epsilon-greedy behavior
    if np.random.rand() > eps:
        return np.argmax(Q[s])
    return np.random.randint(0, len(Q[s]))

def print_policy(Q):
    res = [[0 for _ in range(3)] for _ in range(3)]
    for i in range(9):
        p = eps_greedy_probs_state(Q, i, 0)
        res[i//3][i%3] = p.argmax()
    return res

def is_opt(pi):
    return pi == [[1, 2, 4], [1, 2, 3], [2, 2, 3]]

def monte_carlo(env, Q, gamma, eps_decay, max_steps, episodes_per_iteration, use_is):
    eps = 1
    error = calc_error(eps_greedy_probs(Q, gamma), gamma, Q)
    be = [error]
    returns = np.empty((n_states, n_actions), dtype=object)

    if use_is:
        C = np.zeros((n_states, n_actions))
    
    prev_step, step_count = 1, 0
    for i in range(n_states):
        for j in range(n_actions):
            returns[i, j] = []
    while True:
        if use_is:
            for _ in range(episodes_per_iteration):
                data = episode(env, Q, eps, step_count)
                g = 0
                w = 1
                b_policy = eps_greedy_probs(Q, eps)
                # behavior policy to update only after every episode
                # update q, target policy, and log error every environment step
                # (unlike !use_is, which logs only every episode * episodes_per_iter)
                for step in range(len(data['s'])-1, -1, -1):
                    step_count += 1
                    s, a, r = data['s'][step], data['a'][step], data['r'][step]
                    g = gamma * g + r
                    C[s, a] += w
                    Q[s, a] += (w/C[s, a])*(g - Q[s, a])
                    t_policy = eps_greedy_probs(Q, 0.01)
                    t_policy_true_q = bellman_q(t_policy, gamma)
                    w *= t_policy[s, a]/b_policy[s, a]
                    returns[s, a].append(g)
                    error = np.sum(np.abs(t_policy_true_q - Q))
                    be.append(error)
                    if step_count >= max_steps-1:
                        break
                eps = max(eps - eps_decay * len(data['s']), 0.01)
                if step_count >= max_steps-1:
                    break
            if step_count >= max_steps-1:
                break
        else:
            for _ in range(episodes_per_iteration):
                data = episode(env, Q, eps, step_count)
                g = 0
                for step in range(len(data['s'])-1, -1, -1):
                    step_count += 1
                    s, a, r = data['s'][step], data['a'][step], data['r'][step]
                    g = gamma * g + r
                    returns[s, a].append(g)
                    if step_count >= max_steps:
                        break
                eps = max(eps - eps_decay * len(data['s']), 0.01)
                if step_count >= max_steps:
                    break
            for s in range(n_states):
                for a in range(n_actions):
                    if len(returns[s,a]) > 0:
                        Q[s, a] = sum(returns[s, a]) / len(returns[s, a])
            # fill previous time steps with previous error before updating with new error
            if step_count >= max_steps: 
                for _ in range(prev_step+1, max_steps):
                    be.append(error)
                error = calc_error(eps_greedy_probs(Q, eps), gamma, Q)
                prev_step = step_count
                be.append(error)
                break
            else:
                for _ in range(prev_step+1, step_count):
                    be.append(error)
                error = calc_error(eps_greedy_probs(Q, eps), gamma, Q)
                prev_step = step_count
                be.append(error)
        
    return Q, be


def error_shade_plot(ax, data, stepsize, **kwargs):
    y = np.nanmean(data, 0)
    x = np.arange(len(y))
    x = [stepsize * step for step in range(len(y))]
    (line,) = ax.plot(x, y, **kwargs)
    error = np.nanstd(data, axis=0)
    error = 1.96 * error / np.sqrt(data.shape[0])
    ax.fill_between(x, y - error, y + error, alpha=0.2, linewidth=0.0, color=line.get_color())

init_value = 0.0
gamma = 0.9
max_steps = 2000
horizon = 10

episodes_per_iteration = [1, 10, 50]
decays = [1, 2, 5]
seeds = np.arange(50)

results = np.empty((
    len(episodes_per_iteration),
    len(decays),
    len(seeds),
    max_steps
))

fig, axs = plt.subplots(1, 2)
plt.ion()
plt.show()

use_is = False  # repeat with True
count, total_count = 0, 0
for ax, reward_noise_std in zip(axs, [0.0, 3.0]):
    ax.set_prop_cycle(
        color=["red", "green", "blue", "black", "orange", "cyan", "brown", "gray", "pink"]
    )
    ax.set_xlabel("Steps")
    ax.set_ylabel("Absolute Bellman Error")
    env = gymnasium.make(
        "Gym-Gridworlds/Penalty-3x3-v0",
        max_episode_steps=horizon,
        reward_noise_std=reward_noise_std,
    )
    for j, episodes in enumerate(episodes_per_iteration):
        for k, decay in enumerate(decays):
            for seed in seeds:
                np.random.seed(seed)
                Q = np.zeros((n_states, n_actions)) + init_value
                Q, be = monte_carlo(env, Q, gamma, decay / max_steps, max_steps, episodes, use_is)
                results[j, k, seed] = be
                if is_opt(print_policy(Q)):
                    count += 1
                total_count += 1
            error_shade_plot(
                ax,
                results[j, k],
                stepsize=1,
                label=f"Episodes: {episodes}, Decay: {decay}",
            )
            ax.legend()
            plt.draw()
            plt.pause(0.001)
    print()
    print(count, '/', total_count)

plt.ioff()
plt.show()