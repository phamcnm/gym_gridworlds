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

def bellman_q(pi, gamma, max_iter=1000):
    delta = np.inf
    iter = 0
    Q = np.zeros((n_states, n_actions))
    be = np.zeros((max_iter))
    while delta > 1e-5 and iter < max_iter:
        Q_new = R + (np.dot(P, gamma * (Q * pi)).sum(-1))
        delta = np.abs(Q_new - Q).sum()
        be[iter] = delta
        Q = Q_new
        iter += 1
    return Q

def calc_be(Q, gamma, eps, alg):
    if alg == 'QL':
        target_pi = eps_greedy_probs(Q, 0)
        target_q = bellman_q(target_pi, gamma)
    elif alg == 'SARSA':
        target_pi = eps_greedy_probs(Q, eps)
        target_q = bellman_q(target_pi, gamma)
    else:
        target_pi = eps_greedy_probs(Q, eps)
        target_q = bellman_q(target_pi, gamma)
    return np.abs(target_q - Q).mean()

def eps_greedy_probs(Q, eps):
    pi = np.full((n_states, n_actions), eps / n_actions)  # Fill with epsilon/n_actions
    
    for i in range(n_states):
        max_action = np.argmax(Q[i])  # Find the index of the max element in the row
        pi[i, max_action] = 1 - eps + (eps / n_actions)  # Set the value for the max element
    
    return pi

def eps_greedy_probs_state(Q, s, eps):
    pi_s = np.full((n_actions), eps / n_actions)  # Fill with epsilon/n_actions
    max_action = np.argmax(Q[s])  # Find the index of the max element in the row
    pi_s[max_action] = 1 - eps + (eps / n_actions)  # Set the value for the max element
    return pi_s

def eps_greedy_action(Q, s, eps):
    if np.random.rand() > eps:
        return np.argmax(Q[s])
    return np.random.randint(0, len(Q[s]))

def expected_return(env, Q, gamma, episodes=10):
    G = np.zeros(episodes)
    for e in range(episodes):
        s, _ = env.reset(seed=e)
        done = False
        t = 0
        while not done:
            a = eps_greedy_action(Q, s, 0.0)
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            G[e] += gamma**t * r
            s = s_next
            t += 1
    return G.mean()

def print_policy(Q):
    res = [[0 for _ in range(3)] for _ in range(3)]
    for i in range(9):
        p = eps_greedy_probs_state(Q, i, 0)
        res[i//3][i%3] = p.argmax()
    return res

def is_opt(pi):
    return pi == [[1, 2, 4], [1, 2, 3], [2, 2, 3]]

def td(env, env_eval, Q, gamma, eps, alpha, max_steps, alg):
    be = []
    exp_ret = []
    tde = np.zeros(max_steps)
    tot_steps = 0
    while True:
        s, _ = env.reset(seed=tot_steps)
        done = False
        while not done:
            if tot_steps % 100 == 0:
                greedy_val = expected_return(env_eval, Q, gamma)
                exp_ret.append(greedy_val)
                error = calc_be(Q, gamma, eps, alg)
                be.append(error)
            a = eps_greedy_action(Q, s, eps)
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            target_value = 0
            if not terminated:
                if alg == 'QL':
                    target_value = max(Q[s_next])
                elif alg == 'SARSA':
                    a_next = eps_greedy_action(Q, s_next, eps)
                    target_value = Q[s_next, a_next]
                elif alg == 'Exp_SARSA':
                    pi_s = eps_greedy_probs_state(Q, s_next, eps)
                    for a_next in range(len(pi_s)):
                        target_value += pi_s[a_next] * Q[s_next, a_next]
            td_error = r + gamma * target_value - Q[s, a]
            tde[tot_steps] = abs(td_error)
            Q[s, a] += alpha*td_error
            eps = max(eps - 1.0 / max_steps, 0.01)
            alpha = max(alpha - 0.1 / max_steps, 0.001)
            tot_steps += 1
            if tot_steps == max_steps-1:
                break
            s = s_next
        if tot_steps == max_steps-1:
            break
    return Q, be, tde, exp_ret

# https://stackoverflow.com/a/63458548/754136
def smooth(arr, span):
    re = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")
    re[0] = arr[0]
    for i in range(1, span + 1):
        re[i] = np.average(arr[: i + span])
        re[-i] = np.average(arr[-i - span :])
    return re

def error_shade_plot(ax, data, stepsize, smoothing_window=1, **kwargs):
    y = np.nanmean(data, 0)
    x = np.arange(len(y))
    x = [stepsize * step for step in range(len(y))]
    if smoothing_window > 1:
        y = smooth(y, smoothing_window)
    (line,) = ax.plot(x, y, **kwargs)
    error = np.nanstd(data, axis=0)
    if smoothing_window > 1:
        error = smooth(error, smoothing_window)
    error = 1.96 * error / np.sqrt(data.shape[0])
    ax.fill_between(x, y - error, y + error, alpha=0.2, linewidth=0.0, color=line.get_color())

gamma = 0.99
alpha = 0.1
eps = 1.0
max_steps = 10000
horizon = 10

init_values = [-10, 0.0, 10]
algs = ["QL", "SARSA", "Exp_SARSA"]
seeds = np.arange(10)

results_be = np.zeros((
    len(init_values),
    len(algs),
    len(seeds),
    max_steps // 100,
))
results_tde = np.zeros((
    len(init_values),
    len(algs),
    len(seeds),
    max_steps,
))
results_exp_ret = np.zeros((
    len(init_values),
    len(algs),
    len(seeds),
    max_steps // 100,
))

fig, axs = plt.subplots(1, 3)
plt.ion()
plt.show()

reward_noise_std = 0.0  # re-run with 3.0

for ax in axs:
    ax.set_prop_cycle(
        color=["red", "green", "blue", "black", "orange", "cyan", "brown", "gray", "pink"]
    )
    ax.set_xlabel("Steps")

env = gymnasium.make(
    "Gym-Gridworlds/Penalty-3x3-v0",
    max_episode_steps=horizon,
    reward_noise_std=reward_noise_std,
)

env_eval = gymnasium.make(
    "Gym-Gridworlds/Penalty-3x3-v0",
    max_episode_steps=horizon,
)

print()
print('# of times optimal policies were found')
for i, init_value in enumerate(init_values):
    print()
    print('---init value', init_value)
    for j, alg in enumerate(algs):
        count, total_count = 0, 0
        for seed in seeds:
            np.random.seed(seed)
            Q = np.zeros((n_states, n_actions)) + init_value
            Q, be, tde, exp_ret = td(env, env_eval, Q, gamma, eps, alpha, max_steps, alg)
            results_be[i, j, seed] = be
            results_tde[i, j, seed] = tde
            results_exp_ret[i, j, seed] = exp_ret
            total_count += 1
            if is_opt(print_policy(Q)):
                count += 1
            # print(i, j, seed)
        print(alg, ":", count, "/", total_count)

        label = f"$Q_0$: {init_value}, Alg: {alg}"
        axs[0].set_title("TD Error")
        error_shade_plot(
            axs[0],
            results_tde[i, j],
            stepsize=1,
            smoothing_window=20,
            label=label,
        )
        axs[0].legend()
        axs[0].set_ylim([0, 5])
        axs[1].set_title("Bellman Error")
        error_shade_plot(
            axs[1],
            results_be[i, j],
            stepsize=100,
            smoothing_window=20,
            label=label,
        )
        axs[1].legend()
        axs[1].set_ylim([0, 50])
        axs[2].set_title("Expected Return")
        error_shade_plot(
            axs[2],
            results_exp_ret[i, j],
            stepsize=100,
            smoothing_window=20,
            label=label,
        )
        axs[2].legend()
        axs[2].set_ylim([-5, 1])
        plt.draw()
        plt.pause(0.001)

plt.ioff()
plt.show()