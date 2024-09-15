import gymnasium
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")

n_states = env.observation_space.n
n_actions = env.action_space.n

R = np.zeros((n_states, n_actions))
P = np.zeros((n_states, n_actions, n_states))
T = np.zeros((n_states, n_actions))

env.reset()
for s in range(n_states):
    for a in range(n_actions):
        env.set_state(s)
        s_next, r, terminated, _, _ = env.step(a)
        R[s, a] = r
        P[s, a, s_next] = 1.0
        T[s, a] = terminated

# assume state transitioning are all deterministic in two functions below

def bellman_v(init_value, policy, gamma, threshold):
    env.reset()
    v_curr, v_new = [init_value for _ in range(n_states)], [0 for _ in range(n_states)]
    errors = []
    delta = 1000 # anything > threshold, just for start
    while delta > threshold:
        error, delta = 0, 0
        for s in range(n_states):
            env.set_state(s)
            a = policy[s] # assume deterministic policy
            s_next, r, terminated, _, _ = env.step(a)
            # assume deterministic environment
            val = r if terminated else r + gamma * v_curr[s_next]
            v_new[s] = val
            error += abs(val - v_curr[s])
            delta = max(delta, abs(val - v_curr[s]))
        v_curr, v_new = v_new, [0 for _ in range(n_states)]
        errors.append(error)
    return v_curr, errors

def bellman_q(init_value, policy, gamma, threshold):
    env.reset()
    q_curr, q_new = [init_value for _ in range(n_states * n_actions)], [0 for _ in range(n_states * n_actions)]
    errors = []
    delta = 1000 # anything > threshold, just for start
    while delta > threshold:
        error, delta = 0, 0
        for s in range(n_states):
            for a in range(n_actions):
                env.set_state(s)
                s_next, r, terminated, _, _ = env.step(a)
                a_next = policy[s_next]
                # assume deterministic environment
                if terminated:
                    val = r
                else:
                    val = r + gamma * q_curr[s_next*n_actions + a_next]
                q_new[s*n_actions + a] = val
                error += abs(val - q_curr[s*n_actions + a])
                delta = max(delta, abs(val - q_curr[s*n_actions + a]))
        q_curr = q_new
        q_new = [0 for _ in range(n_states * n_actions)]
        errors.append(error)
    return q_curr, errors

def policy_improvement(v, pi, gamma):
    stable = True
    for s in range(n_states):
        old_a = pi[s]
        curr_max = -float('inf')
        for a in range(n_actions):
            env.set_state(s)
            s_next, r, terminated, _, _ = env.step(a)
            val = r if terminated else r + gamma * v[s_next]
            if val > curr_max:
                curr_max = val
                pi[s] = a # pi to greedily follow v
        if not isinstance(old_a, int) or old_a != pi[s]:
            stable = False
    return pi, stable

def policy_evaluation(v_curr, pi, gamma, threshold, max_iter=None):
    env.reset()
    v_new = [0 for _ in range(n_states)]
    errors = []
    delta = 1000 # anything > threshold, just for start
    num_iter = 0 # count to 5 then stop for GPI
    while True:
        if max_iter is not None and num_iter >= max_iter:
            break
        if delta <= threshold:
            break
        num_iter += 1
        error, delta = 0, 0
        for s in range(n_states):
            if isinstance(pi[s], int): # deterministic
                env.set_state(s)
                s_next, r, terminated, _, _ = env.step(pi[s])
                val = r if terminated else r + gamma * v_curr[s_next]
            else:
                val = 0
                for a, p in pi[s]: # each element is (action, probability)
                    env.set_state(s)
                    s_next, r, terminated, _, _ = env.step(a)
                    val += p*r if terminated else p*(r + gamma * v_curr[s_next])
            v_new[s] = val
            error += abs(v_new[s] - v_curr[s])
            delta = max(delta, abs(v_new[s] - v_curr[s]))
        v_curr = v_new
        v_new = [0 for _ in range(n_states)]
        errors.append(error)
    return pi, v_curr, errors, delta

def policy_iteration(init_value, pi, gamma, threshold):
    v = [init_value for _ in range(n_states)]
    be = []
    while True:
        pi, v, errors, _ = policy_evaluation(v, pi, gamma, threshold)
        be.extend(errors)
        pi, stable = policy_improvement(v, pi, gamma)
        if stable:
            return pi, len(be), be

def generalized_policy_iteration(init_value, pi, gamma, threshold):
    v = [init_value for _ in range(n_states)]
    be = []
    while True:
        pi, v, errors, delta = policy_evaluation(v, pi, gamma, threshold, max_iter=5)
        be.extend(errors)
        pi, stable = policy_improvement(v, pi, gamma)
        if stable and delta <= threshold:
            return pi, len(be), be

def value_iteration(init_value, gamma, threshold):
    env.reset()
    v_curr, v_new = [init_value for _ in range(n_states)], [0 for _ in range(n_states)]
    errors = []
    delta = 1000 # anything > threshold, just for start
    pi = [0 for _ in range(n_states)]
    while delta > threshold:
        error, delta = 0, 0
        for s in range(n_states):
            curr_max = -float('inf')
            for a in range(n_actions):
                env.set_state(s)
                s_next, r, terminated, _, _ = env.step(a)
                # assume deterministic environment
                val = r if terminated else r + gamma * v_curr[s_next]
                if curr_max <= val:
                    curr_max = val
                    pi[s] = a
            v_new[s] = curr_max
            error += abs(v_new[s] - v_curr[s])
            delta = max(delta, abs(v_new[s] - v_curr[s]))
        v_curr, v_new = v_new, [0 for _ in range(n_states)]
        errors.append(error)
    return pi, len(errors), errors

fig, axs = plt.subplots(3, 7)
tot_iter_table = np.zeros((3, 7))

# optimal policy, manually defined
pi_opt = [1, 2, 4, 1, 2, 3, 2, 2, 3]

gamma, threshold = 0.99, 0.00001

for i, init_value in enumerate([-100, -10, -5, 0, 5, 10, 100]):
    axs[0][i].set_title(f'$V_0$ = {init_value}')

    pi, tot_iter, be = value_iteration(init_value, gamma, threshold)
    tot_iter_table[0, i] = tot_iter
    assert np.allclose(pi, pi_opt)
    axs[0][i].plot(be)

    # equiprobable policy
    pi_random = [[(a, 1/n_actions) for a in range(n_actions)] for s in range(n_states)]
    pi, tot_iter, be = policy_iteration(init_value, pi_random, gamma, threshold)
    tot_iter_table[1, i] = tot_iter
    assert np.allclose(pi, pi_opt)
    axs[1][i].plot(be)

    # equiprobable policy
    pi_random = [[(a, 1/n_actions) for a in range(n_actions)] for s in range(n_states)]
    pi, tot_iter, be = generalized_policy_iteration(init_value, pi_random, gamma, threshold)
    tot_iter_table[2, i] = tot_iter
    assert np.allclose(pi, pi_opt)
    axs[2][i].plot(be)

    if i == 0:
        axs[0][i].set_ylabel("VI")
        axs[1][i].set_ylabel("PI")
        axs[2][i].set_ylabel("GPI")

plt.show()
print(tot_iter_table.mean(-1))
print(tot_iter_table.std(-1))

