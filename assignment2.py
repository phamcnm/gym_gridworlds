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

label_size = 7
policy = [1, 2, 4, 1, 2, 3, 2, 2, 3] # optimal policy, manually defined
gammas = [0.01, 0.5, 0.99]
for init_value in [-10, 0, 10]:
    fig, axs = plt.subplots(2, len(gammas))
    fig.suptitle(f"$V_0$: {init_value}")
    for i, gamma in enumerate(gammas):
        v, errors = bellman_v(init_value, policy, gamma, 0.000001)
        tile_values = np.array(v).reshape((3, 3))
        im = axs[0][i].imshow(tile_values)
        cbar = fig.colorbar(im, ax=axs[0][i], fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=label_size)
        cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins='auto')) 
        axs[0][i].set_xticks([])
        axs[0][i].set_yticks([])
        axs[1][i].plot(errors)
        axs[0][i].set_title(f'$\gamma$ = {gamma}')
        axs[1][i].xaxis.set_major_locator(MaxNLocator(integer=True, nbins='auto'))
        axs[1][i].tick_params(axis='x', labelsize=label_size)
        axs[1][i].tick_params(axis='y', labelsize=label_size)

    fig, axs = plt.subplots(n_actions + 1, len(gammas))
    fig.suptitle(f"$Q_0$: {init_value}")
    row_labels = ['LEFT', 'DOWN', 'RIGHT', 'UP', 'STAY']
    for i, gamma in enumerate(gammas):
        q, errors = bellman_q(init_value, policy, gamma, 0.000001)
        for a in range(n_actions):
            axs[a][0].set_ylabel(row_labels[a], rotation=0, labelpad=40, fontsize=12, va='center')
            tile_values = np.array([q[s*n_actions+a] for s in range(n_states)]).reshape((3, 3))
            im = axs[a][i].imshow(tile_values)
            cbar = fig.colorbar(im, ax=axs[a][i], fraction=0.046, pad=0.03)
            cbar.ax.tick_params(labelsize=label_size)
            cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins='auto')) 
            axs[a][i].set_xticks([])
            axs[a][i].set_yticks([])
        axs[-1][i].plot(errors)
        axs[0][i].set_title(f'$\gamma$ = {gamma}')
        axs[0][i].tick_params(axis='x', labelsize=label_size)
        axs[0][i].tick_params(axis='y', labelsize=label_size)
        axs[n_actions][i].xaxis.set_major_locator(MaxNLocator(integer=True, nbins='auto'))
        axs[n_actions][i].tick_params(axis='x', labelsize=label_size)
        axs[n_actions][i].tick_params(axis='y', labelsize=label_size)
        
    plt.show()
