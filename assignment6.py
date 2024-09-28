import numpy as np
import gymnasium
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures  # makes poly features super easy

np.set_printoptions(precision=3, suppress=True)

# Notation for array sizes:
# - S: state dimensionality
# - D: features dimensionality
# - N: number of samples
#
# N is always the first dimension, meaning that states come in arrays of shape
# (N, S) and features in arrays of shape (N, D).
# We recommend to implement the functions below assuming that the input has
# always shape (N, S) and the output (N, D), even when N = 1.

def poly_features(state: np.array, degree: int) -> np.array:
    """
    Compute polynomial features. For example, if state = (s1, s2) and degree = 2,
    the output must be [1, s1, s2, s1*s2, s1**2, s2**2].
    """
    poly = PolynomialFeatures(degree=degree)
    res = poly.fit_transform(state)
    return res


def rbf_features(
    state: np.array,  # (N, S)
    centers: np.array,  # (D, S)
    sigmas: float,
) -> np.array:  # (N, D)
    """
    Computes exp(- ||state - centers||**2 / sigmas**2 / 2).
    """
    l2_squared = np.sum((state[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=-1)
    rbf_values = np.exp(-l2_squared / (sigmas ** 2 * 2))
    return rbf_values

def tile_features(
    state: np.array,  # (N, S)
    centers: np.array,  # (D, S)
    widths: float,
    offsets=[0],  # list of tuples of length S
) -> np.array:  # (N, D)
    """
    Given centers and widths, you first have to get an array of 0/1, with 1s
    corresponding to tile the state belongs to.
    If "offsets" is passed, it means we are using multiple tilings, i.e., we
    shift the centers according to the offsets and repeat the computation of
    the 0/1 array. The final output will sum the "activations" of all tilings.
    We recommend to normalize the output in [0, 1] by dividing by the number of
    tilings (offsets).
    Recall that tiles are squares, so you can't use the L2 Euclidean distance to
    check if a state belongs to a tile, but the absolute distance.
    Note that tile coding is more general and allows for rectangles (not just squares)
    but let's consider only squares for the sake of simplicity.
    """
    features = np.zeros((state.shape[0], centers.shape[0]))
    for offset in offsets:
        new_centers = centers + offset
        diffs = np.abs(state[:, np.newaxis, :] - new_centers[np.newaxis, :, :])
        inside_all_dims = np.abs(diffs) <= widths/2
        activations = np.all(inside_all_dims, axis=-1).astype(int)
        features += activations
    features /= len(offsets)

    return features


def coarse_features(
    state: np.array,  # (N, S)
    centers: np.array,  # (D, S)
    widths: float,
    offsets=[0], # list of tuples of length S
) -> np.array:  # (N, D)
    """
    Same as tile coding, but we use circles instead of squares, so use the L2
    Euclidean distance to check if a state belongs to a circle.
    Note that coarse coding is more general and allows for ellipses (not just circles)
    but let's consider only circles for the sake of simplicity.
    """
    features = np.zeros((state.shape[0], centers.shape[0]))
    for offset in offsets:
        new_centers = centers + offset
        broadcasted_diff = state[:, np.newaxis, :] - new_centers[np.newaxis, :, :]
        dist = np.linalg.norm(broadcasted_diff, axis=-1) #L2
        activations = (dist < widths).astype(int)
        features += activations
    features /= len(offsets)

    return features
    

def aggregation_features(state: np.array, centers: np.array) -> np.array:
    """
    Aggregate states to the closest center. The output will be an array of 0s and
    one 1 corresponding to the closest tile the state belongs to.
    Note that we can turn this into a discrete (finite) representation of the state,
    because we will have as many feature representations as centers.
    """
    l2_squared = np.sum((state[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=-1)
    closest_centers = np.argmin(l2_squared, axis=1)
    N, D = state.shape[0], centers.shape[0]
    aggregated_features = np.zeros((N, D))
    aggregated_features[np.arange(N), closest_centers] = 1
    return aggregated_features

state_size = 2
n_samples = 10
n_centers = 100
state = np.random.rand(n_samples, state_size)  # in [0, 1]

state_1_centers = np.linspace(-0.2, 1.2, n_centers)
state_2_centers = np.linspace(-0.2, 1.2, n_centers)
centers = np.array(
    np.meshgrid(state_1_centers, state_2_centers)
).reshape(state_size, -1).T  # makes a grid of uniformly spaced centers in the plane [-0.2, 1.2]^2
sigmas = 0.2
widths = 0.2
offsets = [(-0.1, 0.0), (0.0, 0.1), (0.1, 0.0), (0.0, -0.1)]

poly = poly_features(state, 2)
aggr = aggregation_features(state, centers)
rbf = rbf_features(state, centers, sigmas)
tile_one = tile_features(state, centers, widths)
tile_multi = tile_features(state, centers, widths, offsets)
coarse_one = coarse_features(state, centers, widths)
coarse_multi = coarse_features(state, centers, widths, offsets)

fig, axs = plt.subplots(1, 6)
extent = [
    state_1_centers[0],
    state_1_centers[-1],
    state_2_centers[0],
    state_2_centers[-1],
]  # to change imshow axes
axs[0].imshow(rbf[0].reshape(n_centers, n_centers), extent=extent, origin='lower')
axs[1].imshow(tile_one[0].reshape(n_centers, n_centers), extent=extent, origin='lower')
axs[2].imshow(tile_multi[0].reshape(n_centers, n_centers), extent=extent, origin='lower')
axs[3].imshow(coarse_one[0].reshape(n_centers, n_centers), extent=extent, origin='lower')
axs[4].imshow(coarse_multi[0].reshape(n_centers, n_centers), extent=extent, origin='lower')
axs[5].imshow(aggr[0].reshape(n_centers, n_centers), extent=extent, origin='lower')
titles = ["RBFs", "Tile (1 Tiling)", "Tile (4 Tilings)", "Coarse (1 Field)", "Coarse (4 Fields)", "Aggreg."]  # we can't plot poly like this
for ax, title in zip(axs, titles):
    ax.plot(state[0][0], state[0][1], marker="+", markersize=12, color="red")
    ax.set_title(title)
plt.suptitle(f"State {state[0]}")
plt.show()

# ################### PART 1
# Submit your heatmaps. 
# Note that the random seed is not fixed, so each of you will plot features 
# of a different point.
# What are the hyperparameters of each FA and how do they affect the shape of
# the function they can approximate?
# - In RBFs the hyperparameter(s) is/are ... More/less ... will affect ...,
#   while narrower/wider ... will affect ...
# - In tile/coarse coding the hyperparameter(s) is/are ...
# - In polynomials the hyperparameter(s) is/are ...
# - In state aggregation the hyperparameter(s) is/are ...
# Discuss each bullet point in at most two sentences.


#################### PART 2
# Consider the function below.

# With SL, (try to) train a linear approximation to fit the above function (y)
# using gradient descent.
# Start with all weights to 0.
# Feel free to use a better learning rate (maybe check the book for suggestions)
# and more iterations.
#
# - Use all 5 FAs implemented above and submit your plots. Select the
#   hyperparameters (degree, centers, sigmas, width, offsets) to achieve the best
#   results (lowest MSE).
# - Assume the state is now 2-dimensional with s in [-10, 10] x [0, 1000], i.e.,
#   the 2nd dimension is in the range [0, 1000] while the 1st is still in [-10, 10].
#   Also, assume that your implementation of RBFs, Tile Coding, and Coarse Coding
#   allows to pass different widths/sigmas for every dimension of the state.
#   How would you change the hyperparameters? Would you have more centers?
#   Or wider sigmas/widths? Both? Justify your answer in one sentence.
#
# Note: we don't want you to achieve MSE 0. Just have a decent plot with each FA,
# or discuss if some FA is not suitable to fit the given function, and report your plots.
# Anything like the demo plot is fine.

x = np.linspace(-10, 10, 100)
y = np.sin(x) + x**2 - 0.5 * x**3 + np.log(np.abs(x))
max_iter = 10000
thresh = 1e-8
centers = np.linspace(-10, 10, 30).reshape(-1, 1)
fig, axs = plt.subplots(2, 3)
axs[0][0].plot(x, y)
axs[0][0].set_title("True Function")

for idx, (name, get_phi) in enumerate(zip(["Poly", "RBFs", "Tiles", "Coarse", "Aggreg."],
    [
        lambda state : poly_features(state, degree=3),
        lambda state : rbf_features(state, centers=centers, sigmas=2),
        lambda state : tile_features(state, centers=centers, widths=1.7, offsets=[-0.5, 0.5]),
        lambda state : coarse_features(state, centers=centers, widths=1, offsets=[-0.3, 0.3]),
        lambda state : aggregation_features(state, centers=centers),
    ])):
    phi = get_phi(x[..., None])  # from (N,) to (N, S) with S = 1
    weights = np.zeros(phi.shape[-1])
    pbar = tqdm(total=max_iter)
    if name == "Poly":
        alpha = 0.000001
    else:
        alpha = 0.05
    for iter in range(max_iter):
        y_hat = phi @ weights
        errors = y - y_hat
        mse = np.mean(errors ** 2)
        gradient = (phi.T @ errors) / len(y)
        weights += alpha * gradient
        pbar.set_description(f"MSE: {mse}")
        pbar.update()
        if mse < thresh:
            break
    print(f"Iterations: {iter}, MSE: {mse}, N. of Features {len(weights)}")
    axs[(idx+1)//3][(idx+1)%3].plot(x, y_hat)
    axs[(idx+1)//3][(idx+1)%3].set_title(f"Approximation with {name} (MSE {mse:.3f})")
plt.show()

# Now repeat the experiment but fit the following function y.
# Submit your plots and discuss your results, paying attention to the
# non-smoothness of the new target function.
# - How did you change your hyperparameters? Did you use more/less wider/narrower features?
# - Consider the number of features. How would it change if your state would be 2-dimensional?
# Discuss each bullet point in at most two sentences.

x = np.linspace(-10, 10, 100)
y = np.zeros(x.shape)
y[0:10] = x[0:10]**3 / 3.0
y[10:20] = np.exp(x[25:35])
y[20:30] = -x[0:10]**3 / 2.0
y[30:60] = 100.0
y[60:70] = 0.0
y[70:100] = np.cos(x[70:100]) * 100.0

max_iter = 10000
thresh = 1e-8
centers = np.linspace(-10, 10, 30).reshape(-1, 1)
fig, axs = plt.subplots(2, 3)
axs[0][0].plot(x, y)
axs[0][0].set_title("True Function")

for idx, (name, get_phi) in enumerate(zip(["Poly", "RBFs", "Tiles", "Coarse", "Aggreg."],
    [
        lambda state : poly_features(state, degree=4),
        lambda state : rbf_features(state, centers=centers, sigmas=0.3),
        lambda state : tile_features(state, centers=centers, widths=1.2, offsets=[-0.5, 0, 0.5]),
        lambda state : coarse_features(state, centers=centers, widths=0.6, offsets=[-0.2, 0.2]),
        lambda state : aggregation_features(state, centers=centers),
    ])):
    phi = get_phi(x[..., None])
    weights = np.zeros(phi.shape[-1])
    pbar = tqdm(total=max_iter)
    if name == "Poly":
        alpha = 0.0000001
    else:
        alpha = 0.05
    for iter in range(max_iter):
        y_hat = phi @ weights
        errors = y - y_hat
        mse = np.mean(errors ** 2)
        gradient = (phi.T @ errors) / len(y)
        weights += alpha * gradient
        # alpha = max(alpha - 1/max_iter, 0.00000001)
        pbar.set_description(f"MSE: {mse}")
        pbar.update()
        if mse < thresh:
            break

    print(f"Iterations: {iter}, MSE: {mse}, N. of Features {len(weights)}")
    axs[(idx+1)//3][(idx+1)%3].plot(x, y_hat)
    axs[(idx+1)//3][(idx+1)%3].set_title(f"Approximation with {name} (MSE {mse:.3f})")
plt.show()


#################### PART 3
# Consider the Gridworld depicted below. The dataset below contains episodes
# collected using the optimal policy, and the heatmap below shows its V-function.
# - Consider the 5 FAs implemented above and discuss why each would be a
#   good/bad choice. Discuss each in at most two sentences.

# The given data is a dictionary of (s, a, r, s', term, Q) arrays.
# Unlike the previous assignments, the state is the (x, y) coordinate of the agent
# on the grid.
# - Run batch semi-gradient TD prediction with a FA of your choice (the one you
#   think would work best) to learn an approximation of the V-function.
#   Use gamma = 0.99. Increase the number of iterations, if you'd like.
#   Plot your result of the true V-function against your approximation using the
#   provided plotting function.

data = np.load("a6_gridworld.npz")
s = data["s"]
a = data["a"]
r = data["r"]
s_next = data["s_next"]
Q = data["Q"]
V = data["Q"].max(-1)  # value of the greedy policy
term = data["term"]
n = s.shape[0]
n_states = 81
n_actions = 5
gamma = 0.99

# needed for heatmaps
s_idx = np.ravel_multi_index(s.T, (9, 9))
unique_s_idx = np.unique(s_idx, return_index=True)[1]

fig, axs = plt.subplots(1, 1)
# surf = axs.tricontourf(s[:, 0], s[:, 1], V)
surf = axs.imshow(V[unique_s_idx].reshape(9, 9))
plt.colorbar(surf)
plt.show()

max_iter = 20000
alpha = 0.1
thresh = 1e-8

xx, yy = np.meshgrid(np.arange(0, 9, 1), np.arange(0, 9, 1))
centers = np.stack([xx, yy], axis=-1).reshape(-1, 2)
# Pick one
# name, get_phi = "Poly", lambda state : poly_features(state, degree=4)
name, get_phi = "RBFs", lambda state : rbf_features(state, centers=centers, sigmas=0.75)
# name, get_phi = "Tiles", lambda state : tile_features(state, centers=centers, widths=1)
# name, get_phi = "Coarse", lambda state : coarse_features(state, ...)
# name, get_phi = "Aggreg.", lambda state: aggregation_features(state, centers=centers)

phi = get_phi(s)
phi_next = get_phi(s_next)
weights = np.zeros(phi.shape[-1])
pbar = tqdm(total=max_iter)
for iter in range(max_iter):
    v_hat = phi @ weights
    v_hat_next = phi_next @ weights
    target = np.where(term, 0, gamma*v_hat_next)
    td_errors = r + target - v_hat
    # td_errors = r + v_hat_next - v_hat
    mse = np.mean((V - v_hat) ** 2)
    gradient = phi.T @ td_errors / len(V)
    weights += alpha * gradient
    alpha = max(alpha - 0.04/max_iter, 0.00001)
    pbar.set_description(f"MSE: {mse}")
    pbar.update()
    if mse < thresh:
        break
td_prediction = phi @ weights
print(f"Iterations: {iter}, MSE: {mse}, N. of Features {len(weights)}")
fig, axs = plt.subplots(1, 2)
axs[0].imshow(V[unique_s_idx].reshape(9, 9))
axs[1].imshow(td_prediction[unique_s_idx].reshape(9, 9))
axs[0].set_title("V")
axs[1].set_title(f"Approx. with RBF (MSE {mse:.7f})")
plt.show()


#################### PART 4
# - Run TD again, but this time learn an approximation of the Q-function.
#   How did you have to change your code?
# - You'll notice that the approximation you have learned seem very wrong. Why?
#   (hint: what is the given data missing? For example, is there any sample for
#   LEFT/RIGHT/UP/DOWN at goals?)
# - You may still be able to learn a Q-function approximation that makes the
#   agent act (almost) optimally. Beside your features and how they generalize,
#   what other hyperparameter is crucial in this scenario?
#
# Note: don't try to learn a Q-function that acts optimally, anything like the
# approximation in the screenshot below is fine.

# needed for heatmaps
s_idx = np.ravel_multi_index(s.T, (9, 9))
unique_s_idx = np.unique(s_idx, return_index=True)[1]

max_iter = 20000
alpha = 0.1
gamma = 0.99
thresh = 1e-8

xx, yy = np.meshgrid(np.arange(0, 9, 1), np.arange(0, 9, 1))
centers = np.stack([xx, yy], axis=-1).reshape(-1, 2)
# name, get_phi = "Tiles", lambda state : tile_features(state, centers=centers, widths=1, offsets=[-0.3, 0.3])
name, get_phi = "RBF", lambda state : rbf_features(state, centers=centers, sigmas=0.7)

phi = get_phi(s)
phi_next = get_phi(s_next)
weights = np.zeros((phi.shape[1], 5))
pbar = tqdm(total=max_iter)
for iter in range(max_iter):
    q_hat = phi @ weights
    value_greedy = np.max(q_hat, axis=1)
    value_greedy_next = np.roll(value_greedy, 1)
    target = np.where(term, 0, gamma * value_greedy_next)
    td_errors = r + target - value_greedy
    td_errors_fat = np.zeros((Q.shape[0], 5))
    td_errors_fat[np.arange(Q.shape[0]), a] = td_errors
    gradient = phi.T @ td_errors_fat / len(Q)
    weights += alpha * gradient
    mse = np.mean((Q - q_hat) ** 2)
    alpha = max(alpha - 0.05/max_iter, 0.00001)
    pbar.set_description(f"MSE: {mse}")
    pbar.update()
    if mse < thresh:
        break
td_prediction = phi @ weights
print(Q[unique_s_idx].argmax(-1).reshape(9, 9))  # check optimal policy
print(f"Iterations: {iter}, MSE: {mse}, N. of Features {len(weights)}")
fig, axs = plt.subplots(2, n_actions)
for i, j in zip(range(n_actions), ["LEFT", "DOWN", "RIGHT", "UP", "STAY"]):
    axs[0][i].imshow(Q[unique_s_idx, i].reshape(9, 9))
    axs[1][i].imshow(td_prediction[unique_s_idx, i].reshape(9, 9))
    axs[0][i].set_title(f"Q {j}")
    axs[1][i].set_title(f"RBF, Q {j} (MSE {mse:.2})")
plt.show()

#################### PART 5
# Discuss similarities and differences between SL regression and RL TD.
# - Discuss loss functions, techniques applicable to minimize it, and additional
#   challenges of RL.
# - What are the differences between "gradient descent" and "semi-gradient
#   descent" for TD?
# - Assume you'd have to learn the Q-function when actions are continuous.
#   How would you change your code?