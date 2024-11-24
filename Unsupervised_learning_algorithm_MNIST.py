# This code illustrates the fast AI implementation of the unsupervised \"biological\" learning algorithm from
# [Unsupervised Learning by Competing Hidden Units](https://doi.org/10.1073/pnas.1820458116)
# on MNIST data set.
# If you want to learn more about this work you can also check out this
# [lecture](https://www.youtube.com/watch?v=4lY-oAY0aQU)
# from MIT's [6.S191 course](http://introtodeeplearning.com/).

import scipy.io
import numpy as np
import matplotlib.pyplot as plt


def draw_weights(ep, fig, synapses, GRID_X, GRID_Y):
    HM = np.zeros((28 * GRID_Y, 28 * GRID_X))
    for y in range(GRID_Y):
        for x in range(GRID_X):
            HM[y * 28 : (y + 1) * 28, x * 28 : (x + 1) * 28] = synapses[
                y * GRID_X + x, :
            ].reshape(28, 28)
    plt.clf()
    norm_const = np.amax(np.absolute(HM))

    im = plt.imshow(HM, cmap="bwr", vmin=-norm_const, vmax=norm_const)
    fig.colorbar(im, ticks=[np.amin(HM), 0, np.amax(HM)])
    plt.title(f"Weights epoch {ep}")
    plt.axis("off")
    fig.canvas.draw()
    plt.pause(0.001)


def load_mnist(file_path, normalize=True):
    try:
        mat = scipy.io.loadmat(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"{file_path} not found. Please provide the dataset.")

    data = np.vstack([mat[f"train{i}"] for i in range(10)])
    if normalize:
        data = data / 255.0  # Normalize pixel intensities to [0, 1]
    return data


def train():
    data = load_mnist("mnist_all.mat")
    IDIM = 784
    NTRAIN = 60000
    LR0 = 2e-2  # learning rate
    GRID_X, GRID_Y = 10, 10
    NHID = GRID_X * GRID_Y
    MAX_EPOCHS = 200  # number of epochs
    BATCH_SIZE = 100  # size of the minibatch
    EPSILON = 1e-30  # numerical precision of weight updates
    DELTA = 0.4  # Strength of the anti-hebbian learning
    P = 2.0  # Lebesgue norm of the weights
    RANK_PARAM = 2  #  must be integer that is bigger or equal than 2

    fig = plt.figure(figsize=(12.9, 10))

    # For every minibatch the overlap with the data `tot_input` is calculated for  each data point and
    # each hidden unit. The sorted strengths of the # activations are stored in `y`. The variable `yl`
    # stores the activations of the post synaptic cells - it is denoted by g(Q) in Eq 3 of the paper.
    # See also Eq 9 and Eq 10.
    # The variable `ds` is the right hand side of Eq 3. The weights are updated after each minibatch
    # in a way so that the largest update is equal to the learning rate at that epoch.
    mu, sigma = 0.0, 1.0  # mean and standard deviation - for weight initialisation
    synapses = np.random.normal(mu, sigma, (NHID, IDIM))
    for ep in range(MAX_EPOCHS):
        lr = LR0 * (1 - ep / MAX_EPOCHS)
        np.random.shuffle(data)
        for i in range(NTRAIN // BATCH_SIZE):
            inputs = np.transpose(data[i * BATCH_SIZE : (i + 1) * BATCH_SIZE, :])
            sig = np.sign(synapses)
            tot_input = np.dot(sig * np.absolute(synapses) ** (P - 1), inputs)

            y = np.argsort(tot_input, axis=0)  # 100x100
            yl = np.zeros((NHID, BATCH_SIZE))  # 100x100
            yl[y[NHID - 1, :], np.arange(BATCH_SIZE)] = 1.0
            yl[y[NHID - RANK_PARAM], np.arange(BATCH_SIZE)] = -DELTA

            xx = np.sum(np.multiply(yl, tot_input), 1)  # 100x1
            # delta_synapses 100x784
            delta_synapses = np.dot(yl, np.transpose(inputs)) - np.multiply(
                np.tile(xx.reshape(xx.shape[0], 1), (1, IDIM)), synapses
            )

            norm_const = np.amax(np.absolute(delta_synapses))
            norm_const = max(norm_const, EPSILON)
            synapses += lr * np.true_divide(delta_synapses, norm_const)

        print(f"Epoch {ep}; norm_const {norm_const}")
        draw_weights(ep, fig, synapses, GRID_X, GRID_Y)
    plt.show()  # pause until user kills the window


if __name__ == "__main__":
    train()
