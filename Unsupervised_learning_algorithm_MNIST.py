# This code illustrates the fast AI implementation of the unsupervised \"biological\" learning algorithm from
# [Unsupervised Learning by Competing Hidden Units](https://doi.org/10.1073/pnas.1820458116)
# on MNIST data set.
# If you want to learn more about this work you can also check out this
# [lecture](https://www.youtube.com/watch?v=4lY-oAY0aQU)
# from MIT's [6.S191 course](http://introtodeeplearning.com/).

import scipy.io
import numpy as np
import matplotlib.pyplot as plt


def draw_weights(ep, fig, synapses, Kx, Ky):
    yy = 0
    HM = np.zeros((28 * Ky, 28 * Kx))
    for y in range(Ky):
        for x in range(Kx):
            HM[y * 28 : (y + 1) * 28, x * 28 : (x + 1) * 28] = synapses[yy, :].reshape(
                28, 28
            )
            yy += 1
    plt.clf()
    nc = np.amax(np.absolute(HM))

    im = plt.imshow(HM, cmap="bwr", vmin=-nc, vmax=nc)
    fig.colorbar(im, ticks=[np.amin(HM), 0, np.amax(HM)])
    plt.title(f"Weights epoch {ep}")
    plt.axis("off")
    fig.canvas.draw()
    plt.pause(0.001)


def train():
    mat = scipy.io.loadmat("mnist_all.mat")
    Nc = 10     # number of classes (0-9)
    IDIM = 784
    Ns = 60000
    M = np.zeros((0, IDIM))
    for i in range(Nc):
        M = np.concatenate((M, mat["train" + str(i)]), axis=0)
    M = M / 255.0

    eps0 = 2e-2  # learning rate
    Kx, Ky = 10, 10
    NHID = Kx * Ky  # number of hidden units that are displayed in Ky by Kx array
    MAX_EPOCHS = 200  # number of epochs
    BATCH_SIZE = 100  # size of the minibatch
    EPSILON = 1e-30  # numerical precision of weight updates
    delta = 0.4  # Strength of the anti-hebbian learning
    p = 2.0  # Lebesgue norm of the weights
    k = 2  # ranking parameter, must be integer that is bigger or equal than 2

    fig = plt.figure(figsize=(12.9, 10))

    # For every minibatch the overlap with the data `tot_input` is calculated for  each data point and
    # each hidden unit. The sorted strengths of the # activations are stored in `y`. The variable `yl`
    # stores the activations of the post synaptic cells - it is denoted by g(Q) in Eq 3 of the paper.
    # See also Eq 9 and Eq 10.
    # The variable `ds` is the right hand side of Eq 3. The weights are updated after each minibatch
    # in a way so that the largest update is equal to the learning rate `eps` at that epoch.
    mu, sigma = 0.0, 1.0  # mean and standard deviation - for weight initialisation
    synapses = np.random.normal(mu, sigma, (NHID, IDIM))
    for ep in range(MAX_EPOCHS):
        eps = eps0 * (1 - ep / MAX_EPOCHS)
        M = M[np.random.permutation(Ns), :]
        for i in range(Ns // BATCH_SIZE):
            inputs = np.transpose(M[i * BATCH_SIZE : (i + 1) * BATCH_SIZE, :])
            sig = np.sign(synapses)
            tot_input = np.dot(sig * np.absolute(synapses) ** (p - 1), inputs)

            y = np.argsort(tot_input, axis=0)  # 100x100
            yl = np.zeros((NHID, BATCH_SIZE))  # 100x100
            yl[y[NHID - 1, :], np.arange(BATCH_SIZE)] = 1.0
            yl[y[NHID - k], np.arange(BATCH_SIZE)] = -delta

            xx = np.sum(np.multiply(yl, tot_input), 1)  # 100x1
            # ds 100x784
            ds = np.dot(yl, np.transpose(inputs)) - np.multiply(
                np.tile(xx.reshape(xx.shape[0], 1), (1, IDIM)), synapses
            )

            nc = np.amax(np.absolute(ds))  # amax - maximum element (scalar)
            nc = max(nc, EPSILON)
            synapses += eps * np.true_divide(ds, nc)

        print(f"Epoch {ep}; nc {nc}")
        draw_weights(ep, fig, synapses, Kx, Ky)
    plt.show()  # pause until user kills the window


if __name__ == "__main__":
    train()
