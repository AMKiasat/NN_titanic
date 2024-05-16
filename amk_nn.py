import numpy as np
import math
import pickle
import matplotlib.pyplot as plt


def sigmoid(x):
    """ It returns 1/(1+exp(-x)) where the values lie between '0' and '1' """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def linear(x):
    """ y = f(x) It returns the input as it is"""
    return x


def linear_derivative(x):
    x1 = np.empty_like(x)
    for k in range(len(x)):
        x1[k][0] = 1

    return x1


def af_derivative_cal(l, x, a):
    if a == 1:
        return l * sigmoid_derivative(x)
    elif a == 5:
        return l * linear_derivative(x)


def feed_forward(n, w, b, af):
    for k in range(len(n) - 1):
        temp = w[k].dot(n[k]) + b[k]

        if af == 1:
            neuro = sigmoid(temp)
        elif af == 5:
            neuro = linear(temp)
        if math.isnan(neuro[0][0]):
            return
        n[k + 1] = neuro


def back_propagate(n, w, b, af, label, lr, pl):

    loss1 = label - n[-1]
    loss_delta = np.absolute(pl) - np.absolute(loss1)
    if np.average(loss_delta) < 0:
        alr = 1.03
    elif np.average(loss_delta) > 0:
        alr = 0.5
    else:
        alr = 1

    delta = af_derivative_cal(loss1, n[-1], af)

    next_w = w[-1].copy()
    b[-1] += np.sum(delta, axis=0, keepdims=True) * lr * alr
    for k in range(len(w[-1].T)):
        w[-1].T[k] += (lr * alr * delta * n[-2][k]).flatten()

    for k in reversed(range(len(w) - 1)):
        loss = next_w.T.dot(delta)
        delta = af_derivative_cal(loss, n[k + 1], af)
        next_w = w[k].copy()
        b[k] += np.sum(delta, axis=0, keepdims=True) * lr * alr
        for j in range(len(w[k].T)):
            w[k].T[j] += (lr * alr * delta * n[k][j]).flatten()
    return loss1


def train(data, label, layer_num, hiddenL_neuron_num, activation_function=1, epoch=20, learning_rate=0.1):
    data_num, input_neuron_num = data.shape
    output_neuron_num = label.max() + 1
    label01 = []

    for i in label:
        tmp = []
        for j in range(output_neuron_num):
            if i == j:
                tmp.append([1])
            else:
                tmp.append([0])
        label01.append(tmp)

    """ Making random wights and biases """
    wi = []
    bi = []
    if layer_num > 2:
        for i in range(layer_num - 1):
            if i == 0:
                tmp = np.random.rand(hiddenL_neuron_num[0], input_neuron_num)
            elif i == layer_num - 2:
                tmp = np.random.rand(output_neuron_num, hiddenL_neuron_num[i - 1])
            else:
                tmp = np.random.rand(hiddenL_neuron_num[i], hiddenL_neuron_num[i - 1])
            wi.append(tmp)
            bias = np.random.rand(1, 1)
            bi.append(bias)
    else:
        tmp = np.random.rand(output_neuron_num, input_neuron_num)
        wi.append(tmp)
        bias = np.random.rand(1, 1)
        bi.append(bias)

    """ Training with epoch """
    count = 0
    for i in range(epoch):
        prev_loss = 0
        for j in range(len(data)):
            tmp = np.array(data[j])
            neurons = [tmp[:, np.newaxis]]
            for k in hiddenL_neuron_num:
                neurons.append(np.zeros(k).T)
            neurons.append(np.zeros(output_neuron_num).T)
            feed_forward(neurons, wi, bi, activation_function)
            prev_loss = back_propagate(neurons, wi, bi, activation_function, label01[j], learning_rate, prev_loss)
            # if j == 0:
            #     print(neurons[-1])
            #     print(label[j], "\n")
    with open('wights.pkl', 'wb') as file:
        pickle.dump(len(wi), file)
        for array in wi:
            np.save(file, array)
    with open('biases.pkl', 'wb') as file:
        pickle.dump(len(bi), file)
        for array in bi:
            np.save(file, array)
    with open('activation_function.txt', 'w') as file:
        file.write(str(activation_function))


def test(data):
    with open('wights.pkl', 'rb') as file:
        wights = pickle.load(file)
        wi = [np.load(file) for _ in range(wights)]
    with open('biases.pkl', 'rb') as file:
        biases = pickle.load(file)
        bi = [np.load(file) for _ in range(biases)]
    with open('activation_function.txt', 'r') as file:
        activation_function = int(file.read())

    output_label = []
    for j in range(len(data)):
        tmp = np.array(data[j])
        neurons = [tmp[:, np.newaxis]]
        for k in wi:
            # print(k.shape)
            neurons.append(np.zeros(k.shape[0]).T)
        feed_forward(neurons, wi, bi, activation_function)
        output_label.append(np.argmax(neurons[-1]))
    return output_label
