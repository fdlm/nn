from __future__ import print_function
import numpy as np
import lasagne as lnn
import dmgr

from utils import Timer, Colors


class NeuralNetwork(object):
    """
    Neural Network. Simple class that holds theano functions for training,
    testing, and processing, plus the lasagne output layer.
    """

    def __init__(self, network, train, test, process):
        """
        Initialises the neural network class.

        :param network: lasagne neural network output layer
        :param train:   theano function for training
        :param test:    theano function for testing
        :param process: theano function to process data, without computing loss
        """
        self.network = network
        self.train = train
        self.test = test
        self.process = process

    def __str__(self):
        """
        Writes the layers of a neural network to a string
        :return: string containing a human-readable description of the
                 network's layers
        """
        repr_str = ''

        for layer in lnn.layers.get_all_layers(self.network):
            if isinstance(layer, lnn.layers.DropoutLayer):
                repr_str += '\t -> dropout p = {:.1f}\n'.format(layer.p)
                continue

            repr_str += '\t{} - {}\n'.format(layer.output_shape, layer.name)

        # return everything except the last newline
        return repr_str[:-1]


def process_batches(batches, func, timer=None):
    """
    Processes batches and collects the predictions for each data point
    :param batches: batch generator. yields inputs and targets
    :param func:    theano function computing the predictions
    :param timer:   utils.Timer object for function timing. if None, there
                    will be no timing
    :return:        numpy array containing the predictions (one per line)
    """

    predictions = []

    for inputs, targets in batches:
        if timer:
            timer.start('theano')

        predictions.append(func(inputs))

        if timer:
            timer.pause('theano')

    return np.vstack(predictions)


def avg_batch_loss(batches, func, timer=None):
    """
    Processes batches, calculates the average loss and computes
    the time needed to compute the function, without getting the data.
    :param batches: batch generator. yields inputs and targets.
    :param func:    theano function to apply to the batch
    :param timer:   utils.Timer object for function timing. if None, there
                    will be no timing.
    :return:        average loss
    """
    total_loss = 0.
    n_batches = 0

    for inputs, targets in batches:
        if timer:
            timer.start('theano')

        total_loss += func(inputs, targets)
        n_batches += 1

        if timer:
            timer.pause('theano')

    if timer:
        timer.stop('theano')

    return total_loss / n_batches


def predict(network, dataset, batch_size):
    return process_batches(
        dmgr.iterators.iterate_batches(dataset, batch_size, shuffle=False,
                                       expand=False),
        network.process
    )


def train(network, train_set, n_epochs, batch_size,
          validation_set=None, early_stop=np.inf, threaded=None):
    """
    Trains a neural network.
    :param network:        NeuralNetwork object.
    :param train_set:      dataset to use for training (see dmgr.datasources)
    :param n_epochs:       maximum number of epochs to train
    :param batch_size:     batch size for training
    :param validation_set: dataset to use for validation (see dmgr.datasources)
    :param early_stop:     number of iterations without loss improvement on
                           validation set that stops training
    :param threaded:       number of batches to prepare in a separate thread
                           if 'None', do not use threading
    :return:               best found parameters. if validation set is given,
                           the parameters that have the smallest loss on the
                           validation set. if no validation set is given,
                           parameters after the last epoch
    """

    best_val_loss = np.inf
    epochs_since_best_val_loss = 0

    best_params = lnn.layers.get_all_param_values(network.network)

    for epoch in range(n_epochs):
        timer = Timer()
        timer.start('epoch')
        timer.start('train')

        train_batches = dmgr.iterators.iterate_batches(
            train_set, batch_size, shuffle=True)

        if threaded:
            train_batches = dmgr.iterators.threaded(train_batches, threaded)

        train_loss = avg_batch_loss(
            train_batches,
            network.train,
            timer
        )

        timer.stop('train')

        if validation_set:
            batches = dmgr.iterators.iterate_batches(
                validation_set, batch_size, shuffle=False
            )
            val_loss = avg_batch_loss(batches, network.test)

        print('Ep. {}/{} {:.1f}s (tr: {:.1f}s th: {:.1f}s)'.format(
            epoch + 1, n_epochs,
            timer['epoch'], timer['train'], timer['theano']),
              end='')
        print('  tl: {:.6f}'.format(train_loss), end='')

        if validation_set:
            # early stopping
            if val_loss < best_val_loss:
                epochs_since_best_val_loss = 0
                best_val_loss = val_loss
                best_params = lnn.layers.get_all_param_values(network.network)
                # green output
                c = Colors.green
            else:
                epochs_since_best_val_loss += 1
                # neutral output
                c = lambda x: x

            print(c('  vl: {:.6f}'.format(val_loss)), end='')

            if epochs_since_best_val_loss >= early_stop:
                print(Colors.yellow('\nEARLY STOPPING!'))
                break
        else:
            best_params = lnn.layers.get_all_param_values(network.network)

        print('')

    return best_params

