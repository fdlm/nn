from __future__ import print_function
import sys
import cPickle as pickle
import numpy as np
import lasagne as lnn
import dmgr
from madmom.evaluation import SimpleEvaluation

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

    def get_parameters(self):
        """
        Get the neural network's parameters (weights, biases, ...)
        :return: parameterse
        """
        return lnn.layers.get_all_param_values(self.network)

    def set_parameters(self, parameters):
        """
        Sets the neural network's parameters (weights, biases, ...)
        :param parameters: parameters to be set
        """
        lnn.layers.set_all_param_values(self.network, parameters)

    def save_parameters(self, filename):
        """
        Saves the neural network's parameters (weights, biases, ...) to a file.
        :param filename: file to store the parameters to
        """
        with open(filename, 'w') as f:
            pickle.dump(self.get_parameters(), f, protocol=-1)

    def load_parameters(self, filename):
        """
        Loads the neural network's parameters from a file.
        :param filename: file to load the parameters from
        """
        with open(filename, 'r') as f:
            params = pickle.load(f)
        self.set_parameters(params)

    def __str__(self):
        """
        Writes the layers of a neural network to a string
        :return: string containing a human-readable description of the
                 network's layers
        """
        repr_str = ''

        skip_next = False

        for layer in lnn.layers.get_all_layers(self.network):
            if skip_next:
                skip_next = False
                continue

            if isinstance(layer, lnn.layers.DropoutLayer):
                repr_str += '\t -> dropout p = {:g}\n'.format(layer.p)
                continue
            if isinstance(layer, lnn.layers.BatchNormLayer):
                repr_str += '\t -> batch norm\n'
                # skip next layer, which is the nonlinearity of the original
                # layer
                skip_next = True
                continue

            repr_str += '\t{} - {}\n'.format(layer.output_shape, layer.name)

        # return everything except the last newline
        return repr_str[:-1]


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

    for batch in batches:
        if timer:
            timer.start('theano')

        total_loss += func(*batch)
        n_batches += 1

        if np.isnan(total_loss):
            raise RuntimeError('NaN loss!')

        if timer:
            timer.pause('theano')
            print('Training... {:.2f}s  tl: {:.3f}'.format(
                timer['train'], total_loss / n_batches), end='\r')

    if timer:
        timer.stop('theano')

    return total_loss / n_batches


def avg_batch_loss_acc(batches, func):
    total_loss = 0.
    total_correct = 0.
    total_weight = 0.
    n_batches = 0

    for batch in batches:
        loss, pred = func(*batch)
        n_batches += 1
        total_loss += loss
        p = pred.reshape(-1, pred.shape[-1])  # flatten predictions and gt
        t = batch[-1].reshape(-1, pred.shape[-1])

        correct_predictions = p.argmax(1) == t.argmax(1)
        if len(batch) < 3:  # no mask!
            total_correct += correct_predictions.mean()
            total_weight += 1
        else:  # we have a mask!
            m = batch[1].flatten()
            total_correct += (correct_predictions * m).mean()
            total_weight += m.mean()

    return total_loss / n_batches, total_correct / total_weight


def predict(network, dataset, batch_size,
            batch_iterator=dmgr.iterators.iterate_batches, **kwargs):
    """
    Processes the dataset and return predictions for each instance.
    """

    predictions = []

    batches = batch_iterator(
        dataset, batch_size, shuffle=False, expand=False, **kwargs
    )

    for batch in batches:
        # skip the targets (last element)
        predictions.append(network.process(*(batch[:-1])))

    return np.vstack(predictions)


def evaluate(network, dataset, batch_size, eval_func,
             batch_iterator=dmgr.iterators.iterate_batches,
             **kwargs):
    """
    Processes the dataset and evaluates the results, assuming the targets
    are in one-hot notation
    :param network:        neural network used for prediction
    :param dataset:        dataset to process
    :param batch_size:     processing batch size
    :param eval_func:      function computing tp, fp, tn, fn given predictions
                           and targets
    :param batch_iterator: which iterator to use
    :param kwargs:         additional parameters for the batch_iterator
    :return:               madmom SimpleEvaluation
    """

    batches = batch_iterator(
        dataset, batch_size, shuffle=False, expand=False, **kwargs
    )

    total_eval = SimpleEvaluation()

    for batch in batches:
        # skip the targets (last element)
        pred = network.process(*(batch[:-1]))
        total_eval += SimpleEvaluation(*eval_func(pred, batch[-1]))

    return total_eval


def predict_rnn(network, dataset, batch_size,
                batch_iterator=dmgr.iterators.iterate_datasources, **kwargs):
    """
    Processes the dataset and return predictions for each instance.
    """

    predictions = []

    batches = batch_iterator(
        dataset, batch_size, shuffle=False, expand=False, **kwargs
    )

    for batch in batches:
        # skip the targets (last element)
        p = network.process(*(batch[:-1]))
        mask = batch[-2]
        predictions.append(p[mask.astype(bool)])

    return np.vstack(predictions)


def train(network, train_set, n_epochs, batch_size,
          validation_set=None, early_stop=np.inf, early_stop_acc=False,
          threaded=None, batch_iterator=dmgr.iterators.iterate_batches,
          save_params=False, updates=None, **kwargs):
    """
    Trains a neural network.
    :param network:        NeuralNetwork object.
    :param train_set:      dataset to use for training (see dmgr.datasources)
    :param n_epochs:       maximum number of epochs to train
    :param batch_size:     batch size for training
    :param validation_set: dataset to use for validation (see dmgr.datasources)
    :param early_stop:     number of iterations without loss improvement on
                           validation set that stops training
    :param early_stop_acc: sets if early stopping should be based on the loss
                           or the accuracy on the training set
    :param threaded:       number of batches to prepare in a separate thread
                           if 'None', do not use threading
    :param batch_iterator: batch iterator to use
    :param save_params:    save neural network parameters after each epoch. If
                           False, do not save. Provide a filename with an int
                           formatter so the epoch number can be inserted if you
                           want to save the parameters.
    :param updates:        List of functions to call after each epoch. Can
                           be used to update learn rates, for example.
                           unctions have to accept one parameter, which is the
                           epoch number
    :param kwargs:         parameters to pass to the batch_iterator
    :return:               best found parameters. if validation set is given,
                           the parameters that have the smallest loss on the
                           validation set. if no validation set is given,
                           parameters after the last epoch
    """

    best_val = np.inf if not early_stop_acc else 0.0
    epochs_since_best_val_loss = 0

    if updates is None:
        updates = []

    best_params = network.get_parameters()
    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(n_epochs):
        timer = Timer()
        timer.start('epoch')
        timer.start('train')

        train_batches = batch_iterator(
            train_set, batch_size, shuffle=True, **kwargs)

        if threaded:
            train_batches = dmgr.iterators.threaded(train_batches, threaded)

        try:
            train_losses.append(
                avg_batch_loss(train_batches, network.train, timer)
            )
        except RuntimeError as e:
            print(Colors.red('Error during training:'), file=sys.stderr)
            print(Colors.red(str(e)), file=sys.stderr)
            return best_params

        timer.stop('train')

        if save_params:
            network.save_parameters(save_params.format(epoch))

        if validation_set:
            batches = batch_iterator(
                validation_set, batch_size, shuffle=False, **kwargs
            )
            if threaded:
                batches = dmgr.iterators.threaded(batches, threaded)
            val_loss, val_acc = avg_batch_loss_acc(batches, network.test)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

        print('Ep. {}/{} {:.1f}s (tr: {:.1f}s th: {:.1f}s)'.format(
            epoch + 1, n_epochs,
            timer['epoch'], timer['train'], timer['theano']),
            end='')
        print('  tl: {:.6f}'.format(train_losses[-1]), end='')

        if validation_set:
            # early stopping
            cmp_val = val_losses[-1] if not early_stop_acc else -val_accs[-1]
            if cmp_val < best_val:
                epochs_since_best_val_loss = 0
                best_val = cmp_val
                best_params = network.get_parameters()
                # green output
                c = Colors.green
            else:
                epochs_since_best_val_loss += 1
                # neutral output
                c = lambda x: x

            print(c('  vl: {:.6f}'.format(val_losses[-1])), end='')
            print(c('  vacc: {:.6f}'.format(val_accs[-1])), end='')

            if epochs_since_best_val_loss >= early_stop:
                print(Colors.yellow('\nEARLY STOPPING!'))
                break
        else:
            best_params = network.get_parameters()

        print('')

        for upd in updates:
            upd(epoch)

    return best_params, train_losses, val_losses
