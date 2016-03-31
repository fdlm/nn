from __future__ import print_function

try:
    import cPickle as pickle
except:
    import pickle

import sys

import numpy as np
import theano

import dmgr
import lasagne as lnn
from utils import Timer, Colors

# simple aliases
get_params = lnn.layers.get_all_param_values
set_params = lnn.layers.set_all_param_values


def save_params(network, filename):
    """
    Saves the neural network's parameters (weights, biases, ...) to a file.
    :param network:  lasagne neural network
    :param filename: file to store the parameters to
    """
    with open(filename, 'w') as f:
        pickle.dump(get_params(network), f, protocol=-1)


def load_params(network, filename):
    """
    Loads the neural network's parameters from a file.
    :param network:  lasagne neural network
    :param filename: file to load the parameters from
    """
    with open(filename, 'r') as f:
        params = pickle.load(f)
        set_params(network, params)


def to_string(network):
    """
    Writes the layers of a neural network to a string
    :return: string containing a human-readable description of the
             network's layers
    """
    repr_str = ''

    skip_next = False

    for layer in lnn.layers.get_all_layers(network):
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


def compile_train_fn(network, input_var, target_var, loss_fn, opt_fn, l1, l2,
                     mask_var=None):
    # create train function
    prediction = lnn.layers.get_output(network)

    # compute loss
    l1 = lnn.regularization.regularize_network_params(
        network, lnn.regularization.l1) * l1
    l2 = lnn.regularization.regularize_network_params(
        network, lnn.regularization.l2) * l2

    if mask_var:
        loss = loss_fn(prediction, target_var, mask_var) + l2 + l1
    else:
        loss = loss_fn(prediction, target_var) + l2 + l1

    # compile train function
    params = lnn.layers.get_all_params(network, trainable=True)
    updates = opt_fn(loss, params)

    if mask_var:
        train_fn = theano.function(
            [input_var, mask_var, target_var], loss, updates=updates)
    else:
        train_fn = theano.function(
            [input_var, target_var], loss, updates=updates)

    return train_fn


def compile_test_func(network, input_var, target_var, loss_fn, l2, l1,
                      mask_var=None):
    prediction = lnn.layers.get_output(network, deterministic=True)
    l1 = lnn.regularization.regularize_network_params(
        network, lnn.regularization.l1) * l1
    l2 = lnn.regularization.regularize_network_params(
        network, lnn.regularization.l2) * l2

    if mask_var:
        loss = loss_fn(prediction, target_var, mask_var) + l2 + l1
        return theano.function(
            [input_var, mask_var, target_var], [loss, prediction])
    else:
        loss = loss_fn(prediction, target_var) + l2 + l1
        return theano.function([input_var, target_var], [loss, prediction])


def compile_process_func(network, input_var, mask_var=None):
    # create process function. process just computes the prediction
    # without computing the loss, and thus does not need target labels
    prediction = lnn.layers.get_output(network, deterministic=True)
    if mask_var:
        return theano.function([input_var, mask_var], prediction)
    else:
        return theano.function([input_var], prediction)


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


def onehot_acc(pred, targ):
    return pred.argmax(1) == targ.argmax(1)


def elemwise_acc(pred, targ, thresh=0.5):
    return ((pred > thresh) == (targ > thresh)).all(axis=1)


def avg_batch_loss_acc(batches, func, acc_func=onehot_acc):
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

        correct_predictions = acc_func(p, t)
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


def train(network, train_fn, train_set, num_epochs, batch_size,
          test_fn=None, validation_set=None, early_stop=np.inf,
          early_stop_acc=False, batch_iterator=dmgr.iterators.iterate_batches,
          threaded=None, save_epoch_params=False, updates=None,
          acc_func=onehot_acc, **kwargs):
    """
    Trains a neural network.
    :param network:        lasagne neural network
    :param train_fn:       theano function that updates the network parameters
    :param train_set:      dataset to use for training (see dmgr.datasources)
    :param num_epochs:       maximum number of epochs to train
    :param batch_size:     batch size for training
    :param test_fn:        theano function that computes the loss.
    :param validation_set: dataset to use for validation (see dmgr.datasources)
    :param early_stop:     number of iterations without loss improvement on
                           validation set that stops training
    :param early_stop_acc: sets if early stopping should be based on the loss
                           or the accuracy on the training set
    :param batch_iterator: batch iterator to use
    :param threaded:       number of batches to prepare in a separate thread
                           if 'None', do not use threading
    :param save_epoch_params: save neural network parameters after each epoch.
                           If False, do not save. Provide a filename with an
                           int formatter so the epoch number can be inserted
                           if you want to save the parameters.
    :param updates:        List of functions to call after each epoch. Can
                           be used to update learn rates, for example.
                           unctions have to accept one parameter, which is the
                           epoch number
    :param acc_func:       which function to use to compute validation accuracy
    :param kwargs:         parameters to pass to the batch_iterator
    :return:               best found parameters. if validation set is given,
                           the parameters that have the smallest loss on the
                           validation set. if no validation set is given,
                           parameters after the last epoch
    """

    if bool(test_fn) != bool(validation_set):
        raise ValueError('If test function is given, validation set is '
                         'necessary (and vice-versa)!')

    best_val = np.inf if not early_stop_acc else 0.0
    epochs_since_best_val_loss = 0

    if updates is None:
        updates = []

    best_params = get_params(network)
    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(num_epochs):
        timer = Timer()
        timer.start('epoch')
        timer.start('train')

        train_batches = batch_iterator(
            train_set, batch_size, shuffle=True, **kwargs)

        if threaded:
            train_batches = dmgr.iterators.threaded(train_batches, threaded)

        try:
            train_losses.append(avg_batch_loss(train_batches, train_fn, timer))
        except RuntimeError as e:
            print(Colors.red('Error during training:'), file=sys.stderr)
            print(Colors.red(str(e)), file=sys.stderr)
            return best_params

        timer.stop('train')

        if save_epoch_params:
            save_params(network, save_epoch_params.format(epoch))

        if validation_set:
            batches = batch_iterator(
                validation_set, batch_size, shuffle=False, **kwargs
            )
            if threaded:
                batches = dmgr.iterators.threaded(batches, threaded)
            val_loss, val_acc = avg_batch_loss_acc(batches, test_fn, acc_func)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

        print('Ep. {}/{} {:.1f}s (tr: {:.1f}s th: {:.1f}s)'.format(
            epoch + 1, num_epochs,
            timer['epoch'], timer['train'], timer['theano']),
            end='')
        print('  tl: {:.6f}'.format(train_losses[-1]), end='')

        if validation_set:
            # early stopping
            cmp_val = val_losses[-1] if not early_stop_acc else -val_accs[-1]
            if cmp_val < best_val:
                epochs_since_best_val_loss = 0
                best_val = cmp_val
                best_params = get_params(network)
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
            best_params = get_params(network)

        print('')

        for upd in updates:
            upd(epoch)

    # set the best parameters found
    set_params(network, best_params)
    return train_losses, val_losses, val_accs


class LearnRateSchedule:

    def __init__(self, learning_rate, interval, factor):
        """
        Learn rate schedule
        :param learning_rate:  shared variable containing the learn rate
        :param interval:    after how many epochs to change the learn rate
        :param factor:      by which factor to change the learn rate
        """
        self.interval = interval
        self.factor = factor
        self.learn_rate = learning_rate

    def __call__(self, epoch):
        if (epoch + 1) % self.interval == 0:
            self.learn_rate.set_value(np.float32(
                self.learn_rate.get_value() * self.factor)
            )
