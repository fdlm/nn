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
    Save the neural network's parameters (weights, biases, ...) to a
    pickle file.

    Parameters
    ----------
    network : lasagne neural network handle
        Neural Network of which the parameters should be saved
    filename : string
        Name of the file in which the parameters should be saved.
    """
    with open(filename, 'w') as f:
        pickle.dump(get_params(network), f, protocol=-1)


def load_params(network, filename):
    """
    Load a neural network's parameters from a pickle file.

    Parameters
    ----------
    network : lasagne neural network handle
        Neural network to load to parameters for
    filename : string
        Name of the file where parameters are stored
    """
    with open(filename, 'r') as f:
        params = pickle.load(f)
        set_params(network, params)


def to_string(network):
    """
    Create a string representation of a lasagne neural network

    Parameters
    ----------
    network : lasagne neural network handle
        Neural network to convert to a string

    Returns
    -------
    string
        string containing a human-readable description of the network's layers
    """
    repr_str = ''

    # this indicates if we need to skip the next layer. it is used when
    # printing batchnorm layers, since they are followed by a non-linearity
    # layer
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
                     mask_var=None, tags=None):
    """
    Compiles a Theano function that can be used to optise the parameters
    of a given neural network

    Parameters
    ----------
    network : lasagne neural network handle
        Neural network whose parameters are to be trained
    input_var :
    target_var :
    loss_fn :
    opt_fn :
    l1 :
    l2 :
    mask_var :
    tags :

    Returns
    -------

    """
    prediction = lnn.layers.get_output(network)
    tags = tags or {}

    # compute loss
    reg_tags = {'regularizable': True}
    reg_tags.update(tags)
    l1 = lnn.regularization.regularize_network_params(
        network, lnn.regularization.l1, tags=reg_tags) * l1
    l2 = lnn.regularization.regularize_network_params(
        network, lnn.regularization.l2, tags=reg_tags) * l2

    if mask_var:
        loss = loss_fn(prediction, target_var, mask_var) + l2 + l1
    else:
        loss = loss_fn(prediction, target_var) + l2 + l1

    # compile train function
    params = lnn.layers.get_all_params(network, trainable=True, **tags)
    updates = opt_fn(loss, params)

    if mask_var:
        train_fn = theano.function(
            [input_var, target_var, mask_var], loss, updates=updates)
    else:
        train_fn = theano.function(
            [input_var, target_var], loss, updates=updates)

    return train_fn


def compile_test_func(network, input_var, target_var, loss_fn, l2, l1,
                      mask_var=None, tags=None):

    reg_tags = {'regularizable': True}
    reg_tags.update(tags or {})

    prediction = lnn.layers.get_output(network, deterministic=True)
    l1 = lnn.regularization.regularize_network_params(
        network, lnn.regularization.l1, tags=reg_tags) * l1
    l2 = lnn.regularization.regularize_network_params(
        network, lnn.regularization.l2, tags=reg_tags) * l2

    if mask_var:
        loss = loss_fn(prediction, target_var, mask_var) + l2 + l1
        return theano.function(
            [input_var, target_var, mask_var], [loss, prediction])
    else:
        loss = loss_fn(prediction, target_var) + l2 + l1
        return theano.function([input_var, target_var], [loss, prediction])


def compile_process_func(network, input_var, mask_var=None):
    # create process function. process just computes the prediction
    # without computing the loss, and thus does not need target labels

    if isinstance(network, lnn.layers.Layer):
        prediction = lnn.layers.get_output(network, deterministic=True)
    else:
        prediction = network

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
        t = batch[1].reshape(-1, pred.shape[-1])

        correct_predictions = acc_func(p, t)
        if len(batch) < 3:  # no mask!
            total_correct += correct_predictions.mean()
            total_weight += 1
        else:  # we have a mask!
            m = batch[-1].flatten()
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


def train(network, num_epochs, train_fn, train_batches, test_fn=None,
          validation_batches=None, threads=None, early_stop=np.inf,
          early_stop_acc=False, save_epoch_params=False, callbacks=None,
          acc_func=onehot_acc, train_acc=False):
    """
    Train a neural network by updating its parameters.

    Parameters
    ----------
    network : lasagne neural network handle
        Network to be trained.
    num_epochs: int
        Maximum number of epochs to train
    train_fn : theano function
        Function that computes the loss and updates the network parameters.
        Takes parameters from the batch iterators
    train_batches : batch iterator
        Iterator that yields mini batches from the training set. Must be able
        to re-iterate multiple times.
    test_fn : theano function
        Function that computes loss and predictions of the network.
        Takes parameters from the batch iterators.
    validation_batches : batch iterator
        Iterator that yields mini batches from the validation set. Must be able
        to re-iterate multiple times.
    threads : int
        Number of threads to use to prepare mini batches. If None, use
        a single thread.
    early_stop : int
        Number of iterations without loss improvement on validation set that
        stops training.
    early_stop_acc : boolean
        Use validation accuracy instead of loss for early stopping.
    save_epoch_params : str or False
        Save neural network parameters after each epoch. If False, do not save.
        If you want to save the parameters, provide a filename with an
        int formatter so the epoch number can be inserted.
    callbacks : list of callables
        List of callables to call after each training epoch. Can be used to k
        update learn rates or plot data. Functions have to accept the
        following parameters: current epoch number, lists of per-epoch train
        losses, train accuracies, validation losses, validation accuracies.
        The last three lists may be empty, depending on other parameters.
    acc_func : callable
        Function to use to compute accuracies.
    train_acc : boolean
        Also compute accuracy for training set. In this case, the training
        loss will be also re-computed after an epoch, which leads to lower
        train losses than when not using this parameter.

    Returns
    -------
    tuple of four lists
        Train losses, trian accuracies, validation losses,
        validation accuracies for each epoch
    """

    if (test_fn is not None) != (validation_batches is not None):
        raise ValueError('If test function is given, validation set is '
                         'necessary (and vice-versa)!')

    best_val = np.inf if not early_stop_acc else 0.0
    epochs_since_best_val_loss = 0

    if callbacks is None:
        callbacks = []

    if callbacks is None:
        callbacks = []

    best_params = get_params(network)
    train_losses = []
    val_losses = []
    val_accs = []
    train_accs = []

    if threads is not None:
        def threaded(it):
            return dmgr.iterators.threaded(it, threads)
    else:
        def threaded(it):
            return it

    for epoch in range(num_epochs):
        timer = Timer()
        timer.start('epoch')
        timer.start('train')

        try:
            train_losses.append(
                avg_batch_loss(threaded(train_batches), train_fn, timer))
        except RuntimeError as e:
            print(Colors.red('Error during training:'), file=sys.stderr)
            print(Colors.red(str(e)), file=sys.stderr)
            return best_params

        timer.stop('train')

        if save_epoch_params:
            save_params(network, save_epoch_params.format(epoch))

        if validation_batches:
            val_loss, val_acc = avg_batch_loss_acc(
                threaded(validation_batches), test_fn, acc_func)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

        if train_acc:
            train_loss, tr_acc = avg_batch_loss_acc(
                threaded(train_batches), test_fn, acc_func)
            train_losses[-1] = train_loss
            train_accs.append(tr_acc)

        print('Ep. {}/{} {:.1f}s (tr: {:.1f}s th: {:.1f}s)'.format(
            epoch + 1, num_epochs,
            timer['epoch'], timer['train'], timer['theano']),
            end='')
        print('  tl: {:.6f}'.format(train_losses[-1]), end='')

        if train_acc:
            print('  tacc: {:.6f}'.format(tr_acc), end='')

        if validation_batches:
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

        for cb in callbacks:
            cb(epoch, train_losses, val_losses, train_accs, val_accs)

    # set the best parameters found
    set_params(network, best_params)
    return train_losses, val_losses, train_accs, val_accs


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

    def __call__(self, epoch, *args, **kwargs):
        if (epoch + 1) % self.interval == 0:
            self.learn_rate.set_value(np.float32(
                self.learn_rate.get_value() * self.factor)
            )
