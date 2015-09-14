from __future__ import print_function
import numpy as np
import lasagne as lnn
import dmgr

from collections import namedtuple
from utils import Timer, Colors


NeuralNetwork = namedtuple('NeuralNetwork', 'network train test process')


def process_batches(batches, func, timer=None):
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


def train(network, train_set, n_epochs, batch_size,
          validation_set=None, early_stop=np.inf):

    best_val_loss = np.inf
    epochs_since_best_val_loss = 0

    best_params = lnn.layers.get_all_param_values(network.network)

    for epoch in range(n_epochs):
        timer = Timer()
        timer.start('epoch')
        timer.start('train')

        train_loss = process_batches(
            dmgr.iterators.iterate_batches(train_set, batch_size, shuffle=True),
            network.train,
            timer
        )

        timer.stop('train')

        if validation_set:
            batches = dmgr.iterators.iterate_batches(
                validation_set, batch_size, shuffle=False
            )
            val_loss = process_batches(batches, network.test)

        print('Ep. {}/{} {:.1f}s (tr: {:.1f}s th: {:.1f}s)'.format(
            epoch + 1, n_epochs,
            timer['epoch'], timer['train'], timer['theano']),
              end='')
        print('  tl: {:.6f}'.format(train_loss), end='')

        if validation_set:
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


def to_string(network):
    repr_str = ''

    for layer in lnn.layers.get_all_layers(network):
        if isinstance(layer, lnn.layers.DropoutLayer):
            repr_str += '\t -> dropout p = {:.1f}\n'.format(layer.p)
            continue

        repr_str += '\t{} - {}\n'.format(layer.output_shape, layer.name)

    # return everything except the last newline
    return repr_str[:-1]
