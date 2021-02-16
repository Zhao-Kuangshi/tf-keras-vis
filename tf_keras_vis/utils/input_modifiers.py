from abc import ABC, abstractmethod

import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import random_rotation


class InputModifier(ABC):
    """Abstract class for defining an input modifier.
    """
    @abstractmethod
    def __call__(self, seed_input):
        """Implement modification to the input before processing gradient descent.

        # Arguments:
            seed_input: An N-dim numpy array.
        # Returns:
            The modified `seed_input`.
        """
        raise NotImplementedError()


class Jitter(InputModifier):
    def __init__(self, jitter=0.05):
        """Implements an input modifier that introduces random jitter.
            Jitter has been shown to produce crisper activation maximization images.

        # Arguments:
            jitter: The amount of jitter to apply, scalar or sequence. If a scalar, same jitter is
                applied to all image dims. If sequence, `jitter` should contain a value per image
                dim. A value between `[0., 1.]` is interpreted as a percentage of the image
                dimension. (Default value: 0.05)
        """
        self.jitter = None
        self._jitter = jitter

    def __call__(self, seed_input, data_format='channels_last'):
        if data_format == 'channels_last':
            if self.jitter is None:
                self.jitter = [
                    dim * self._jitter if self._jitter < 1. else self._jitter
                    for dim in seed_input.shape[1:-1]
                ]
            return tf.roll(seed_input, [np.random.randint(-j, j + 1) for j in self.jitter],
                           tuple(range(len(seed_input.shape))[1:-1]))
        elif data_format == 'channels_first':
            if self.jitter is None:
                self.jitter = [
                    dim * self._jitter if self._jitter < 1. else self._jitter
                    for dim in seed_input.shape[2:]
                ]
            return tf.roll(seed_input, [np.random.randint(-j, j + 1) for j in self.jitter],
                           tuple(range(len(seed_input.shape))[2:]))


class Rotate(InputModifier):
    def __init__(self, degree=1.):
        """Implements an input modifier that introduces random rotation.
            Rotate has been shown to produce crisper activation maximization images.

        # Arguments:
            degree: The amount of rotation to apply.
        """
        self.rg = degree

    def __call__(self, seed_input, data_format='channels_last'):
        if len(seed_input.shape) != 4:  # the seed_input is not 2D image
            logging.warning('The input modifier `Rotate` is only valid for 2D '
                            'images. You have seed_input with shape '
                            f'{tuple(seed_input.shape)}, this input will not '
                            'be modified.')
            return seed_input
        if data_format == 'channels_last':
            if tf.is_tensor(seed_input):
                seed_input = seed_input.numpy()
            seed_input = np.array([
                random_rotation(x, self.rg, row_axis=0, col_axis=1, channel_axis=2) for x in seed_input
            ])
            return tf.constant(seed_input)
        if data_format == 'channels_first':
            if tf.is_tensor(seed_input):
                seed_input = seed_input.numpy()
            seed_input = np.array([
                random_rotation(x, self.rg, row_axis=1, col_axis=2, channel_axis=0) for x in seed_input
            ])
            return tf.constant(seed_input)
        raise ValueError('Invalid data_format parameter.')
