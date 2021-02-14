from abc import ABC, abstractmethod

import tensorflow as tf

from tf_keras_vis.utils import listify, is_channels_first


class ModelVisualization(ABC):
    """Visualization class for Keras models.
    """
    def __init__(self, model, model_modifier=None, clone=True, data_format=None):
        """Create Visualization class instance that analize the model for debugging.

        # Arguments
            model: The `tf.keras.Model` instance. This model will be cloned by
                `tf.keras.models.clone_model` function and then will be modified by
                `model_modifier` according to need. Therefore the model will be NOT modified.
            model_modifier: A function that modify `model` instance. For example, in
                ActivationMaximization normally, this function is used to replace the softmax
                function that was applied to the model outputs.
            clone: A bool. If you won't model to be copied, you can set this option to False.
            data_format: A `str` instance among 'channels_first' or
                'channels_last', the default is None which means the data
                format of all the inputs is `channels_last`. If your model has
                more than one input, you can pass a list of data format. For
                example, `data_format=['channels_first', 'channels_last']`.
        """
        if clone:
            self.model = tf.keras.models.clone_model(model)
            self.model.set_weights(model.get_weights())
        else:
            self.model = model
        if model_modifier is not None:
            new_model = model_modifier(self.model)
            if new_model is not None:
                self.model = new_model
        # Determines the data format of the model
        if data_format is None and len(self.model.inputs) == 1:
            if is_channels_first(self.model):
                self.data_format = ['channels_first']
            else:
                self.data_format = ['channels_last']
        elif data_format is None:  # the default is 'channels_last'
            self.data_format = ['channels_last'] * len(self.model.inputs)
        else:
            data_format = listify(data_format)
            for i, df in enumerate(data_format):
                if df.lower() in ['channels_first', 'channel_first',
                                  'channels first', 'channel first']:
                    data_format[i] = 'channels_first'
                elif df.lower() in ['channels_last', 'channel_last',
                                    'channels last', 'channel last']:
                    data_format[i] = 'channels_last'
                else:
                    raise ValueError('the given parameter `data_format` has '
                                     f'invalid value `{df}` at position {i}.')

    @abstractmethod
    def __call__(self):
        """Analize the model.

        # Returns
            Results of analizing the model.
        """
        raise NotImplementedError()

    def _get_losses_for_multiple_outputs(self, loss):
        losses = listify(loss)
        if len(losses) == 1 and len(losses) < len(self.model.outputs):
            losses = losses * len(self.model.outputs)
        if len(losses) != len(self.model.outputs):
            raise ValueError(('The model has {} outputs, '
                              'but the number of loss-functions you passed is {}.').format(
                                  len(self.model.outputs), len(losses)))
        return losses

    def _get_seed_inputs_for_multiple_inputs(self, seed_input):
        seed_inputs = listify(seed_input)
        if len(seed_inputs) != len(self.model.inputs):
            raise ValueError(('The model has {} inputs, '
                              'but the number of seed-inputs tensors you passed is {}.').format(
                                  len(self.model.inputs), len(seed_inputs)))
        seed_inputs = (x if tf.is_tensor(x) else tf.constant(x) for x in seed_inputs)
        seed_inputs = (tf.expand_dims(x, axis=0) if len(x.shape) == len(tensor.shape[1:]) else x
                       for x, tensor in zip(seed_inputs, self.model.inputs))
        return list(seed_inputs)
