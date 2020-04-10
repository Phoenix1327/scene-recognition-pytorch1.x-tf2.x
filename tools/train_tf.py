# This is modified from https://github.com/tensorflow/models/blob/master/official/vision/image_classification/classifier_trainer.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pprint
from typing import Any, Tuple, Text, Optional, Mapping

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from official.modeling.hyperparams import params_dict
from official.utils.logs import logger
from official.utils import hyperparams_flags
from official.vision.image_classification.configs import configs

def define_classifier_flags():
  """Defines common flags for image classification."""
  hyperparams_flags.initialize_common_flags()
  flags.DEFINE_string(
      'data_dir',
      default=None,
      help='The location of the input data.')
  flags.DEFINE_string(
      'mode',
      default=None,
      help='Mode to run: `train`, `eval`, `train_and_eval` or `export`.')
  flags.DEFINE_bool(
      'run_eagerly',
      default=None,
      help='Use eager execution and disable autograph for debugging.')
  flags.DEFINE_string(
      'model_type',
      default=None,
      help='The type of the model, e.g. EfficientNet, etc.')
  flags.DEFINE_string(
      'dataset',
      default=None,
      help='The name of the dataset, e.g. ImageNet, etc.')
  flags.DEFINE_integer(
      'log_steps',
      default=100,
      help='The interval of steps between logging of batch level stats.')

def _get_params_from_flags(flags_obj: flags.FlagValues):

    """Get ParamsDict from flags."""
    model = flags_obj.model_type.lower()
    dataset = flags_obj.dataset.lower()
    # TODO(zhz): Why is this a ParamsDict type
    params = configs.get_config(model=model, dataset=dataset)

    flags_overrides = {
        'model_dir': flags_obj.model_dir,
        'mode': flags_obj.mode,
        'model': {
            'name': model,
        },
        'runtime': {
            'run_eagerly': flags_obj.run_eagerly,
            'tpu': flags_obj.tpu,
        },
        'train_dataset': {
            'data_dir': flags_obj.data_dir,
        },
        'validation_dataset': {
            'data_dir': flags_obj.data_dir,
        },
        'train': {
            'time_history': {
                'log_steps': flags_obj.log_steps,
            },
        },
    }

    overriding_configs = (flags_obj.config_file,
                            flags_obj.params_override,
                            flags_overrides)

    pp = pprint.PrettyPrinter()

    logging.info('Base params: %s', pp.pformat(params.as_dict()))

    for param in overriding_configs:
        logging.info('Overriding params: %s', param)
        # Set is_strict to false because we can have dynamic dict parameters.
        params = params_dict.override_params_dict(params, param, is_strict=False)

    params.validate()
    params.lock()

    logging.info('Final model parameters: %s', pp.pformat(params.as_dict()))
    return params


def run(flags_obj: flags.FlagValues,
        strategy_override: tf.distribute.Strategy = None) -> Mapping[str, Any]:
    """Runs Image Classification model using native Keras APIs.
    Args:
        flags_obj: An object containing parsed flag values.
        strategy_override: A `tf.distribute.Strategy` object to use for model.
    Returns:
        Dictionary of training/eval stats
    """
    params = _get_params_from_flags(flags_obj)
    if params.mode == 'train_and_eval':
        return train_and_eval(params, strategy_override)
    elif params.mode == 'export_only':
        export(params)
    else:
        raise ValueError('{} is not a valid mode.'.format(params.mode))

def main(_):
    with logger.benchmark_context(flags.FLAGS):
        stats = run(flags.FLAGS)
    if stats:
        logging.info('Run stats:\n%s', stats)

if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    define_classifier_flags()
    flags.mark_flag_as_required('data_dir')
    flags.mark_flag_as_required('mode')
    flags.mark_flag_as_required('model_type')
    flags.mark_flag_as_required('dataset')

    app.run(main)