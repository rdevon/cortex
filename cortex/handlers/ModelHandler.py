import logging
from torch import nn
from cortex.handlers import Handler

LOGGER = logging.getLogger('cortex.data_setup')

class ModelHandler(Handler):
    """
    Simple dict-like container for nn.Module's
    """

    _type = nn.Module
    _get_error_string = 'Model `{}` not found. You must add it in `build_models` (as a dict entry). Found: {}'
    _special = Handler()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #super().unsafe_set('special', Handler())

    def check_key_value(self, k, v):
        if k in self:
            LOGGER.warning(
                'Key {} already in MODEL_HANDLER, ignoring.'.format(k))
            return False

        if k in self._protected:
            raise KeyError('Keyword `{}` is protected.'.format(k))

        if isinstance(v, (list, tuple)):
            for v_ in v:
                self.check_key_value(k, v_)
        elif self._type and not isinstance(v, self._type):
            raise ValueError(
                'Type `{}` of `{}` not allowed. Only `{}` and subclasses (or tuples of {}) are supported'.format(
                    type(v), k, self._type, self._type))

        return True

    def get_special(self, key):
        return self._special[key]

    def add_special(self, **kwargs):
        self._special.update(**kwargs)
