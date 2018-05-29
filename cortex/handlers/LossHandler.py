import torch
from cortex.handlers import Handler, ModelHandler

MODEL_HANDLER = ModelHandler()

class LossHandler(Handler):
    '''
    Simple dict-like container for losses
    '''

    _type = torch.Tensor
    _get_error_string = 'Loss `{}` not found. You must add it as a dict entry'

    def check_key_value(self, k, v):
        super().check_key_value(k, v)
        if k not in MODEL_HANDLER:
            raise AttributeError(
                'Keyword `{}` not in the model_handler. Found: {}.'.format(
                    k, tuple(
                        MODEL_HANDLER.keys())))
        return True

    def __setitem__(self, k, v):
        passes = self.check_key_value(k, v)
        if len(v.size()) > 0:
            raise ValueError(
                'Loss size must be a scalar. Got {}'.format(
                    v.size()))
        if passes:
            super().__setitem__(k, v)
