class Handler(dict):
    '''
    Simple dict-like container with support for `.` access
    Note: some of the functionalty might not work correctly as a dict, but so far simple tests pass.
    '''

    __delattr__ = dict.__delitem__
    _protected = dir({})
    _type = None
    _get_error_string = 'Keyword `{}` not found (add as a dict entry). Found: {}'


    def check_key_value(self, k, v):
        if k in self._protected:
            raise KeyError('Keyword `{}` is protected.'.format(k))

        if self._type and not isinstance(v, self._type):
            raise ValueError('Type `{}` of `{}` not allowed. Only `{}` and subclasses are supported'.format(
                type(v), k, self._type))

        return True

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.check_key_value(k, v)
        super().__init__(**kwargs)

    def __setitem__(self, k, v):
        passes = self.check_key_value(k, v)
        if passes:
            super().__setitem__(k, v)

    def unsafe_set(self, k, v):
        super().__setitem__(k, v)

    def __setattr__(self, k, v):
        self.__setitem__(k, v)

    def __getattr__(self, k):
        if k.startswith('__'):
            return super.get(k)
        try:
            v = super().__getitem__(k)
        except KeyError:
            raise KeyError(self._get_error_string.format(k, tuple(self.keys())))
        return v

    def update(self, **kwargs):
        _kwargs = Handler()
        for k, v in kwargs.items():
            passes = self.check_key_value(k, v)
            if passes:
                _kwargs[k] = v
        super().update(**_kwargs)