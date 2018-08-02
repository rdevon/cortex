"""This is the module docstring."""

import inspect

def f(x):
    """This is the function docstring."""
    return 2 * x

# f.__doc__ = 'None'
# print(f.__doc__)
# print(type(f.__doc__))
# lines = 'None'.expandtabs().split('\n')
# print(lines)
NoneType = type(None)
x = None
type(x) == NoneType
isinstance(x, NoneType)
True
doc = inspect.cleandoc('')
print(doc)