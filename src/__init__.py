from contextlib import contextmanager
import sys, os

class SortSeqError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def simple_decorator(decorator):
    '''This decorator can be used to turn simple functions
    into well-behaved decorators, so long as the decorators
    are fairly simple. If a decorator expects a function and
    returns a function (no descriptors), and if it doesn't
    modify function attributes or docstring, then it is
    eligible to use this. Simply apply @simple_decorator to
    your decorator and it will automatically preserve the
    docstring and function attributes of functions to which
    it is applied.'''
    def new_decorator(f):
        g = decorator(f)
        g.__name__ = f.__name__
        g.__doc__ = f.__doc__
        g.__dict__.update(f.__dict__)
        return g
    # Now a few lines needed to make simple_decorator itself
    # be a well-behaved decorator.
    new_decorator.__name__ = decorator.__name__
    new_decorator.__doc__ = decorator.__doc__
    new_decorator.__dict__.update(decorator.__dict__)
    return new_decorator

@simple_decorator
def shutthefuckup(func):
    """
    Silences the standard output from any decroated function
    """

    # Define the wrapper function
    def func_wrapper(*arg,**kwargs):
        stdobak = sys.stdout
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            try:
                return func(*arg,**kwargs)
            finally:
                sys.stdout = stdobak
    
    # Return the wrapper function
    return func_wrapper
    
