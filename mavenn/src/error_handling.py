"""error_handling.py: Utilities for handling expected errors."""
from functools import wraps


class MavennError(Exception):
    """
    Represents an anticipated error in MAVE-NN execution.

    Parameters
    ----------
    message: (str)
        The message passed to check(). This only gets passed to the
        MavennError constructor when the condition passed to check() is
        False.
    """

    def __init__(self, message):
        """Construct class instance."""
        self.message = message

    def __str__(self):
        """Return string representation of error."""
        return self.message


class DebugResult:
    """Container class used for testing."""

    def __init__(self, result, mistake):
        """Construct class instance."""
        self.result = result
        self.mistake = mistake


def check(condition, message):
    """
    Check whether a condition is satisfied; if not, throw MavennError.

    Parameters
    ----------
    condition: (bool)
        A condition that, if false, halts mavenn execution and raises a
        clean error to user

    message: (str)
        The string to show user if condition is False.

    Returns
    -------
    None
    """
    if not condition:
        raise MavennError(message)


def handle_errors(func):
    """
    Handle anticipated errors in a function.

    This decorator allows the user to pass the keyword argument
    'should_fail' to any wrapped function.

    If should_fail is None (or is not set by user), the function executes
    normally, and can be called as

        result = func(*args, **kwargs)

    In particular, Python execution will halt if any errors are raised.

    However, if the user specifies should_fail=True or should_fail=False, then
    Python will not halt even in the presence of an error. Moreover, the
    function will return a tuple, e.g.,

        result, mistake = func(*args, should_fail=True, **kwargs)

    with mistake flagging whether or not the function failed or succeeded
    as expected.
    """
    @wraps(func)  # So wrapped_func has the same docstring, etc., as func
    def wrapped_func(*args, **kwargs):

        # Get should_fail debug flag AND remove should_fail from kwargs dict.
        # Need to use pop method so that 'should_fail' variable is not passed
        # to func() within try/except below
        should_fail = kwargs.pop('should_fail', None)

        # Otherwise, user passed something other than a bool for
        # should_fail, which isn't valid.
        check(should_fail in (True, False, None),
              'FATAL: should_fail = %s is not bool or None' % should_fail)

        # Default values for returned variables
        result = None
        mistake = None

        try:
            # Execute function
            result = func(*args, **kwargs)

            # If running functional test and expect to fail
            if should_fail is True:
                print('UNEXPECTED SUCCESS.')
                mistake = True

            # If running functional test and expect to pass
            elif should_fail is False:
                print('Expected success.')
                mistake = False

            # If user did not pass should_fail then nothing more to do.
            else:
                pass

        except MavennError as e:

            # If running functional test and expect to fail
            if should_fail is True:
                print('Expected error:',
                      e.__str__())
                mistake = False

            # If running functional test and expect to pass
            elif should_fail is False:
                print('UNEXPECTED ERROR:',
                      e.__str__())
                mistake = True

            # Otherwise, raise error
            else:
                raise e

        # If not in debug mode, and no error was detected above,
        if should_fail is None:

            # Just return result
            return result

        # Otherwise, if in debug mode
        else:

            # If func is a constructor,
            # extract self from args[0],
            # and set mistake attribute
            if func.__name__ == "__init__":
                args[0].mistake = mistake
                return None

            # Otherwise, return result and mistake status as attributes
            # of container class object
            else:
                return DebugResult(result, mistake)

    # Return the wrapped function to the user
    return wrapped_func
