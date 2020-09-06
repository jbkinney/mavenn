# Standard imports
import re
import glob

# MAVE-NN imports
import mavenn
from mavenn.src.error_handling import handle_errors, check


@handle_errors
def demo(name=None, print_code=True):

    """
    Performs a demonstration of the mavenn software.

    parameters
    -----------
    name: (str, None)
        Name of demo to run. If None, a list of valid demo names
        will be printed.

    print_code: (bool)
        If True, text of the demo file will be printed along with
        the output of the demo file. If False, only the demo output
        will be shown.

    returns
    -------
    None

    """

    demos_dir = mavenn.__path__[0] +'/examples/demos'
    demo_file_names = glob.glob(f'{demos_dir}/*.py')
    demos_dict = {}
    for file_name in demo_file_names:
        base_name = file_name.split('/')[-1]
        pattern = '^(.*)\.py$'
        key = re.match(pattern, base_name).group(1)
        demos_dict[key] = file_name
    demo_names = list(demos_dict.keys())
    demo_names.sort()

    # If no input, list valid names
    if name is None:
        print("Please enter a demo name. Valid are:")
        print("\n".join([f'"{name}"' for name in demo_names]))

    # If input is valid, run demo
    elif name in demo_names:

        # Get demo file name
        file_name = demos_dict[name]

        # Print code if requested
        if print_code:
            with open(file_name, 'r') as f:
                content = f.read()
                line = '-------------------------------------------------------'
                print('Running %s:\n%s\n%s\n%s' % \
                      (file_name, line, content, line))
        else:
            pass

        # Run demo
        exec(open(file_name).read())

    # If input is invalid, raise error
    else:
        # raise error
        check(False,
              f'name = {name} is invalid. Must be one of {demo_names}')

