#!/usr/bin/env python
# 2025.01.22 Works - JBK

# Insert mavenn at beginning of path
import os
import sys
abs_path_to_mavenn = os.path.dirname(os.path.abspath(__file__)) + './'
sys.path.insert(0, abs_path_to_mavenn)

#Load mavenn and verify path is local
import mavenn
print('Using mavenn located at', mavenn.__path__[0])

def main():
    # Check path to mavenn
    print(mavenn.__path__)

    # Get list of valid names
    demo_names = mavenn.run_demo(print_names=False)

    # Get list of names entered by user
    args = sys.argv
    if len(args) == 1:
        print(f'Running all demos: {demo_names}...')
        user_names = demo_names
    else:
        user_names = sys.argv[1:]

    # Run demos one-by-one
    for name in user_names:
        print('=' * 80)
        if name not in demo_names:
            print(f'Unable to run {repr(name)}; invalid name.'
                f' Valid names are {demo_names}')
        else:
            print(f'Running demo {repr(name)}')
            mavenn.run_demo(name)

    print(f'Done running {sys.argv[0]}')

if __name__ == '__main__':
    main()