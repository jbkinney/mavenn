#!/usr/bin/env python
# Tested 25.01.22 by JBK

# Insert mavenn at beginning of path
import os
import sys
abs_path_to_mavenn = os.path.dirname(os.path.abspath(__file__)) + './'
sys.path.insert(0, abs_path_to_mavenn)

#Load mavenn and check path
import mavenn
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
