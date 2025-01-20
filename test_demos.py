#!/usr/bin/env python

# Insert mavenn at beginning of path
import sys
import os
import glob
path_to_mavenn_local = '.'
sys.path.insert(0, path_to_mavenn_local)

#Load mavenn and check path
import mavenn
print(mavenn.__path__)

# First run training demo
mavenn.run_demo("mpsa_ge_training")
exit()

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

# Clean up figure files
print('=' * 80)
print('Removing pic files...')
for pic_file in glob.glob('./*.png'):
    print(f'Removing {pic_file}...')
    os.remove(pic_file)
print(f'Done running {sys.argv[0]}')
