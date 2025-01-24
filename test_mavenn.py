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

    # Run tests
    mavenn.run_tests()

if __name__ == '__main__':
    main()