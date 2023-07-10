.. _installation:

Installation Instructions
=========================

using the ``pip`` package manager by executing the following at the
commandline: ::

    $ pip install mavenn
    $ pip install mavenn --upgrade

Please note that the latest version of MAVE-NN conflicts with NumPy's version (1.24.3)
due to a conflicting dependency with TensorFlow. This is the reason, 
some users may have to run a pip upgrade command (shown above) 
after installation on their command line.


Alternatively, you can clone MAVE-NN from
`GitHub <https://github.com/jbkinney/mavenn>`_ by doing
this at the command line: ::

    $ cd appropriate_directory
    $ git clone https://github.com/jbkinney/mavenn.git

where ``appropriate_directory`` is the absolute path to where you would like
MAVE-NN to reside. Then add this to the top of any Python file in
which you use MAVE-NN: ::

    # Insert local path to MAVE-NN at beginning of Python's path
    import sys
    sys.path.insert(0, 'appropriate_directory/mavenn')

    #Load mavenn
    import mavenn


