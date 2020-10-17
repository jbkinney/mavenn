#! /usr/bin/env bash

# Remove current tutorials in distribution directory
rm -r mavenn/examples/tutorials

# Mirror tutorials
cp -r docs/tutorials mavenn/examples/.
