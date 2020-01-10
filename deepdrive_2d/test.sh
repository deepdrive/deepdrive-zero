#!/usr/bin/env bash

set -e  # Abort script at first error, when a command exits with non-zero status (except in until or while loops, if-tests, list constructs)
set -u  # Attempt to use undefined variable outputs error message, and forces an exit
set -o pipefail  # Causes a pipeline to return the exit status of the last command in the pipe that returned a non-zero return value.


python test.py


echo Running tests without Numba ---------------------------------------------
# Run without numba
NUMBA_DISABLE_JIT=1 python test.py