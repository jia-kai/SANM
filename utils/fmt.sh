#!/bin/bash -e

set -u -e

clang-format $1 | sponge $1
