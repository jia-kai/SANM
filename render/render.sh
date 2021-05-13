#!/bin/bash -e

[ -z "$NO_BG" ] && bg=--background
exec blender $bg \
    --python $(dirname $(readlink -f $0))/mesh_visual.py \
    --render-frame 1 -- \
    "$@"
