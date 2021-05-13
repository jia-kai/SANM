#!/bin/bash -e
for outname in $@; do :; done
[ -f $outname ] && exit 0
./render.sh "$@"
mv ${outname}0001.png ${outname}
