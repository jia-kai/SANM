#!/bin/bash -e

f0=$1
f1=$2
fout=$3

if [ -z "$fout" ]; then
    echo "usage: $0 <input0> <input1> <output>"
    echo "  vertically remove white edges and then merge the images"
    exit 1
fi

IFS=" x+" read w0 h0 x0 y0 < <(convert $f0 -format "%@\n" info:)
IFS=" x+" read w1 h1 x1 y1 < <(convert $f1 -format "%@\n" info:)

montage -geometry +0+10 -tile 1x2 \
    "$f0[x$h0+0+$y0]" "$f1[x$h1+1+$y1]" $fout
