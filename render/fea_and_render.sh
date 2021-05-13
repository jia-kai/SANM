#!/bin/bash -e

dst=$1
rconf=$2
obj=$3
shift 3

[ -f $dst ] && exit 0

confdir=$(readlink -f ../config)
fea=$(readlink -f ../build/fea/fea)
render=$(readlink -f render.sh)
replace_vtx=$(readlink -f ../utils/replace_vtx_coord.py)
rconf=$(readlink -f configs/$rconf)
renormal=$(readlink -f $(dirname $0)/renormal.py)

configs[0]=$confdir/sys.json
i=0
while [ -n "$1" ]; do
    let i=$(($i+1))
    configs[$i]=$confdir/$1
    shift
done

mkdir -p $dst-work
cd $dst-work
[ -f fea_done ] || $fea "${configs[@]}"
touch fea_done
if [[ "$obj" == *.vtx  ]]; then
    $replace_vtx $confdir/model/${obj%.vtx}.1.obj $obj $obj-repl.obj
    obj=$obj-repl.obj
fi
if grep smooth $rconf; then
    $renormal $obj $obj-renormal.obj
    obj=$obj-renormal.obj
fi
$render $rconf 100 100 $obj out
mv *.png ../$(basename $dst)
