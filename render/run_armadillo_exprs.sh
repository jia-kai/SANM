#!/bin/bash -e

exe="$1"

if [ -z "$exe" ]; then
    echo "run large scale experiments (with the armadillo model)"
    echo "output will be written to current dir"
    echo "usage: $0 <executable>"
    exit 1
fi

exe=$(readlink -f $exe)

conf=$(readlink -f $(dirname $0))/../config
sys=$conf/sys-mt32.json

mkdir -p default
cd default
[ -f done ] || $exe $sys $conf/armadillo.json &> log
touch done
cd ..

mkdir -p interm
cd interm
[ -f done ] || $exe $sys $conf/armadillo.json \
    $conf/override_save_interm.json &> log
touch done
cd ..

for i in 1 2 4 8 16 32; do
    mkdir -p mt$i
    cd mt$i
    [ -f done ] || $exe $conf/sys-mt$i.json $conf/armadillo.json &> log
    touch done
    cd ..
done

mkdir -p neo_incomp
cd  neo_incomp
[ -f done ] || $exe $sys $conf/armadillo.json \
    $conf/override_neo_incomp.json &> log
touch done
cd ..

mkdir -p inverse
cd inverse
[ -f done ] || $exe $sys $conf/armadillo.json \
    $conf/override_inverse.json &> log
touch done
cd ..

mkdir -p inverse_neo_incomp
cd inverse_neo_incomp
[ -f done ] || $exe $sys $conf/armadillo.json \
    $conf/override_neo_incomp.json $conf/override_inverse.json &> log
touch done
cd ..
