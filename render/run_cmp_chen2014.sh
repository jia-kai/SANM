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
sys=$conf/sys.json

mkdir -p chen-fwdbar
cd chen-fwdbar
[ -f done ] || $exe $sys $conf/bar.json &> log
touch done
cd ..

mkdir -p chen-invbar
cd chen-invbar
[ -f done ] || $exe $sys $conf/bar.json $conf/override_inverse.json &> log
touch done
cd ..

mkdir -p chen-fwdplant
cd chen-fwdplant
[ -f done ] || $exe $sys $conf/plant.json &> log
touch done
cd ..

mkdir -p chen-invplant
cd chen-invplant
[ -f done ] || $exe $sys $conf/plant.json $conf/override_inverse.json &> log
touch done
cd ..

sys=$conf/sys-mt4.json
mkdir -p chen-fwdbar-mt4
cd chen-fwdbar-mt4
[ -f done ] || $exe $sys $conf/bar.json &> log
touch done
cd ..

mkdir -p chen-invbar-mt4
cd chen-invbar-mt4
[ -f done ] || $exe $sys $conf/bar.json $conf/override_inverse.json &> log
touch done
cd ..

mkdir -p chen-fwdplant-mt4
cd chen-fwdplant-mt4
[ -f done ] || $exe $sys $conf/plant.json &> log
touch done
cd ..

mkdir -p chen-invplant-mt4
cd chen-invplant-mt4
[ -f done ] || $exe $sys $conf/plant.json $conf/override_inverse.json &> log
touch done
cd ..
