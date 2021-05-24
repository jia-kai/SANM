#!/bin/bash -e

# make the videos

cd $(dirname $0)

fea=$(readlink -f ../build/fea/fea)
confdir=$(readlink -f ../config)
render=$(readlink -f render_ondemand.sh)
rconfdir=$(readlink -f configs)

mkdir -p output_vid
cd output_vid

function run_single {
    mkdir -p $1
    cd $1
    fin=$2
    rconf=$3
    shift 3
    if [ ! -f fea_done ]; then
        $fea $confdir/{sys.json,$fin.json,override_save_interm.json} $@
        touch fea_done
    fi
    for i in *.obj.json; do
        objname=${i%.json}
        $render $rconf 50 100 $objname $objname.png
    done
    cd ..
}

# run model render_config
function run {
    run_single $1-sanm $@
    run_single $1-newton $@ $confdir/override_baseline_noproj.json
    run_single $1-levmar $@ $confdir/override_baseline_levmar.json
}

run armadillo_small $rconfdir/armadillo.json
