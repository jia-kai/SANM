#!/bin/bash -e

name=$1
mode=$2

if [ -z "$name" -o "(" "$mode" != "g" -a "$mode" != "d" ")" ]; then
    echo "run benchmark comparison with baseline methods"
    echo "usage: $0 <output name> <mode:g|d> <base configs ...>"
fi
shift 2

fea=$(readlink -f ../build/fea/fea)
confdir=$(readlink -f ../config)

configs[0]=$confdir/sys.json
i=0
while [ -n "$1" ]; do
    let i=$(($i+1))
    configs[$i]=$confdir/$1
    shift
done

function run_single {
    mkdir -p $1
    cd $1
    shift
    if [ ! -f fea_done ]; then
        echo "${configs[@]}" $@ > cmd
        $fea "${configs[@]}" $@ 2>&1 | tee log
        [ "${PIPESTATUS[0]}" == '0' ] || exit 1
        touch fea_done
    fi
    cd ..
}

mkdir -p output_cmp_with_baseline/$name
cd output_cmp_with_baseline/$name

for energy in arap neo_comp neo_incomp; do
    mkdir -p $energy
    cd $energy

    econf=$confdir/override_${energy}.json
    if [ "$mode" = "g" -a "$energy" = "arap" -a "$name" = "armadillo_small-g" ]; then
        econf="$econf $confdir/override_stiff_material.json"
    fi
    echo "$econf"
    run_single sanm $econf
    if [ "$mode" = "g" -o "$energy" = "arap" ]; then
        run_single baseline $econf $confdir/override_baseline.json
        run_single baseline_noproj $econf $confdir/override_baseline_noproj.json
        [ -n "$RUN_LEVMAR" -a "$mode" = "g" ] && \
            run_single baseline_levmar $econf $confdir/override_baseline_levmar.json
    fi
    run_single sanm_no_pade $econf $confdir/override_no_pade.json

    cd ..
done
