#!/bin/bash -e

# plot the output files generated by running the testcase

gnuplot -p -e 'plot
    "polar_decomp_approx_logerr.txt" using 1:2 with lines,
    "polar_decomp_approx_logerr.txt" using 1:3 with lines;
    '
gnuplot -p -e 'plot
    "log_det_approx_logerr.txt" using 1:2 with lines,
    "log_det_approx_logerr.txt" using 1:3 with lines;
    '
