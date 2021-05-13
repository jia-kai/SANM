# SANM: A Symbolic Asymptotic Numerical Solver

This repository is the official implementation of the SANM solver to appear at
SIGGRAPH 2021.

SANM is a framework that automates and generalizes the Asymptotic Numerical
Methods (ANM) to solve symbolically represented nonlinear systems via numerical
continuation and **higher-order approximations**(unlike Newtonian methods that
essentially use first or second order approximations). In a nutshell, SANM
automatically extends a parameterized curve in a high-dimensional space from a
given starting point, where the curve is implicitly defined by a symbolically
represented function. We are typically interested in solving the endpoint of the
curve at a specific parameter value. SANM can be thousands of times faster than
Levenberg-Marquardt in solving the endpoint.

Please read our paper for more technical details.

This repository contains the [SANM library](libsanm) and its [application](fea)
to mesh deformation problems.

## Building

SANM needs a recent compiler that supports C++20. We have only tested SANM on
64-bit Linux systems. We use CMake to build SANM.

The only external dependency is Intel MKL (recently renamed to Intel oneAPI Math
Kernel Library (oneMKL)). Other dependencies are included as submodules.

Commands for a fresh build:

```sh
git submodule update --init --recursive
mkdir build
cd build
cmake .. -DMKLROOT=/path/to/intel/mkl # or -DONEAPIROOT=/path/to/intel/oneapi
make
```

Execute `./tests/sanm_tests` to run the test cases.

## Usage

Use `./fea/fea` to run the mesh deformation applications. This program needs
some JSON configurations to specify the functionality.

For example, to computes the deformed Bob model under gravity with SANM:

```sh
./fea/fea ../config/sys.json ../config/bob.json`
```

It generates an output `bob-i0-neohookean_i.obj` that can be visualized by
MeshLab. To solve the same problem with Newton's energy minimization, run:

```sh
./fea/fea ../config/sys.json ../config/bob.json ../config/override_baseline_noproj.json
```
### Reproducing the paper results

The [render](render) directory contains the tools for reproducing the results
reported in the paper.

```sh
cd render

# Step 1: Run parallel comparison
# Run these commands on a machine with at least 32 CPU cores.
mkdir output_parallel
cd output_parallel
../run_armadillo_exprs.sh
../run_cmp_chen2014.sh

# Step 2: Generate results for comparing with Newton's methods
# Change the parallelism (-j4 and -j6) according to hardware configuration
make -f Makefile.cmp_with_baseline -j4
# run the LevMar method, which is too slow so we use more parallel jobs
RUN_LEVMAR=1 make -f Makefile.cmp_with_baseline  -j6

# Step 3: Render the images
# Run these commands on a machine with a GPU to speedup rendering
# Blender is required
pip3 install pymeshlab
make -f Makefile.render

# Step 4: Generate tables and plots
# Dependencies: matplotlib pandas seaborn scipy
./gen_table_figs.py

# The outputs are placed in the output directory
ls output
```

# Citation

If SANM is helpful to your research, please cite us as

```text
@article{jia2021sanm,
  title={{SANM}: A Symbolic Asymptotic Numerical Solver with Applications in
      Mesh Deformation},
  author={Jia, Kai},
  journal={ACM Transactions on Graphics},
  year={2021},
  publisher={ACM}
}
```
