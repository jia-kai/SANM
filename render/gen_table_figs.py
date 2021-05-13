#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# so X11 would not be needed
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
import scipy.stats
from scipy.stats.mstats import gmean

from pathlib import Path
import collections
import json
import glob

g_tot_time_sanm = 0
g_tot_time_other = 0
g_all_speedup = []
g_gravity_speedup = []
g_deform_speedup = []

def speedup(sanm, other, is_deform):
    global g_tot_time_sanm
    global g_tot_time_other
    g_tot_time_sanm += sanm
    g_tot_time_other += other
    s = other / sanm
    g_all_speedup.append(s)
    if is_deform:
        g_deform_speedup.append(s)
    else:
        g_gravity_speedup.append(s)
    return s

def mean_confidence_interval(data, confidence=0.95):
    a = np.ascontiguousarray(data, dtype=np.float64)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def set_plot_style():
    # see http://www.jesshamrick.com/2016/04/13/reproducible-plots/
    plt.style.use(['seaborn-white', 'seaborn-paper'])
    matplotlib.rcParams.update({
        'font.size': 14,
        'legend.fontsize': 'medium',
        'axes.labelsize': 'large',
        'axes.titlesize': 'medium',
        'xtick.labelsize': 'large',
        'ytick.labelsize': 'large',
    })

def scalability():
    threads = np.array([1, 2, 4, 8, 16, 32], dtype=np.float64)
    times = np.empty_like(threads)
    for i, j in enumerate(threads):
        data = read_only_json_in_dir(f'output_parallel/mt{int(j)}')
        times[i] = data['time_solve']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xticks([1, 8, 16, 32])
    ax.plot(threads, times, '-o', label='SANM Solving Time')
    idealx = np.linspace(threads[0], threads[-1], 100)
    ax.plot(idealx, times[0] / idealx, '--', label='Ideal Parallelism')
    ax.set_ylim((0, np.max(times) * 1.1))
    ax.grid(axis='y')
    ax.set_xlabel('# threads')
    ax.set_ylabel('Time (seconds)')
    ax.legend(loc='best', fancybox=True, framealpha=0.9, borderpad=1,
              frameon=True)
    # ax.set_title('Comparing End-to-end Solving Time against Number of Threads')

    fig.tight_layout()
    fig.savefig('output/scalability.pdf', metadata={'CreationDate': None})

def read_only_json_in_dir(dname, check_inv=False):
    f = glob.glob(f'{dname}/*.json')
    assert len(f) == 1, f'json files in {dname}: {f}'
    f, = f
    with open(f) as fin:
        ret = json.load(fin)
        if check_inv:
            assert ret['nr_inverted'] == 0, f'inverted in {f}'
        return ret

def ftime(x):
    return f'{x:.2f}'

def frms(x):
    return f'{x:.1e}'.replace('e-0', 'e-')

def fsanm_iter(s):
    df = s['iter_deform']
    rf = s['iter_refine']
    return f'{df+rf}({rf})'

MESHES = ['armadillo_small', 'bar', 'bifur3', 'bob', 'human', 'plant']
MESHES_DISP = {'armadillo_small': 'armadillo-s'}

ENERGIES = ['arap', 'neo_comp', 'neo_incomp']
ENERGIES_DISP = {'arap': 'ARAP',
                 'neo_comp': 'NC',
                 'neo_incomp': 'NI'}

class BoldMin:
    def __init__(self, v, disp):
        self.v = float(v)
        if isinstance(disp, collections.Callable):
            disp = disp(v)
        assert isinstance(disp, str)
        self.disp = disp

    @classmethod
    def apply(cls, cols):
        vals = []
        for idx, v in enumerate(cols):
            if isinstance(v, cls):
                vals.append((v.v, idx, v.disp))
                cols[idx] = v.disp
        vals.sort()
        for m, sty in zip(vals[:2], ['bf', 'it']):
            cols[m[1]] = r'\text%s{%s}' % (sty, m[2])

def gen_table_gravity(fout=None):
    if fout is None:
        with open('output/gravity.tex', 'w') as fout:
            return gen_table_gravity(fout)

    for mesh in MESHES:
        for energy in ENERGIES:
            cols = []
            basedir = Path(f'output_cmp_with_baseline/{mesh}-g/{energy}')
            sanm = read_only_json_in_dir(basedir / 'sanm', True)
            if energy is ENERGIES[0]:
                title = ((r'\multirow{3}{*}{\parbox{5em}{'
                          r'%s \\'
                          r'{\tiny V=%d F=%d}}}') %
                         (MESHES_DISP.get(mesh, mesh), sanm['mesh_V'],
                          sanm['mesh_F']))
            else:
                title = ''
            cols.extend([
                title,
                ENERGIES_DISP[energy],
                sanm['iter'],
                BoldMin(sanm['time_solve'], ftime),
                frms(sanm['force_rms_recomp']),
            ])

            min_other = float('inf')
            for b in ['baseline_noproj', 'baseline', 'baseline_levmar']:
                bv = read_only_json_in_dir(basedir / b, 'levmar' not in b)
                if 'levmar' in b:
                    iters = bv['iter_tot']
                else:
                    iters = f'{bv["iter_tot"]}({bv["iter_refine"]})'
                cols.extend([
                    iters,
                    BoldMin(bv['time'], ftime),
                    frms(bv['force_rms_recomp']),
                ])
                if bv['nr_inverted']:
                    cols[-2] = cols[-2].disp
                    cols[-1] += r'\tnote{*}'
                else:
                    min_other = min(min_other, bv['time'])

            cols.append('{:.2f}'.format(speedup(sanm['time_solve'], min_other,
                                                False)))
            BoldMin.apply(cols)
            fout.write(' & '.join(map(str, cols)))
            fout.write(r' \\')
            fout.write('\n')

def gen_table_deform(fout=None):
    if fout is None:
        with open('output/deform.tex', 'w') as fout:
            return gen_table_deform(fout)

    for mesh in MESHES:
        if mesh == 'bar':
            mesh = 'bar2'
        cols = []
        basedir = Path(f'output_cmp_with_baseline/{mesh}-d/{ENERGIES[0]}')
        sanm = read_only_json_in_dir(basedir / 'sanm', True)
        cols.extend([
            MESHES_DISP.get(mesh, mesh),
            fsanm_iter(sanm),
            BoldMin(sanm['time'], ftime),
            frms(sanm['force_rms_recomp']),
        ])

        min_other = float('inf')
        for b in ['baseline_noproj', 'baseline']:
            bv = read_only_json_in_dir(basedir / b, False)
            cols.extend([
                f'{bv["iter_tot"]}({bv["iter_refine"]})',
                BoldMin(bv['time'], ftime),
                frms(bv['force_rms_recomp']),
            ])
            if bv['nr_inverted']:
                cols[-2] = cols[-2].disp
                cols[-1] += r'\tnote{*}'
            else:
                min_other = min(min_other, bv['time'])

        cols.append('{:.2f}'.format(speedup(sanm['time'], min_other, True)))

        for e in ENERGIES[1:]:
            basedir = Path(f'output_cmp_with_baseline/{mesh}-d/{e}')
            s1 = read_only_json_in_dir(basedir / 'sanm', True)
            cols.extend([
                fsanm_iter(s1),
                ftime(s1['time']),
                frms(s1['force_rms_recomp']),
            ])

        BoldMin.apply(cols)
        fout.write(' & '.join(map(str, cols)))
        fout.write(r' \\')
        fout.write('\n')

def gen_table_chen_cmp():
    def get(fwd, model, mt):
        mt = ['', '-mt4'][mt]
        j = read_only_json_in_dir(f'output_parallel/chen-{fwd}{model}{mt}')
        return j['iter'], '{:.2f}'.format(j['time_solve'])

    chen_data = [
        [2, 2.38],
        [3, 7.07],
        [3, 3.25],
        [4, 9.27],
    ]

    with open('output/chen-cmp.tex', 'w') as fout:
        for fwd in ['inv', 'fwd']:
            for model in ['bar', 'plant']:
                cols = [f'{fwd}. {model}']
                i0, t0 = get(fwd, model, 0)
                i1, t1 = get(fwd, model, 1)
                assert i0 == i1
                cd = chen_data[int(fwd == 'fwd') * 2 + int(model == 'plant')]
                cols.extend([i0, t0, t1])
                cols.extend(cd)
                fout.write(' & '.join(map(str, cols)))
                fout.write(r' \\')
                fout.write('\n')

def plot_armadillo_g():
    img = mpimg.imread('output/armadillo-g-fwd.png')
    width = img.shape[1]
    fig, ax = plt.subplots(figsize=(10, 1.7))
    ax.imshow(img, interpolation='none')
    n = 11
    ax.set_xlim((0, width + n))
    ax.set_yticks([])
    ax.set_xticks((np.arange(n) + 0.5) / n * width)
    ax.set_xticklabels(list(map('{:.1f}'.format, np.arange(n) / (n - 1))))
    ax.set_xlabel(r'$\lambda$')
    fig.tight_layout()
    fig.savefig('output/armadillo-g-fwd.pdf', metadata={'CreationDate': None})

def gen_table_aramadillo_cmp():
    def ftime(x):
        return f'{x:.2f}'

    def frms(x):
        return f'{x:.1e}'.replace('e-0', 'e-').replace('e+0', 'e+')

    def gen(fout, dpath, name):
        json = read_only_json_in_dir(f'output_parallel/{dpath}')
        rms = None
        with open(f'output_parallel/{dpath}/log') as fin:
            for i in fin.readlines():
                if 'iter=' in i:
                    assert rms is None
                    comp = i.split()
                    for j in range(len(comp)):
                        if comp[j].startswith('iter='):
                            assert j
                            rms = float(comp[j - 1])
                            break
        assert rms is not None

        cols = [name, json['iter'], ftime(json['time_solve']), frms(rms)]
        fout.write(' & '.join(map(str, cols)))
        fout.write(r' \\')
        fout.write('\n')

    with open('output/armadillo-cmp.tex', 'w') as fout:
        gen(fout, 'default', 'fwd. NC')
        gen(fout, 'inverse', 'inv. NC')
        gen(fout, 'neo_incomp', 'fwd. NI')
        gen(fout, 'inverse_neo_incomp', 'inv. NI')

def plot_plant_gd():
    img = mpimg.imread('output/plant-gd.png')
    height, width = img.shape[:2]
    fig, ax = plt.subplots(figsize=(8, 1.7))
    ax.imshow(img, interpolation='none')
    n = 7
    ax.set_xlim((0, width + n))
    ax.set_yticks([])
    ax.set_xticks((np.arange(n) + 0.5) / n * width)
    cx = 1.76 / n * width
    cy = 0.11 * height
    cr = 0.1 * height
    circ = plt.Circle((cx, cy), cr, color='r', fill=False)
    plt.text(cx + cr * 1.08, cy, 'control\nhandles', size=cr / 4,
             ha='left', va='top')
    ax.add_patch(circ)
    labels = ['rest', 'gravity\nequilibrium']
    for i in range(5):
        labels.append(rf'$\lambda={(i+1)*0.2:.1f}$')
    ax.set_xticklabels(labels)

    fig.tight_layout()
    fig.savefig('output/plant-gd.pdf', metadata={'CreationDate': None})

def get_sparse_solver_time(expr):
    ret = 0
    nr = 0
    with open(Path(expr) / 'log') as fin:
        for line in fin:
            p = line.split()
            if p[0] in ['sparse_prep', 'sparse_solve']:
                assert p[2].startswith('tot=')
                ret += float(p[2].split('=')[1])
                nr += 1
    assert nr == 2
    return ret

def numdefs():
    pade_saved_iters = []
    sparse_solver_times = []
    for mode in 'gd':
        keyi = 'iter_tot' if mode == 'd' else 'iter'
        keyt = 'time' if mode == 'd' else 'time_solve'
        for mesh in MESHES:
            for energy in ENERGIES:
                if mode == 'd' and mesh == 'bar':
                    mesh = 'bar2'
                basedir = Path(
                    f'output_cmp_with_baseline/{mesh}-{mode}/{energy}')
                s0p = basedir / 'sanm'
                s0 = read_only_json_in_dir(s0p, True)
                s1 = read_only_json_in_dir(basedir / 'sanm_no_pade', True)
                assert s0['pade'] and not s1['pade']
                pade_saved_iters.append(s1[keyi] - s0[keyi])
                sparse_solver_times.append(
                    get_sparse_solver_time(s0p) / s0[keyt])

    with open('output/numdefs.tex', 'w') as fout:
        def w(k, v):
            print(r'\newcommand{\%s}{%s}' % (k, v), file=fout)
        w('padeIterReduce',
          r'${:.2f}\pm{:.2f}$'.format(
              *mean_confidence_interval(pade_saved_iters)))
        w('padeNrCases', str(len(pade_saved_iters)))
        w('overallSpeedup', '{:.2f}'.format(g_tot_time_other / g_tot_time_sanm))
        w('gmeanSpeedup', '{:.2f}'.format(gmean(g_all_speedup)))
        w('gmeanSpeedupGravity', '{:.2f}'.format(gmean(g_gravity_speedup)))
        w('gmeanSpeedupDeform', '{:.2f}'.format(gmean(g_deform_speedup)))
        w('speedupNrCases', str(len(g_all_speedup)))
        m, h = mean_confidence_interval(sparse_solver_times)
        w('sparseSolverTimeUsed',
          r'${:.2f}\%\pm{:.2f}\%$'.format(m * 100, h * 100))


def main():
    gen_table_gravity()
    gen_table_deform()
    # plot_plant_gd()
    # gen_table_aramadillo_cmp()
    plot_armadillo_g()
    gen_table_chen_cmp()
    set_plot_style()
    scalability()
    numdefs()

if __name__ == '__main__':
    main()
