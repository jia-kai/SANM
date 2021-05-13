#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

def main():
    parser = argparse.ArgumentParser(
        description='replace vertex coordinates in an .obj file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('obj', help='original obj file')
    parser.add_argument('vtx', help='new vertex coordinates, one per line')
    parser.add_argument('output', help='output obj file')
    args = parser.parse_args()

    with open(args.vtx) as fin:
        vtx = [i.strip() for i in fin.readlines()]

    vtx_id = 0
    with open(args.obj) as fin:
        with open(args.output, 'w') as fout:
            for line in fin:
                if line.startswith('v '):
                    fout.write(f'v {vtx[vtx_id]}\n')
                    vtx_id += 1
                else:
                    fout.write(line)
    assert vtx_id == len(vtx)
    print(f'remember to recompute normals (e.g., using MeshLab): {args.output}')

if __name__ == '__main__':
    main()
