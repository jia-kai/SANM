#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pymeshlab

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(args.input)
    ms.re_compute_vertex_normals(weightmode=2)
    ms.save_current_mesh(file_name=args.output)

if __name__ == '__main__':
    main()
