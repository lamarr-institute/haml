import os
from argparse import ArgumentParser

import numpy as np

from . import parse_file


def remove_empty_lines(s: str):
    return '\n'.join([line for line in s.splitlines() if line.strip()])


def get_args():
    p = ArgumentParser()
    p.add_argument('file', help='input HAML file')
    p.add_argument('-d', '--directory', default='.', help='output directory for generated files')
    p.add_argument('-a', '--all', action='store_true', help='generate all possible files')
    p.add_argument('-s', '--sample', type=int, default=0, help='randomly sample files')
    p.add_argument('--seed', type=int, default=None, help='random seed for sampling')
    p.add_argument('--rvlimit', type=int, default=1, help='number of samples when running `all` on random variables with infinite support')
    p.add_argument('--keep-empty-lines', action='store_true')
    return p.parse_args()


def main():
    args = get_args()
    obj = parse_file(args.file)

    global RANDOM_VALUE_LIMIT
    RANDOM_VALUE_LIMIT = args.rvlimit

    rng = np.random.default_rng(args.seed)
    basename = os.path.basename(args.file).rsplit('.', 1)[0]
    if args.all:
        if input(f'Proceed to create {obj.num_combinations()} files? [y/n] ').lower() != 'y':
            exit(0)
        for i, s in enumerate(obj.all(random_state=rng)):
            with open(os.path.join(args.directory, basename+f'_{i}.yml'), 'w') as f:
                f.write(s if args.keep_empty_lines else remove_empty_lines(s))
        
    elif args.sample > 0:
        for i in range(args.sample):
            with open(os.path.join(args.directory, basename+f'_{i}.yml'), 'w') as f:
                content = obj.random(random_state=rng)
                f.write(content if args.keep_empty_lines else remove_empty_lines(content))


if __name__ == '__main__':
    main()