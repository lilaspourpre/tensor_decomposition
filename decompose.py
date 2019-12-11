import argparse
import os
import json
import numpy as np
import sparse
from tensorly.contrib.sparse.decomposition import parafac, partial_tucker
from tensorly.contrib.sparse import tensor

from utils import *


def parse_arguments():
    parser = argparse.ArgumentParser("Tensor decomposition using PARAFAC/TUCKER")
    parser.add_argument("-i", "--input_path", dest="input_path", required=True)
    parser.add_argument("-o", "--output_path", dest="output_path", required=False)
    parser.add_argument("-e", "--embedding_size", dest="embedding_size", default=50)
    parser.add_argument('-a', '--algorithm', dest='algorithm', choices=['parafac', 'tucker'], default='parafac')
    parser.add_argument('-m', '--min_count', dest='min_count', default=5)
    parser.add_argument('-n', "--first_n", dest='first_n', default=1000,
                        help="number of lines to cut (use not all dataset)")
    return parser.parse_args()


def create_tensor(lines, verbs, subjects, objects):
    indices = create_indices(lines, objects, verbs, subjects)
    values = [1] * len(indices)
    coords = np.array(indices, dtype=np.int32).T
    values = np.array(values, dtype=np.float64)
    X = sparse.COO(coords, values, shape=(len(verbs), len(subjects), len(objects)))
    X = tensor(X)
    return X


def create_indices(lines, objects, verbs, subjects):
    indices = [(verbs[v], subjects[s], objects[o]) for v, s, o in lines]
    return indices


def save_to_file(data, vocab, output_path):
    with open(output_path, 'w', encoding='utf-8') as w:
        for word, vector in zip(vocab, data):
            w.write(f"{word}\t{json.dumps(list(vector))}\n")


def main():
    args = parse_arguments()
    lines, verb2id, subject2id, object2id = get_dict_and_samples(args.input_path, args.min_count, args.first_n)
    large_tensor = create_tensor(lines, verb2id, subject2id, object2id)

    if args.algorithm == 'tucker':
        weights, factors = partial_tucker(large_tensor, modes=[0, 1, 2], rank=args.embedding_size, init='svd')
    else:
        weights, factors = parafac(large_tensor, rank=args.embedding_size, init='random')

    assert [factor.shape[0] for factor in factors] == [len(verb2id), len(subject2id), len(object2id)]

    output_path = os.path.join(args.output_path,
                               f"{args.input_path[5:13]}-{args.algorithm}_e{args.embedding_size}_"
                               f"min-count-{args.min_count}_cut-{args.first_n}")

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    save_to_file(factors[0], verb2id, os.path.join(output_path, 'verbs.tsv'))
    save_to_file(factors[1], subject2id, os.path.join(output_path, 'subjects.tsv'))
    save_to_file(factors[2], object2id, os.path.join(output_path, 'objects.tsv'))


if __name__ == '__main__':
    main()
