import argparse
from sklearn.cluster import AffinityPropagation
import numpy as np
import os
import json
from collections import defaultdict

from utils import *


def parse_arguments():
    parser = argparse.ArgumentParser("Tensor decomposition using PARAFAC/TUCKER")
    parser.add_argument("-i", "--input_path", dest="input_path", required=True)
    parser.add_argument("-v", "--vector_path", dest="vector_path", required=True)
    parser.add_argument('-m', '--min_count', dest='min_count', default=5)
    parser.add_argument('-n', "--first_n", dest='first_n', default=1000,
                        help="number of lines to cut (use not all dataset)")
    return parser.parse_args()


def get_vectors(path):
    return read_vectors(os.path.join(path, "verbs.tsv")), read_vectors(os.path.join(path, "subjects.tsv")),\
           read_vectors(os.path.join(path, "objects.tsv"))


def read_vectors(path):
    w2vec = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            word, v_string = line.split('\t')
            vector = np.array(json.loads(v_string))
            w2vec[word] = vector
    return w2vec


def concat_vectors(lines, verbs, subjects, objects):
    return np.array([np.concatenate([verbs[v], subjects[s], objects[o]]) for v, s, o in lines])


def group_result(indices: list, triplets: list) -> dict:
    groups = defaultdict(list)
    assert len(indices) == len(triplets)
    for i, triplet in zip(indices, triplets):
        groups[i].append(triplet)
    return groups


def main():
    args = parse_arguments()
    verb2vec, subject2vec, object2vec = get_vectors(args.vector_path)
    lines, _, _, _ = get_dict_and_samples(args.input_path, args.min_count, args.first_n)
    concatenated = concat_vectors(lines, verb2vec, subject2vec, object2vec)
    print(f"Shape: {concatenated.shape}")
    ap = AffinityPropagation()
    result = ap.fit_predict(concatenated)
    groups = group_result(result, lines)
    print(f"Number of clusters: {len(groups)}")


if __name__ == '__main__':
    main()
