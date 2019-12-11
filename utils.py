import gzip
from nltk.corpus import stopwords


def get_dict_and_samples(filename: str, min_count: int, first_n: int) -> tuple:
    subject_set = set()
    object_set = set()
    verb_set = set()
    lines = []

    with gzip.open(filename, 'r') as gz:
        for line in gz:
            v, s, o, count = line.decode("utf-8")[:-1].split('\t')
            if float(count) > min_count and is_valid(s) and is_valid(o):
                subject_set.add(s)
                object_set.add(o)
                verb_set.add(v)
                lines.append((v, s, o))
            if len(lines) == first_n:
                break

    print(f"Verbs (dim1): {len(verb_set)}, Subjects (dim2): {len(subject_set)}, Objects (dim3): {len(object_set)}")
    return lines, {word: i for i, word in enumerate(verb_set)}, \
           {word: i for i, word in enumerate(subject_set)}, \
           {word: i for i, word in enumerate(object_set)}


def is_valid(word: str):
    return len(word) > 1 and word.isalnum() and word not in stopwords.words('english')
