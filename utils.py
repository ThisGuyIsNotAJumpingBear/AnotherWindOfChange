import pickle
import json
import csv

def open_pickle_file(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)
    

def get_sorted_tweets():
    return open_pickle_file('data/sorted_tweets.pkl')

def get_target_words():
    return open_pickle_file('data/target_words.pkl')


def load_instances():
    files = ['data/test-codalab-10k.data.jl', 'data/train.data.jl', 'data/trial.data.jl', 'data/validation.data.jl']
    instances = []
    for file in files:
        with open(file) as f:
            for jl_str in f:
                instances.append(json.loads(jl_str))
    
    return instances

def load_labels():
    label_files = ['data/test.gold.tsv', 'data/train.labels.tsv', 'data/trial.gold.tsv', 'data/validation.labels.tsv']
    labels = {}
    for file in label_files:
        with open(file) as fd:
            rd = csv.reader(fd, delimiter="\t")
            for row in rd:
                labels[row[0]] = row[1]
    return labels

def load_annotator_labels():
    labels = {}
    with open("data/annotator.tsv") as fd:
        rd = csv.reader(fd, delimiter=" ")
        for row in rd:
            labels[row[0]] = row[1]

    return labels