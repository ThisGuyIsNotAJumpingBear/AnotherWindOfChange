import pickle
import json

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