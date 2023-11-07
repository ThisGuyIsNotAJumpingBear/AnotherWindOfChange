
import json 
import os 
from collections import defaultdict

files = ['data/test-codalab-10k.data.jl', 'data/train.data.jl', 'data/trial.data.jl', 'data/validation.data.jl']

def load_instances(fn):

    instances = []
    with open(fn) as f:
        for jl_str in f:
            instances.append(json.loads(jl_str))
    
    return instances


date_dict = defaultdict(list)

tweet1_info = ['id', 'word', 'tweet1']
tweet2_info = ['id', 'word', 'tweet2']

for file in files:
    instances = load_instances(file)
    for pair in instances:
        tweet1 = {'id': pair['id'], 'word': pair['word'], }
        tweet2 = {}
