
import json 
from collections import defaultdict
import pickle

files = ['data/test-codalab-10k.data.jl', 'data/train.data.jl', 'data/trial.data.jl', 'data/validation.data.jl']

def load_instances(fn):

    instances = []
    with open(fn) as f:
        for jl_str in f:
            instances.append(json.loads(jl_str))
    
    return instances

parent_info = ['id', 'word']
tweet_info = ['text', 'tokens', 'token_idx', 'date']

def pop_dicts(data):

    tweet_arrs = []
    for i in range(2):
        tweet = {}
        for key in parent_info:
            tweet[key] = data[key]
        
        tweet_key = f"tweet{i+1}"
        for key in tweet_info:
            tweet[key] = data[tweet_key][key]
        
        tweet_arrs.append(tweet)
    
    return tweet_arrs

def extract_year(tweet_data):
    return tweet_data['date'][:4]

def assert_tweet_in_correct_key(date_dict):
    years = list(date_dict.keys())

    for year in years:
        for tweet in date_dict[year]:
            assert tweet['date'][:4] == year
    
    assert True


def main():
    date_dict = defaultdict(list)

    for file in files:
        instances = load_instances(file)
        for data in instances:
            tweet1, tweet2 = pop_dicts(data)
            tweet1_year, tweet2_year = extract_year(tweet1),extract_year(tweet2)
            date_dict[tweet1_year].append(tweet1)
            date_dict[tweet2_year].append(tweet2)

    assert_tweet_in_correct_key(date_dict)

    with open('data/sorted_tweets.pkl', 'wb') as fp:
        pickle.dump(date_dict, fp)
    
    with open('data/sorted_tweets.pkl', 'rb') as fp:
        date_dict_test = pickle.load(fp)
        assert_tweet_in_correct_key(date_dict_test)
    

if __name__ == "__main__":
    main()




        

