
import json 
from collections import defaultdict
import pickle
import csv


files = ['data/test-codalab-10k.data.jl', 'data/train.data.jl', 'data/trial.data.jl', 'data/validation.data.jl']
label_files = ['data/test.gold.tsv', 'data/train.labels.tsv', 'data/trial.gold.tsv', 'data/validation.labels.tsv']


def load_labels():
    labels = {}
    for file in label_files:
        with open(file) as fd:
            rd = csv.reader(fd, delimiter="\t")
            for row in rd:
                labels[row[0]] = row[1]
    return labels

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
        if "target_word" not in year:
            for tweet in date_dict[year]:
                assert tweet['date'][:4] == year
    
    assert True

def main():
    date_dict = defaultdict(list)
    labels = load_labels()

    for file in files:
        instances = load_instances(file)
        for data in instances:
            tweet1, tweet2 = pop_dicts(data)
            if data["id"] in labels.keys():
                label = labels[data['id']]
                tweet1["label"], tweet2["label"] = label, label
                print(tweet1)
            tweet1_year, tweet2_year = extract_year(tweet1), extract_year(tweet2)
            date_dict[tweet1_year].append(tweet1)
            date_dict[tweet2_year].append(tweet2)
            if tweet1["word"] not in date_dict[f"target_word_{tweet1_year}_{tweet2_year}"]:
                date_dict[f"target_word_{tweet1_year}_{tweet2_year}"].append(tweet1["word"])

    assert_tweet_in_correct_key(date_dict)

    with open('data/sorted_tweets.pkl', 'wb') as fp:
        pickle.dump(date_dict, fp)
    
    with open('data/sorted_tweets.pkl', 'rb') as fp:
        date_dict_test = pickle.load(fp)
        assert_tweet_in_correct_key(date_dict_test)

def get_target_words_per_year():
    with open('data/target_words.pkl', 'rb') as fp:
        target_words = pickle.load(fp)

    print(len(target_words))

    years = [("2019", "2020"), ("2020", "2021")]

    with open('data/sorted_tweets.pkl', 'rb') as fp:
        date_dict = pickle.load(fp)
        assert_tweet_in_correct_key(date_dict)

    for year1, year2 in years:
        string = f"target_word_{year1}_{year2}"
        print(len(date_dict[string]))
    
if __name__ == "__main__":
    # main()
    get_target_words_per_year()




        

