import pickle

def open_pickle_file(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)
    

def get_sorted_tweets():
    return open_pickle_file('data/sorted_tweets.pkl')

def get_target_words():
    return open_pickle_file('data/target_words.pkl')