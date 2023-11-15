import gensim
import pickle
from utils import get_sorted_tweets, get_target_words
       
def main():
    #since dataset is small, we chose a larger k 
    date_dict = get_sorted_tweets()
    
    years = ["2019", "2020", "2021"]
    for year in years:
        target_words = save_target_words(year)

        k = 12
        model = gensim.models.Word2Vec(
            sg=1, # skipgram
            hs=0, # negative sampling
            negative=k, # number of negative samples
            workers=4,
            vector_size=100
        )

        # Train
        # sentences = PathLineSentences(corpDir)
        
        sentence_list = []
        data_year = date_dict[year]
        for data in data_year:
            sentence_list.append(data["tokens"])

        model.build_vocab(sentence_list)
        model.train(sentence_list, total_examples=model.corpus_count, epochs=20)

        print(model)

        #if normalize
        # if is_len:
        #     # L2-normalize vectors
        #     model.init_sims(replace=True)

        # # Save the vectors and the model
        outpath = f'model_files/sngs_{year}'
        model.wv.save(outpath)
        model.save(outpath + '.model')

        word2vec = gensim.models.KeyedVectors.load(outpath)
        w2v_vocab = word2vec.key_to_index

        words = list(w2v_vocab.keys())
        for target in target_words:
            if target not in words:
                print(target)


def main3():
    target_words = get_target_words()
    
    word2vec_path = "model_files/sngs_2019"
    word2vec = gensim.models.KeyedVectors.load(word2vec_path)
    w2v_vocabulary = word2vec.key_to_index
    print(w2v_vocabulary)

    words = list( w2v_vocabulary.keys())
    for target in target_words:
        if target not in words:
            print(target)


def save_target_words(year):
    target_words = set()

    date_dict = get_sorted_tweets()
    
    for data in date_dict[year]:
        target_words.add(data["word"])
    
    with open(f'data/target_words_{year}.pkl', 'wb') as fp:
        pickle.dump(list(target_words), fp)
    
    return list(target_words)
    

if __name__ == "__main__":
    # save_target_words("2019")
    main()