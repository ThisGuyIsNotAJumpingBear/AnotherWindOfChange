
import gensim
from procrustes import orthogonal
import numpy as np
import pickle
from sklearn import metrics
from utils import get_target_words

def cosine_dist(v1, v2):
    return 1 - metrics.pairwise.cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]

def get_same_words(word_set1, word_set2):
    return word_set1.intersection(word_set2)

def create_matrices(word2vec1, word2vec2):

    vector_size = 100
    vocab1 = set(word2vec1.index_to_key)
    vocab2 = set(word2vec2.index_to_key)

    intersect = get_same_words(vocab1, vocab2)

    mat1 = np.zeros((len(intersect), vector_size))
    mat2 = np.zeros((len(intersect), vector_size))

    for i, word in enumerate(intersect):
        mat1[i] = word2vec1.get_vector(word)
        mat2[i] = word2vec2.get_vector(word)
    

    return list(intersect), mat1, mat2

def display_procrutes_result(result):
    # display Procrustes results
    aq = np.dot(result.new_a, result.t)
    print("Procrustes Error = ", result.error)
    print("Does AQ and B matrices match?", np.allclose(aq, result.new_b))

    print("Transformation Matrix T = ")
    print(result.t)
    print("")
    print("Matrix A (after translation and scaling) = ")
    print(result.new_a)
    print("")
    print("Matrix AQ = ")
    print(aq)
    print("")
    print("Matrix B (after translation and scaling) = ")
    print(result.new_b)


def run(year1, year2):
    target_words = get_target_words()

    word2vec_path1 = f"model_files/sngs_{year1}"
    word2vec_path2 = f"model_files/sngs_{year2}"
    word2vec1 = gensim.models.KeyedVectors.load(word2vec_path1)
    word2vec2 = gensim.models.KeyedVectors.load(word2vec_path2)

    intersect, mat1, mat2 = create_matrices(word2vec1, word2vec2)

    result = orthogonal(mat1, mat2, scale=True, translate=True)
    # display_procrutes_result(result)

    cosine_distances = {}
    for target in target_words:
        try:
            idx = intersect.index(target)
            cosine_distances[target] = cosine_dist(result.new_a[idx], result.new_b[idx])
        except:
            pass
            # print(f"target word {target} is not found in the intersect of both corpora")
    
    return cosine_distances

def main():
    years = [("2019", "2020"), ("2020", "2021")]

    for year1, year2 in years:
        cd = run(year1, year2)
        print(len(cd.keys()))

if __name__ == '__main__':
    main()