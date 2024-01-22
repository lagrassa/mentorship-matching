from .nlp_utils import get_document_vector
from gensim.models import TfidfModel, KeyedVectors
import numpy as np

def get_word2vec_model():
    model_path = "data/word2vec_model.bin"
    word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    return word2vec_model


#path = api.load("word2vec-google-news-300", return_path=True)
#word2vec_model = api.load("word2vec-google-news-300")
#word2vec_model.save_word2vec_format("word2vec_model.bin", binary=True)
def compute_distance(word2vec_model, mentor_tokens, student_tokens):
    mentor_vector = get_document_vector(mentor_tokens, word2vec_model)
    student_vector = get_document_vector(student_tokens, word2vec_model)
    cosine_similarity = cosine_similarity_func(student_vector, mentor_vector)
    distance = word2vec_model.wmdistance(student_tokens, mentor_tokens)
    print("cosine similarity", cosine_similarity)
    print("distance", distance)
    return distance


def cosine_similarity_func(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0

    return dot_product / (norm_vec1 * norm_vec2)


def compute_similarity_old():
    index = SparseMatrixSimilarity(tfidf[mentor_corpus], num_features=len(dictionary))
    # similarity = index[mentor_tfidf]
    query_document = student_interests.split()
    query_bow = dictionary.doc2bow(query_document)
    sims = index[tfidf[query_bow]]
    return sims
