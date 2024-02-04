import logging
import numpy as np
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import os
import gensim
from gensim.models import KeyedVectors
# Set file names for train and test data
mentor_file = "data/mentor_desc.cor"
fake_mentor_file = "data/fake_mentor_desc.cor"
mentee_file = "data/mentee_desc.cor"
training_data_file = "data/training_data.cor"


import smart_open

def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

train_corpus = list(read_corpus(training_data_file))
#test_corpus = list(read_corpus(lee_test_file, tokens_only=True))
mentor_corpus = list(read_corpus(mentor_file, tokens_only=True))
fake_mentor_corpus = list(read_corpus(fake_mentor_file, tokens_only=True))
mentor_corpus.extend(fake_mentor_corpus)
mentee_corpus = list(read_corpus(mentee_file, tokens_only=True))

def train_and_eval_model(vector_size = 1, min_count=2, epochs=10, window=5, inference_epochs=1000):
    model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=2, epochs=epochs, window=window)
    model.build_vocab(train_corpus)
    model.compute_loss = True
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    mentor_vectors = [model.infer_vector(sentence, epochs=inference_epochs) for sentence in mentor_corpus]
    mentee_vectors = [model.infer_vector(sentence, epochs=inference_epochs) for sentence in mentee_corpus]
    mentor_keyed_vectors = KeyedVectors(vector_size=model.vector_size)
    for i, vector in enumerate(mentor_vectors):
        mentor_keyed_vectors.add_vector(i, vector)
    mentor_vectors = np.vstack(mentor_vectors)
    mentee_vectors = np.vstack(mentee_vectors)


    example_ids = [5,7,8,12,25,30]
    test_ids = [1,18,30,48,6,8,71,77]
    test_labels = [33, 34, 35, 36, 37, 38, 39,40]
    classification_score = 0
    sim_score = 0
    rel_sim_score = 0
    all_scores = []
    for mentee_id in range(len(mentee_corpus)):
        if mentee_id in test_ids:
            mentor_id = test_labels[test_ids.index(mentee_id)]
            #best_id, score = compute_distances_with_mentors(model, mentor_vectors, mentor_corpus, mentee_id, verbose=True)
            best_id, score = compute_similarities_with_mentors(model, mentor_keyed_vectors, mentor_corpus, mentee_id, verbose=True)
            mentee_vector = model.infer_vector(mentee_corpus[mentee_id])
            label_mentor_vector = model.infer_vector(mentor_corpus[mentor_id])
            #sim = np.dot(mentee_vector, label_mentor_vector) / (np.linalg.norm(mentee_vector) * np.linalg.norm(label_mentor_vector))
            sim = 1/np.linalg.norm(mentee_vector - label_mentor_vector)
            sim_score += sim
            rel_sim_score += sim / score
            classification_score += int(best_id==mentor_id)

    print("The similarity score is", sim_score)
    print("The relative similarity score is", rel_sim_score)
    print("Classification score is", classification_score)
    return sim_score, rel_sim_score, classification_score




def compute_similarities_with_mentors(model, mentor_keyed_vectors, mentor_corpus, mentee_id, verbose=False):
    inferred_vector = model.infer_vector(mentee_corpus[mentee_id])
    sims = mentor_keyed_vectors.most_similar([inferred_vector], topn=3)
    best_score = sims[0][1]
    if verbose:
        print(" ".join(mentee_corpus[mentee_id]))
        for idx, score in sims:
            print("Sim_score", score)
            print("Mentor text", " ".join(mentor_corpus[idx]))
    return sims[0]

def compute_distances_with_mentors(model, mentor_vectors, mentor_corpus, mentee_id, verbose=False):
    inferred_vector = model.infer_vector(mentee_corpus[mentee_id])
    distances = np.linalg.norm(mentor_vectors - inferred_vector, axis=1)
    best_id = np.argmin(distances)
    best_score = 1/np.min(distances)
    return best_id, best_score

sim_scores = []
rel_sim_scores = []
classification_scores = []
confs = []
#for epoch in [1,5,10,50,500]: 
epoch_per_run = []
vector_size_per_run = []
window_per_run = []
min_count_per_run = []
inference_epoch_per_run = []
for epoch in [100]: 
    for vector_size in [30]:
        for window in [3]:
            for min_count in [2]:
                for inference_epochs in [10]:
                    conf = {"vector_size":vector_size, "min_count":min_count, "epochs":epoch, "window":window, "inference_epochs":inference_epochs}
                    epoch_per_run.append(epoch)
                    vector_size_per_run.append(vector_size)
                    window_per_run.append(window)
                    min_count_per_run.append(min_count)
                    inference_epoch_per_run.append(inference_epochs)
                    sim_score, rel_sim_score, classification_score = train_and_eval_model(**conf)
                    sim_scores.append(sim_score)
                    rel_sim_scores.append(rel_sim_score)
                    classification_scores.append(classification_score)
                    confs.append(conf.copy())
    """
    print("Results for epoch", epoch)
    print("Sim scores", sim_scores)
    print("Rel sim scores", rel_sim_scores)
    print("classification scores", classification_scores)
    best_score = np.argmax(classification_scores)
    print("best idx", best_score)
    print("best conf", confs[best_score])
    """

#print("epochs", epoch_per_run)
#print("vectors", vector_size_per_run)
#print("windows", window_per_run)
#print("min scounts", min_count_per_run)
print("infs", inference_epoch_per_run)

print("Results over epochs", epoch)
print("Sim scores", sim_scores)
print("Rel sim scores", rel_sim_scores)
print("classification scores", classification_scores)
best_score = np.argmax(classification_scores)
print("best idx", best_score)
print("best conf", confs[best_score])
print("Complete")

