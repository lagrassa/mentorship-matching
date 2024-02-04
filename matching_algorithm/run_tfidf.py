from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


mentor_file = "data/mentor_desc.cor"
fake_mentor_file = "data/fake_mentor_desc.cor"
mentee_file = "data/mentee_desc.cor"
training_data_file = "data/training_data.cor"

import smart_open
def read_corpus(fname, tokens_only=False):
    lines = []
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if len(line) < 3:
                continue
            lines.append(line)
    return lines

train_corpus = read_corpus(training_data_file)
mentor_corpus = read_corpus(mentor_file)
mentee_corpus = read_corpus(mentee_file)

query_document = "using AI to detect cancer from images"

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(train_corpus+mentor_corpus)

# Transform the query document into a TF-IDF vector
query_vector = vectorizer.transform([query_document])
mentor_matrix = vectorizer.transform(mentor_corpus)
mentee_matrix = vectorizer.transform(mentee_corpus)


for i in range(len(mentee_corpus)):
    query_vector = mentee_matrix[i]
    similarity_scores = cosine_similarity(query_vector,mentor_matrix)
    most_similar_index = similarity_scores.argmax()
    max_sim_score = max(similarity_scores[0])
    if max_sim_score > 0.4:
        print("max similarity score", max_sim_score)
        print("Mentee interest", mentee_corpus[i])
        print(mentor_corpus[most_similar_index])
