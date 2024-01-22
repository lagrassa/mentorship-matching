def preprocess_text(interest_text):
    tokens = word_tokenize(interest_text.lower())  # Convert to lowercase and tokenize
    stoplist = set('for a of the and to in student work also do how what when where why how from by my can be interested interest I'.split(' '))
    texts = [[word for word in document.lower().split() if word not in stoplist and word.isalnum]
             for document in tokens]
    return texts
def get_document_vector(tokens, model):
    vectors = [model[token] for token in tokens if token in model]
    if vectors:
        return sum(vectors) / len(vectors)
    return None
