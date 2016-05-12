import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel

from text_preprocessor import TextPreprocessor


def unigram_overlap(citations, pmid, mesh_term):
    ''' Returns the number of unigrams that overlap between the
    title of an article and a MeSH term.

    Assumes article_title and term are strings. '''

    words = mesh_term.split()
    overlap = 0

    article_title = citations[pmid]['title']

    for word in words:
        if word in article_title:
            overlap += 1

    return overlap


def bigram_overlap(citations, pmid, mesh_term):
    ''' Returns the number of bigrams that overlap between the
    title and abstract of an article and a MeSH term.

    Assumes article_title, article_abstract, and term are all strings. '''

    words = mesh_term.split()
    bigrams = zip(words, words[1:])
    overlap = 0

    article_title = citations[pmid]['title']
    article_abstract = citations[pmid]['abstract']

    for bigram in bigrams:
        gram = bigram[0] + " " + bigram[1]

        if gram in article_title:
            overlap += 1

        if gram in article_abstract:
            overlap += 1

    return overlap


def citation_similarities(citations, pmid, mesh_term):
    '''Return the sum of similarity scores between the article we're
    trying to predict and every citation that contains this MeSH term. '''
    total_score = 0.0

    for cite_pmid in citations[pmid]['cites']:
        if mesh_term in citations[cite_pmid]['mesh']:
            total_score += citations.similarity_scores[pmid][cite_pmid]

    return total_score


def neighboring_similarities(citations, pmid, mesh_term):
    '''Returns total neighborosity score of each article_neighbors that
    the term appears in.'''
    total_score = 0.0

    article_neighbors = citations[pmid]['neighbors']

    for neighbor, score in article_neighbors:
        if neighbor not in citations:
            continue

        if mesh_term in citations[neighbor]["mesh"]:
            total_score += float(score)

    return total_score


def sum_similarities(citations, pmid, mesh_term):
    ''' Sum the total neighborosity score and citation similarity score for
    a candidate MeSH term. '''
    return neighboring_similarities(citations, pmid, mesh_term) + citation_similarities(citations, pmid, mesh_term)


def neighboring_count(citations, pmid, mesh_term):
    ''' Return the count of how many times the given MeSH term
    appears in neighbors '''
    count = 0

    article_neighbors = citations[pmid]['neighbors']

    for neighbor, score in article_neighbors:
        if neighbor not in citations:
            continue

        if mesh_term in citations[neighbor]["mesh"]:
            count += 1

    return count


def citation_count(citations, pmid, mesh_term):
    ''' Count the number of citations for each article. '''
    count = 0

    article_citations = citations[pmid]['cites']

    for citation in article_citations:
        if mesh_term in citations[citation]['mesh']:
            count += 1

    return count


def sum_count(citations, pmid, mesh_term):
    ''' Sum the times this mesh term appears in an article's
    citations and neighbor articles.'''
    return neighboring_count(citations, pmid, mesh_term) + citation_count(citations, pmid, mesh_term)


def get_target(citations, pmid, mesh_term):
    ''' Returns a 0/1 (true/false) depending on whether a candidate MeSH term
    is actually contained in the article. Used by ListNET.'''

    return 1 if mesh_term in citations[pmid]['mesh'] else 0

# features = {
#     'bigram_overlap': bigram_overlap,
#     'citation_count': citation_count,
#     'neighboring_count': neighboring_count,
#     'unigram_overlap': unigram_overlap,
#     'neighboring_similiarities': neighboring_similarities,
#     'citation_similarities': citation_similarities
# }

features = [
    bigram_overlap,
    citation_count,
    # neighboring_count,
    # sum_count,
    unigram_overlap,
    # neighboring_similarities,
    citation_similarities
    # sum_similarities
]


def get_tf_idf_model(citations=None):
    if citations is None:
        citations = TextPreprocessor()
        citations.preprocess()

    documents = [
        citation['title'] + ' \n' + citation['abstract']
        for citation in list(citations.values())
    ]
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2))
    bigrams = bigram_vectorizer.fit_transform(documents)

    tfidf = TfidfTransformer().fit_transform(bigrams)

    return citations, bigram_vectorizer, tfidf


def get_most_similar_documents(tfidf_matrix, vectorizer, query):
    query_tfidf = TfidfTransformer().fit_transform(
        vectorizer.transform([query])
    )
    document_similarities = linear_kernel(query_tfidf, tfidf_matrix).flatten()
    return document_similarities.argsort()[::-1]


if __name__ == '__main__':
    citations = TextPreprocessor()

    input("Hit enter to continue...")

    count = 0
    skipped_count = 0
    for article, attributes in citations.items():
        # Skip neighboring articles
        if len(attributes['neighbors']) == 0:
            skipped_count += 1
            continue

        candidate_terms = get_candidates(citations, article)

        # Make features
        # Just printing them for now
        print("Article: {}".format(article))
        count += 1
        for term in candidate_terms:
            features = dict()
            features['unigram'] = unigram_overlap(citations, article, term)
            features['bigram'] = bigram_overlap(citations, article, term)
            features['citation_frequency'] = citation_count(
                citations, attributes['cites'], term)

            features['neighbor_frequency'] = neighbor_freq
            features['neighbor_score'] = neighbor_score

            print("Candidate Term: {}, Features: {}".format(term, features))
        print("\n")

    print('Count: {}'.format(count))
    print('Skipped Count: {}'.format(skipped_count))
