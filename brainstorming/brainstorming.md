# How to extract sum of citation document similarity scores for each MeSH term
### For each article in the articles dataset (e.g. SMALL200):
- Vectorize each article. -> CountVectorizer() and TfidfVectorizer()
- Vectorize each of its citations.
- Compute document similarity scores between articles and each of their citations.
- Merge MeSH terms of citations together via set union.
- Loop through every article, check if MeSH term is contained or not.

# Some of our own terminology
- Candidate MeSH term - a MeSH term that is drawn from an article's citations or nearest-neighbor articles.

# Features that we have:
- Raw count of the # of times a MeSH term shows up in the citations.
- Sum of similarity scores for the citations that a MeSH term shows up in.
- Raw count of the # of times a MeSH term shows up in the 25 nearest-neighbor articles.
- Sum of similarity scores for the neighbor articles that a MeSH term shows up in.

# Features that we'd like to have:
- (Binary feature) whether or not MeSH term is a synonym with any bigrams in title/abstract

# ListNet terminology, applied to our problem:
- Query = Article we're trying to predict MeSH terms for
- Document = A candidate MeSH term for the article
- Target = Whether or not candidate MeSH term is correct
    - (i.e., whether or not humans chose the MeSH term for the article)

# Some Drawbacks to this Approach:
- MeSH terms that are not found within the citations/nearest-neighbors simply get fked over
- We can find a bunch of candidate MeSH terms, but they wouldn't necessarily contain all the "true" MeSH terms.

# So, What does ListNet need to run in our situation???
- query_id (qid) = PMID of article we're trying to predict
- for every article, a list of feature vectors corresponding to an article's candidate MeSH terms
- Important thing: Need to generate a unique ID (I think) for every single MeSH term.
- ground truth values (0 or 1) for whether a candidate MeSH term belongs lols.
