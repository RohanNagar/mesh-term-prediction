# Copyright 2011 Hugo Larochelle. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
#    1. Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright notice, this list
#       of conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY Hugo Larochelle ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Hugo Larochelle OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are those of the
# authors and should not be interpreted as representing official policies, either expressed
# or implied, of Hugo Larochelle.

import numpy as np
import logging
from generic import OnlineLearner


"""
The ``learners.ranking`` module contains learners meant for ranking problems. 
The MLProblems for these learners should be iterators over 
triplets (input,target,query), where input is a list of
document representations and target is a list of associated 
relevance scores for the given query.

The currently implemented algorithms are:

* RankingFromClassifier:  a ranking model based on a classifier.
* RankingFromRegression:  a ranking model based on a regression model.
* ListNet:                ListNet ranking model.

"""


def default_merge(input, query):
    return input


def err_and_ndcg(output, target, max_score, k=10):
    """
    Computes the ERR and NDCG score (taken mostly from here:
    http://learningtorankchallenge.yahoo.com/evaluate.py.txt)
    """

    err = 0.
    ndcg = 0.
    l = [int(x) for x in target]
    r = [int(x) + 1 for x in output]
    nd = len(target)  # Number of documents
    assert len(output) == nd, 'Expected %d ranks, but got %d.' % (nd, len(r))

    # The first element is the gain of the first document in the predicted
    # ranking
    gains = [-1] * nd
    assert max(r) <= nd, 'Ranks larger than number of documents (%d).' % (nd)
    for j in range(nd):
        gains[r[j] - 1] = (2.**l[j] - 1.0) / (2.**max_score)
    assert min(gains) >= 0, 'Not all ranks present.'

    p = 1.0
    for j in range(nd):
        r = gains[j]
        err += p * r / (j + 1.0)
        p *= 1 - r

    dcg = sum([g / np.log(j + 2) for (j, g) in enumerate(gains[:k])])
    gains.sort()
    gains = gains[::-1]
    ideal_dcg = sum([g / np.log(j + 2) for (j, g) in enumerate(gains[:k])])
    if ideal_dcg:
        ndcg += dcg / ideal_dcg
    else:
        ndcg += 1.

    return (err, ndcg)


class ListNet(OnlineLearner):
    """ 
    ListNet ranking model.

    This implementation only models the distribution of documents
    appearing first in the ranked list (this is the setting favored in
    the experiments of the original ListNet paper). ListNet is trained
    by minimizing the KL divergence between a target distribution
    derived from the document scores and ListNet's output
    distribution.

    Option ``n_stages`` is the number of training iterations over the
    training set.

    Option ``hidden_size`` determines the size of the hidden layer (default = 50).

    Option ``learning_rate`` is the learning rate for stochastic
    gradient descent training (default = 0.01).

    Option ``weight_per_query`` determines whether to weight each
    ranking example (one for each query) by the number of documents to
    rank. If True, the effect is to multiply the learning rate by
    the number of documents for the current query. If False, no weighting
    is applied (default = False).

    Option ``alpha`` controls the entropy of the target distribution
    ListNet is trying to predict: ``target = exp(alpha *
    scores)/sum(exp(alpha * scores))`` (default = 1.).

    Option ``merge_document_and_query`` should be a 
    callable function that takes two arguments (the 
    input document and the query) and outputs a 
    merged representation for the pair which will
    be fed to ListNet. By default, it is assumed
    that the document representation already contains
    query information, and only the document the input
    document is returned.

    Option ``seed`` determines the seed of the random number generator
    used to initialize the model.

    **Required metadata:**

    * ``'scores'``

    | **Reference:** 
    | Learning to Rank: From Pairwise Approach to Listwise Approach
    | Cao, Qin, Liu, Tsai and Li
    | http://research.microsoft.com/pubs/70428/tr-2007-40.pdf

    """

    def __init__(self, n_stages, hidden_size=50,
                 learning_rate=0.01,
                 weight_per_query=False,
                 alpha=1.,
                 merge_document_and_query=default_merge,
                 seed=1234):

        self.n_stages = n_stages
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weight_per_query = weight_per_query
        self.alpha = alpha
        self.merge_document_and_query = merge_document_and_query
        self.seed = seed

        self.stage = 0

    def initialize_learner(self, metadata):
        self.rng = np.random.mtrand.RandomState(self.seed)
        input_size = metadata['input_size']
        self.max_score = max(metadata['scores'])
        self.V = (
            2 * self.rng.rand(input_size, self.hidden_size) - 1) / input_size

        self.c = np.zeros((self.hidden_size))
        self.W = (2 * self.rng.rand(self.hidden_size, 1) - 1) / \
            self.hidden_size
        self.b = np.zeros((1))

    def update_learner(self, example):
        input_list = example[0]
        relevances = example[1]
        query = example[2]
        n_documents = len(input_list)

        target_probs = np.zeros((n_documents, 1))
        input_size = input_list[0].shape[0]
        inputs = np.zeros((n_documents, input_size))
        logging.debug('inputs size:{}'.format(inputs.shape))


        for t, r, il, input_ in zip(target_probs, relevances, input_list, inputs):
            t[0] = np.exp(self.alpha * r)
            input_[:input_size] = self.merge_document_and_query(il, query)
        target_probs = target_probs / np.sum(target_probs, axis=0)

        hid = np.tanh(np.dot(inputs, self.V) + self.c)

        outact = np.dot(hid, self.W) + self.b
        outact -= np.max(outact)
        expout = np.exp(outact)
        output = expout / np.sum(expout, axis=0)

        doutput = output - target_probs
        dhid = np.dot(doutput, self.W.T) * (1 - hid**2)

        if self.weight_per_query:
            lr = self.learning_rate * n_documents
        else:
            lr = self.learning_rate
        self.W -= lr * np.dot(hid.T, doutput)
        self.b -= lr * np.sum(doutput)
        self.V -= lr * np.dot(inputs.T, dhid)
        self.c -= lr * np.sum(dhid, axis=0)

    def use_learner(self, example):
        input_list = example[0]
        n_documents = len(input_list)
        query = example[2]

        input_size = input_list[0].shape[0]
        inputs = np.zeros((n_documents, input_size))
        for il, input_ in zip(input_list, inputs):
            input_[:input_size] = self.merge_document_and_query(il, query)

        hid = np.tanh(np.dot(inputs, self.V) + self.c)
        outact = np.dot(hid, self.W) + self.b
        outact -= np.max(outact)
        expout = np.exp(outact)
        output = expout / np.sum(expout, axis=0)
        ordered = np.argsort(-output.ravel())
        order = np.zeros(len(ordered))
        order[ordered] = list(range(len(ordered)))
        return order

    def cost(self, output, example):
        return err_and_ndcg(output, example[1], self.max_score)
