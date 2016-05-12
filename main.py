import sys
import argparse
import logging

import numpy as np
import os

from text_preprocessor import TextPreprocessor

from feature_extraction import features as mesh_features
from feature_extraction import get_target
from listnet import ListNet
from sklearn.preprocessing import MinMaxScaler


def get_candidates(citations, pmid):
    # Find candidate mesh terms (neighbor/cited articles)
    candidate_terms = set()

    attributes = citations[pmid]

    for pmid, _ in attributes['neighbors']:
        if pmid in citations:
            candidate_terms |= set(citations[pmid]["mesh"])

    for pmid in attributes['cites']:
        if pmid in citations:
            candidate_terms |= set(citations[pmid]["mesh"])

    return candidate_terms

def get_neighbor_candidates(citations, pmid):
    # Find candidate mesh terms (neighboring articles)
    candidate_terms = set()

    attributes = citations[pmid]

    for pmid, _ in attributes['neighbors']:
        if pmid in citations:
            candidate_terms |= set(citations[pmid]["mesh"])

    return candidate_terms

def get_citation_candidates(citations, pmid):
    # Find candidate mesh terms (cited articles)
    candidate_terms = set()

    attributes = citations[pmid]

    for pmid in attributes['cites']:
        if pmid in citations:
            candidate_terms |= set(citations[pmid]["mesh"])

    return candidate_terms


def engineer_features(citations, pmid):
    candidates = get_candidates(citations, pmid)
    data = [
        np.asarray([func(citations, pmid, candidate)
                    for func in mesh_features])
        for candidate in candidates
    ]
    return data

class LNet():
    def __init__(self):
        self.current_file = None

    def switch(self, num_iters, learning_rate):
        with open('models/model_iter%d_gamma%.02f' % (num_iters, learning_rate)) as params_file:
            self.weights = np.asarray([float(weight) for weight in params_file.readlines()])

    def get_score(self, citations, pmid, mesh_term):
        features = np.asarray([func(citations, pmid, mesh_term) for func in mesh_features])
        logging.info("Features: {}".format(features))
        return self.weights @ features
                


def listnet_score(lnet, citations, pmid, mesh_term):
    return lnet.get_score(citations, pmid, mesh_term)



def generate_targets(citations, pmid):
    candidates = get_candidates(citations, pmid)
    targets = np.asarray([get_target(citations, pmid, candidate)
                          for candidate in candidates])
    return targets

def scale_features(features):
    # Scale the features
    feature_matrix = np.vstack(features)
    min_max_scaler = MinMaxScaler()
    return min_max_scaler.fit_transform(feature_matrix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Ranker for MeSH terms')

    parser.add_argument('--citations-file', '-c', type=str, default='')
    parser.add_argument('--num-mesh-terms', '-k', type=int)
    parser.add_argument('-v', '--verbosity', action='count', default=0)
    parser.add_argument('-t', '--test', action='store_true')
    args = parser.parse_args()

    if(args.verbosity == 0):
        level = logging.ERROR
    elif(args.verbosity == 1):
        level = logging.WARN
    elif(args.verbosity == 2):
        level = logging.INFO
    elif(args.verbosity >= 3):
        level = logging.DEBUG

    logging.basicConfig(
        format='[%(filename)s] %(levelname)s: %(message)s',
        level=level,
    )

    logging.info('Loading dataset')
    if args.citations_file:
        logging.debug('Loading from ' + args.citations_file)

        logging.error('Not implemented yet!')
        sys.exit(1)

        citations = {}
    else:
        logging.debug('Loading from raw sources')
        if args.test:
            citations = TextPreprocessor(use_cfg='config/test.cfg')
        else:
            citations = TextPreprocessor()

    metadata = {
        'input_size': len(citations.articles),
        'scores': [1]
    }
    logging.debug(
        'Number of training examples: {}'.format(metadata['input_size']))

    ranker = ListNet(n_stages=1, hidden_size=6)
    ranker.initialize_learner(metadata)
    logging.info('Write features out to file')

    ''' Train ListNet:
    candidates - is a list of feature vectors (ndarray)
    targets - is a list of relevance scores - whether or not
    the candidate MeSH term is a correct choice (0 or 1)
    query - is the pmid of the current article.
    '''
    if args.test:
        with open('test.txt', 'w') as f:
            for pmid in citations.articles:
                candidates = get_candidates(citations, pmid)
                logging.debug('Building features for %d' % pmid)
                logging.debug(engineer_features)
                features = engineer_features(citations, pmid)
                # Scale the features
                # features = scale_features(features)
                for feature_vec in features:
                    current_features = 'qid:{} '.format(pmid)
                    current_features += ' '.join(('%s:%s' % (feat_num + 1, val) for feat_num, val in enumerate(feature_vec)))
                    f.write(current_features + '\n')



            # model = LNet()
            # model.switch(50, .01)
            # logging.info("{}"
            #     .format(model.get_score(citations, 9317033, 'mutation')))
            # logging.info("weights:{}".format(model.weights))
            # Score for MeSH term {} in article{} is:\n

    else:
        with open('features.txt', 'w') as f:
            for pmid in citations.articles:
                logging.debug('Ranking MeSH terms for %s' % pmid)
                candidates = get_candidates(citations, pmid)
                logging.debug('Building features for %d' % pmid)
                logging.debug(engineer_features)
                features = engineer_features(citations, pmid)

                # Scale the features
                # features = scale_features(features)

                targets = generate_targets(citations, pmid)
                training_example = [features, targets, pmid]
                for target, feature_vec in zip(targets, features):
                    current_features = '{} qid:{} '.format(target, pmid)

                    current_features += ' '.join(('%s:%s' % (feat_num + 1, val) for feat_num, val in enumerate(feature_vec)))

                    f.write(current_features + '\n')

