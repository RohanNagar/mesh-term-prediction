
def compare_terms(predicted, actual):
    ''' Returns a list of 1s or 0s where a 1 means that the predicted term
    is in the actual term list and a 0 means it is not. '''
    actual_set = set(actual)
    return [1 if p in actual_set else 0 for p in predicted[:25]]


def get_metrics(predicted, actual):
    ''' Returns the metrics for one article, given the predicted terms and its actual terms. '''
    return compare_terms(predicted, actual), len(actual)


def precision(metrics):
    total_num_terms = 0
    for terms, _ in metrics:
        total_num_terms += sum(terms)                        
    
    return total_num_terms / (len(metrics[0][0])*len(metrics))


def recall(metrics):
    total_num_terms = 0
    total_actual_len = 0
    for terms, actual_len in metrics:
        total_num_terms += sum(terms)
        total_actual_len += actual_len

    return total_num_terms / total_actual_len


def f_score(metrics):
    p = precision(metrics)
    r = recall(metrics)
    return (2*p*r) / (p+r)


def avg_precision(metric):
    common_terms, actual_len = metric

    total_rank_sum = 0
    for r, is_correct in enumerate(common_terms, start=1):
        correct_up_to_r = sum(common_terms[:r])
        total_rank_sum += is_correct * (correct_up_to_r / r)

    return total_rank_sum / actual_len


def mean_avg_precision(metrics):
    total_avg_precision = 0
    for metric in metrics:
        total_avg_precision += avg_precision(metric)

    return total_avg_precision / len(metrics)


def get_scores(preds, actuals):
    '''
    Takes in two lists:
    preds -- A list of lists of strings that are the predicted terms for each article.
    actuals -- A list of lists of strings that are the corresponding actual terms for each article.

    Returns 4 metrics: precision, recall, f_score, and mean_avg_precision.
    '''
    if len(preds) != len(actuals):
        raise Exception('Length of prediction and actual lists should be the same.')
    
    metrics = [get_metrics(p, a) for p, a in zip(preds, actuals)]
    return precision(metrics), recall(metrics), f_score(metrics), mean_avg_precision(metrics)


def main():
    preds = [
        ['term1', 'term2', 'term3', 'term5', 'term8'], # Predictions for article 1
        ['term2', 'term6', 'term8', 'term10', 'term14'] # Predictions for article 2
    ]

    actuals = [
        ['term1', 'term3', 'term6', 'term8'], # Actual terms for article 1
        ['term1', 'term5', 'term9', 'term11', 'term12'] # Actual terms for article 2
    ]

    precision, recall, f_score, mean_avg_precision = get_scores(preds, actuals)
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F-Score: {}'.format(f_score))
    print('Mean Average Precision: {}'.format(mean_avg_precision))


if __name__ == '__main__':
    main()
