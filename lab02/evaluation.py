from typing import List


def precision_recall(expected_results: List[bool], actual_results: List[bool]) -> (float, float):
    """Compute the precision and recall of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The precision of the predicted results.
        float
            The recall of the predicted results.
    """
    TP: float = 0  # true positives
    FN: float = 0  # false negatives
    FP: float = 0  # false positives
    TN: float = 0  # true negatives

    for expected, actual in zip(expected_results, actual_results):
        if expected == True and actual == True:
            TP += 1
        elif expected == False and actual == True:
            FN += 1
        elif expected == True and actual == False:
            FP += 1
        elif expected == False and actual == False:
            TN += 1

    if (TP == 0): return 0, 0

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return recall, precision


def F1_score(expected_results: List[bool], actual_results: List[bool]) -> float:
    """Compute the F1-score of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The F1-score of the predicted results.
    """
    precision, recall = precision_recall(expected_results, actual_results)
    
    if (precision == 0 or recall == 0): return 0
    
    f_measure = (2 * precision * recall) / (precision + recall)

    return f_measure
