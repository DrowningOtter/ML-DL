import numpy as np
import typing
from collections import defaultdict


def kfold_split(num_objects: int,
                num_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds
       (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects: number of objects in train set
    num_folds: number of folds for cross-validation split

    Returns:
    list of length num_folds, where i-th element of list
    contains tuple of 2 numpy arrays, he 1st numpy array
    contains all indexes without i-th fold while the 2nd
    one contains i-th fold
    """
    step = num_objects // num_folds
    ret_list = []
    for i in range(1, num_folds):
        if True:
            first = np.hstack((np.arange(0, (i - 1) * step), np.arange(i * step, num_objects)))
            second = np.arange((i - 1) * step, i * step)
        else:
            first = np.arange(step, num_objects)
            second = np.arange(0, step)
        ret_list.append((first, second))
    if step * (num_folds - 1) < num_objects:
        first = np.arange(0, (num_folds - 1) * step)
        second = np.arange((num_folds - 1) * step, num_objects)
        ret_list.append((first, second))
    return ret_list


def knn_cv_score(X: np.ndarray, y: np.ndarray, parameters: dict[str, list],
                 score_function: callable,
                 folds: list[tuple[np.ndarray, np.ndarray]],
                 knn_class: object) -> dict[str, float]:
    """Takes train data, counts cross-validation score over
    grid of parameters (all possible parameters combinations)

    Parameters:
    X: train set
    y: train labels
    parameters: dict with keys from
        {n_neighbors, metrics, weights, normalizers}, values of type list,
        parameters['normalizers'] contains tuples (normalizer, normalizer_name)
        see parameters example in your jupyter notebook

    score_function: function with input (y_true, y_predict)
        which outputs score metric
    folds: output of kfold_split
    knn_class: class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight),
    value - mean score over all folds
    """
    ret_dict = {}
    for nnei in parameters['n_neighbors']:
        for metr in parameters['metrics']:
            for wei in parameters['weights']:
                for norm in parameters['normalizers']:
                    results = np.zeros(len(folds))
                    for i in range(len(folds)):
                        current_fold = folds[i]
                        train_data = X[current_fold[0]]
                        train_data_output = y[current_fold[0]]
                        validation_data = X[current_fold[1]]
                        validation_data_output = y[current_fold[1]]
                        current_model = knn_class(n_neighbors=nnei, weights=wei, metric=metr, n_jobs=-1)
                        if norm[0] is not None:
                            norm[0].fit(train_data)
                            train_data = norm[0].transform(train_data)
                            validation_data = norm[0].transform(validation_data)
                        current_model.fit(train_data, train_data_output)
                        pred = current_model.predict(validation_data)
                        results[i] = score_function(validation_data_output, pred)
                    ret_dict[(norm[1], nnei, metr, wei)] = results.mean()
    return ret_dict
