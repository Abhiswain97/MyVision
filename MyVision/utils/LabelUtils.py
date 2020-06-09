import pandas as pd
import numpy as np


def categorify(df, label_column, start_from):
    """
    Takes in a pandas dataframe, the label column & a start index.
    makes the categorical labels into integers
    You can specify which index to start labelling from 0 or 1.
    returns a dictionary mapping the label to it's integer value.

    :param df: pd.Dataframe
    :param label_column: str
    :param start_from: int(0 or 1)

    :returns: 
        Dictionary of label to integer class mapping.
    """
    if all(isinstance(item, str) for item in df[label_column].values):
        print('Converting to integer labels...')
        
        label_dict = {}
        for k, v in enumerate(df[label_column].unique(), start=start_from):
            label_dict[v] = k

        return label_dict
    else:
        assert all(isinstance(item, int) for item in df[label_column].values), "Labels already in integer format"



def probs_to_labels(probs_, thresh=0.5):
    """
    Takes in a list of probabilities and makes them into labels.
    Takes care of multiclass and binary labels automatically.
    For binary labels, set a threshold probability for classification.

    :param: probs_: list
    :param: thresh: int

    :returns:
        List of predicted classes.
    """
    if all(isinstance(item, float) for item in probs_):
        raise ValueError("Expects float probability values and not integer values")
    if len(probs_) > 1:
        pred_labels = [np.argmax(np.array(p)) for p in probs_]
    else:
        pred_labels = [1 if p > thresh else 0 for p in probs_]

    return pred_labels
