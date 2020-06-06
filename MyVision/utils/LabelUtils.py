import pandas as pd


def make_multiclass_labels(df, label_column, start_from):
    """
    Takes in a pandas dataframe, the label column & a start index.
    makes the categorical labels into integers
    You can specify which index to start labelling from 0 or 1.
    returns a dictionary mapping the label to it's integer value.

    :param df: pd.Dataframe
    :param label_column: str
    :param start_from: int(0 or 1)

    :return: Dict
    """
    label_dict = {}
    for k, v in enumerate(df[label_column].unique(), start=start_from):
        label_dict[v] = k

    return label_dict
