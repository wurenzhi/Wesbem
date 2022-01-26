import re
import numpy as np


def apply_to_xy(func, x, y):
    return func(x), func(y)


def to_str_lower(x):
    x = str(x).lower()
    return x


def remove_symbols(x):
    x = re.sub("[^\w\d\s]", "", x)
    return x


def remove_digits(x):
    x = re.sub("\d", "", x)
    return x


def keep_digits(x):
    x = re.sub("[^\d]", "", x)
    return x


def split_by_space(x, s=" "):
    x = str(x).split(s)
    return x


def find_pattern_xy(x, y, pattern):
    return re.findall(pattern, x), re.findall(pattern, y)


def preprocess_simple(x):
    x = to_str_lower(x)
    x = remove_symbols(x)
    return x


def decision_contain(x, y, val_is=1, val_no=-1):
    if len(x) == 0 or len(y) == 0:
        return 0
    if x in y or y in x:
        return val_is
    else:
        return val_no


def decision_by_score(score, th_l, th_h, inverse=False):
    val = 1
    if inverse:
        val = -1
    if score < th_l:
        return -val
    elif score > th_h:
        return val
    else:
        return 0


def decision_by_equal(x, y, val_equal=1, val_unequal=-1):
    if x == y:
        return val_equal
    else:
        return -val_unequal


def overlap_score(row, attr):
    x, y = apply_to_xy(to_str_lower, row[attr + "_l"], row[attr + "_r"])
    if x == "nan" or y == "nan":
        return np.nan, np.nan
    elif len(x) == 0 or len(y) == 0:
        return np.nan, np.nan
    x = re.sub("[\[\(].*[\]\)]", "", x)
    y = re.sub("[\[\(].*[\]\)]", "", y)
    x = re.sub("[^\w\d\s]", " ", x)
    y = re.sub("[^\w\d\s]", " ", y)
    x, y = split_by_space(x, y)
    x = set(x)
    y = set(y)
    intersect = x.intersection(y)
    score = len(intersect) / (min(len(x), len(y)))
    score2 = len(intersect) / (max(len(x), len(y)))
    return score, score2, len(x), len(y)
