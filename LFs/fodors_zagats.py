import re
import Levenshtein as lev

from .utils import *

LF_dict = {}


def labeling_function(f):
    lf_name = f.__name__
    LF_dict[lf_name] = f
    return f


@labeling_function
def city_contain(row):
    x, y = apply_to_xy(preprocess_simple, row.city_l, row.city_r)
    return decision_contain(x, y)


@labeling_function
def name_edit(row):
    x, y = apply_to_xy(preprocess_simple, row.name_l, row.name_r)
    if len(x) == 0 or len(y) == 0:
        return 0
    score = lev.distance(x, y) / max(len(x), len(y))
    return decision_by_score(score, 0.1, 0.8, inverse=True)


@labeling_function
def name_contain(row):
    x, y = apply_to_xy(preprocess_simple, row.name_l, row.name_r)
    return decision_contain(x, y, val_no=0)


@labeling_function
def name_unmatch(row):
    x, y = apply_to_xy(preprocess_simple, row.name_l, row.name_r)
    if len(x) == 0 or len(y) == 0:
        return 0
    x, y = apply_to_xy(split_by_space, x, y)
    if len(set(x).intersection(y)) == 0:
        return -1
    return 0


def preprocess_addr(x):
    x = preprocess_simple(x)
    x = x.replace("road", "rd")
    x = x.replace("ninth", "9th")
    return x


@labeling_function
def addr_contain(row):
    x, y = apply_to_xy(preprocess_addr, row.addr_l, row.addr_r)
    return decision_contain(x, y, val_no=0)


@labeling_function
def addr_edit(row):
    x, y = apply_to_xy(preprocess_addr, row.addr_l, row.addr_r)
    x, y = apply_to_xy(remove_digits, x, y)
    edit = lev.distance(x, y)
    if edit == 1:
        return 1
    return 0


@labeling_function
def addr_num_match(row):
    x, y = apply_to_xy(to_str_lower, row.addr_l, row.addr_r)
    x = re.findall("\d{3,}", x)
    y = re.findall("\d{3,}", y)
    if len(x) == 0 or len(y) == 0:
        return 0
    if len(set(x).intersection(y)) == 0:
        return -1
    else:
        return 1


@labeling_function
def type_match(row):
    x, y = apply_to_xy(to_str_lower, row.type__l, row.type__r)
    return decision_contain(x, y)


@labeling_function
def addr_unmatch(row):
    x, y = apply_to_xy(preprocess_simple, row.addr_l, row.addr_r)
    if len(x) == 0 or len(y) == 0:
        return 0
    x, y = apply_to_xy(split_by_space, x, y)
    if len(set(x).intersection(y)) == 0:
        return -1
    return 0

@labeling_function
def phone_match(row):
    x, y = apply_to_xy(to_str_lower, row.phone_l, row.phone_r)
    x, y = apply_to_xy(keep_digits, x, y)
    if len(x) == 0 or len(y) == 0:
        return 0
    return decision_by_equal(x, y)


@labeling_function
def city_unmatch(row):
    x, y = apply_to_xy(preprocess_simple, row.city_l, row.city_r)
    if len(x) == 0 or len(y) == 0:
        return 0
    x, y = apply_to_xy(split_by_space, x, y)
    if len(set(x).intersection(y)) == 0:
        return -1
    return 0

@labeling_function
def phone_area_code_unmatch(row):
    x, y = apply_to_xy(to_str_lower, row.phone_l, row.phone_r)
    x = re.sub("[^\d]", " ", x)
    y = re.sub("[^\d]", " ", y)
    x, y = apply_to_xy(split_by_space, x, y)
    if x[0] == y[0]:
        return 0
    return -1
