import Levenshtein as lev
from .LF_utils import *

LF_dict = {}


def labeling_function(f):
    lf_name = f.__name__
    LF_dict[lf_name] = f
    return f


def preprocess_title(x):
    x = to_str_lower(x)
    x = re.sub("[^\d\w]", " ", x)
    return x


@labeling_function
def title_overlap(row):
    x, y = apply_to_xy(preprocess_title, row.title_l, row.title_r)
    if x == "nan" or y == "nan":
        return -1
    x, y = set(x.split()), set(y.split())
    if len(x) == 0 or len(y) == 0:
        return -1
    if len(x.intersection(y)) / min(len(x), len(y)) > 0.8:
        return 1
    elif len(x.intersection(y)) / max(len(x), len(y)) < 0.1:
        return -1
    else:
        return 0


import nltk

english_words = set(nltk.corpus.words.words())


def extract_possible_product_codes(x):
    x = x.replace("-", " ")
    x = x.replace(".", " ")
    x = x.replace("/", " ")
    x = re.sub("[^\w\d]", " ", x)
    cand_w = [w for w in nltk.wordpunct_tokenize(x) if w.lower() not in english_words]
    if len(cand_w) == 0:
        return set([])

    def look_like_product_code(w):
        n_upper = sum(list(map(lambda x: x.isupper(), w)))
        n_digits = sum(list(map(lambda x: x.isdigit(), w)))
        if n_upper > 0 and n_digits > 0:
            return True
        elif n_digits > 0 and n_digits < len(w):
            return True
        # elif n_digits > 4:
        #    return True
        # elif n_upper > 4:
        #    return True
        else:
            return False

    res = list(filter(lambda s: look_like_product_code(s), cand_w))
    res_lower = set([s.lower() for s in res])
    return res_lower


@labeling_function
def code_contain(row):
    x, y = apply_to_xy(extract_possible_product_codes, str(row.title_l), str(row.title_r))
    if len(x) == 0 or len(y) == 0:
        return 0
    inter = x.intersection(y)
    if x == inter or y == inter:
        return 0
    else:
        return -1


@labeling_function
def code_diff_small_edit(row):
    x_, y_ = str(row.title_l), str(row.title_r)
    x, y = apply_to_xy(extract_possible_product_codes, str(x_), str(y_))
    if (list(x) != list(y)) and lev.distance(x_.lower(), y_.lower()) < 5:
        return -1
    else:
        return 0


@labeling_function
def code_overlap(row):
    x, y = apply_to_xy(extract_possible_product_codes, str(row.title_l), str(row.title_r))
    if len(x) == 0 or len(y) == 0:
        return 0
    inter = x.intersection(y)
    if len(inter) / min(len(x), len(y)) > 0.5:
        return 1
    elif len(inter) / max(len(x), len(y)) < 0.1:
        return -1
    else:
        return 0


@labeling_function
def title_edit(row):
    x, y = apply_to_xy(preprocess_title, row.title_l, row.title_r)
    if x == "nan" or y == "nan":
        return -1
    if len(x) == 0 or len(y) == 0:
        return -1
    dist = lev.distance(x, y)
    if dist / max(len(x), len(y)) < 0.2:
        return 1
    elif dist / max(len(x), len(y)) > 0.9:
        return -1
    else:
        return 0


@labeling_function
def title_num_overlap(row):
    x, y = apply_to_xy(to_str_lower, row.title_l, row.title_r)
    if x == "nan" or y == "nan":
        return -1
    if len(x) == 0 or len(y) == 0:
        return -1
    x, y = find_pattern_xy(x, y, "\d+")
    x, y = set(x), set(y)
    if len(x) == 0 or len(y) == 0:
        return 0
    inter = x.intersection(y)
    if len(inter) / max(len(x), len(y)) > 0.8:
        return 1
    elif len(inter) / max(len(x), len(y)) < 0.1:
        return -1
    else:
        return 0


@labeling_function
def longest_code_match(row):
    x, y = apply_to_xy(extract_possible_product_codes, str(row.title_l), str(row.title_r))
    if len(x) == 0 or len(y) == 0:
        return 0
    x, y = list(x), list(y)
    x = list(sorted(x, key=lambda it: len(it)))
    y = list(sorted(y, key=lambda it: len(it)))
    if x[-1].lower() == y[-1].lower():
        return 1
    else:
        return 0


@labeling_function
def screen_size_unmatch(row):
    x, y = apply_to_xy(to_str_lower, row.screen_size_l, row.screen_size_r)
    x, y = find_pattern_xy(x, y, "\d+")
    x, y = set(x), set(y)
    if len(x) == 0 or len(y) == 0:
        return 0
    inter = x.intersection(y)
    if len(inter) == 0:
        return -1
    return 0


@labeling_function
def reso_unmatch(row):
    x, y = apply_to_xy(to_str_lower, row.image_resolution_l, row.image_resolution_r)
    x, y = find_pattern_xy(x, y, "\d+")
    x, y = set(x), set(y)
    if len(x) == 0 or len(y) == 0:
        return 0
    inter = x.intersection(y)
    if len(inter) == 0:
        return -1
    return 0


@labeling_function
def zoom_unmatch(row):
    x, y = apply_to_xy(to_str_lower, row.zoom_optical_l, row.zoom_optical_r)
    x, y = find_pattern_xy(x, y, "\d+")
    x, y = set(x), set(y)
    if len(x) == 0 or len(y) == 0:
        return 0
    inter = x.intersection(y)
    if len(inter) == 0:
        return -1
    return 0


@labeling_function
def model_match(row):
    x, y = apply_to_xy(to_str_lower, row.model_l, row.model_r)
    if x == "nan" or y == "nan":
        return 0
    if x == y:
        return 1
    else:
        return -1
