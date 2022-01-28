import Levenshtein as lev
from .utils import *

LF_dict = {}


def labeling_function(f):
    lf_name = f.__name__
    LF_dict[lf_name] = f
    return f


def preprocess_name(x):
    x = to_str_lower(x)
    x = re.sub("[\[\(].*[\]\)]", "", x)
    x = re.sub("[^\w\d\s]", " ", x)
    return x


@labeling_function
def name_overlap(row):
    x, y = apply_to_xy(preprocess_name, row.name_l, row.name_r)
    x, y = apply_to_xy(split_by_space, x, y)
    x, y = set(x), set(y)
    score = len(x.intersection(y)) / (min(len(x), len(y)))
    return decision_by_score(score, 0.1, 0.6)


@labeling_function
def name_word_overlap(row):
    x, y = apply_to_xy(to_str_lower, row.name_l, row.name_r)
    x, y = find_pattern_xy(x, y, "\w{3,}")
    x, y = set(x), set(y)
    if len(x) == 0 or len(y) == 0:
        return 0
    score = len(x.intersection(y)) / (min(len(x), len(y)))
    return decision_by_score(score, 0.1, 0.8)


@labeling_function
def name_number_overlap(row):
    x, y = apply_to_xy(to_str_lower, row.name_l, row.name_r)
    x, y = find_pattern_xy(x, y, "\d+")
    x, y = set(x), set(y)
    if len(x) == 0 or len(y) == 0:
        return 0
    score = len(x.intersection(y)) / (min(len(x), len(y)))
    return decision_by_score(score, 0.1, 0.8)


@labeling_function
def name_contain(row):
    x, y = apply_to_xy(to_str_lower, row.name_l, row.name_r)
    x = re.sub("\.0", "", x)
    y = re.sub("\.0", "", y)
    x, y = apply_to_xy(preprocess_name, x, y)
    return decision_contain(x, y,val_no=0)


@labeling_function
def name_edit(row):
    def preprocess_name(x):
        x = to_str_lower(x)
        x = re.sub("[\[\(].*[\]\)]", "", x)
        x = re.sub("[^\w\d\s]", "", x)
        return x
    x, y = apply_to_xy(preprocess_name, row.name_l, row.name_r)
    if lev.distance(x, y) / min(len(x), len(y)) < 0.1:
        return 1
    elif lev.distance(x, y) / max(len(x), len(y)) > 0.8:
        return -1
    return 0


@labeling_function
def brand_match(row):
    x, y = apply_to_xy(to_str_lower, row.name_l, row.name_r)
    x, y = apply_to_xy(split_by_space, x, y)
    if x[0] != y[0]:
        return -1
    else:
        return 1


@labeling_function
def price_unmatch(row):
    x = re.sub(r"[^\d|\.]", "", str(row.price_l))
    y = re.sub(r"[^\d|\.]", "", str(row.price_r))
    if len(x) == 0 or len(y) == 0:
        return 0
    x, y = float(x), float(y)
    if x == 0 or y == 0:
        return 0
    if max(x / (y + 1e-6), y / (x + 1e-6)) > 5:
        return -1
    elif max(x / (y + 1e-6), y / (x + 1e-6)) < 1.1:
        return 1
    else:
        return 0


@labeling_function
def size_unmatch(row):
    x = str(row.name_l) + " " + str(row.description_l)
    y = str(row.name_r) + " " + str(row.description_r)
    x, y = find_pattern_xy(x, y, "[\d\.]+\'")
    if len(x) == 0 or len(y) == 0:
        return 0
    x, y = x[0], y[0]
    if x != y:
        return -1
    else:
        return 1


@labeling_function
def description_overlap(row):
    x = set(str(row.description_l).split())
    y = set(str(row.description_r).split())
    if "nan" in x or "nan" in y:
        return 0
    score = len(x.intersection(y)) / (min(len(x), len(y)))
    return decision_by_score(score, 0.1, 0.6)


import nltk

english_words = set(nltk.corpus.words.words())


def extract_possible_product_codes(x):
    x = x.replace("-", "")
    x = x.replace(".", "")
    x = x.replace("/", "")
    cand_w = [w for w in nltk.wordpunct_tokenize(x) if w.lower() not in english_words]
    if len(cand_w) == 0:
        return set([])

    def look_like_product_code(w):
        n_upper = sum(list(map(lambda x: x.isupper(), w)))
        n_digits = sum(list(map(lambda x: x.isdigit(), w)))
        if n_upper > 0 and n_digits > 0:
            return True
        elif n_digits > 4:
            return True
        elif n_upper > 4:
            return True
        else:
            return False

    res = list(filter(lambda s: look_like_product_code(s), cand_w))
    res_lower = set([s.lower() for s in res])
    return res_lower


@labeling_function
def compare_product_code_name(row):
    codes_l, codes_r = apply_to_xy(extract_possible_product_codes, str(row.name_l), str(row.name_r))
    intersect = codes_l.intersection(codes_r)
    if len(codes_l) == 0 or len(codes_r) == 0:
        return 0
    elif len(intersect) > 0:
        return 1
    else:
        return -1


@labeling_function
def compare_product_code_desc(row):
    codes_l, codes_r = apply_to_xy(extract_possible_product_codes, str(row.description_l), str(row.description_r))
    intersect = codes_l.intersection(codes_r)
    if len(codes_l) == 0 or len(codes_r) == 0:
        return 0
    score = len(intersect) / min(len(codes_l), len(codes_r))
    return decision_by_score(score, 0.1, 0.5)


def find_code(x):
    x = x + " "
    x = re.sub("[^\w\d\s]", "", x)
    x = re.findall("\s\d*(?:[A-Z]+\d+)+[A-Z]*\s", x)
    x = [it[1:-1] for it in x]
    return x


@labeling_function
def prod_code_name_lf(row):
    x, y = apply_to_xy(find_code, str(row.name_l), str(row.name_r))
    if len(x) == 0 or len(y) == 0:
        return 0
    contain = [True if xi in yi or yi in xi else False for xi in x for yi in y]
    if sum(contain) == 0:
        return -1
    else:
        return 1


@labeling_function
def prod_code_des_lf(row):
    x, y = apply_to_xy(find_code, str(row.description_l), str(row.description_r))
    if len(x) == 0 or len(y) == 0:
        return 0
    contain = [True if xi in yi or yi in xi else False for xi in x for yi in y]
    if sum(contain) == 0:
        return -1
    else:
        return 1
