import Levenshtein as lev
from .LF_utils import *

LF_dict = {}


def labeling_function(f):
    lf_name = f.__name__
    LF_dict[lf_name] = f
    return f


def preprocess_author(x):
    x = to_str_lower(x)
    x = re.sub("[^\w\d\s,]", "", x)
    x = x.strip()
    x = x.split(",")
    x = [it.strip() for it in x]
    x = [it for it in x if len(it) > 0]
    for i in range(len(x)):
        wrds = x[i].split()
        wrds = [it[0] for it in wrds]
        if len(wrds) > 2:
            wrds = [wrds[0], wrds[-1]]
        x[i] = " ".join(sorted(wrds))
    x = sorted(x)
    x = "#".join(x)
    return x


def author_score(row):
    x, y = apply_to_xy(to_str_lower, row.authors_l, row.authors_r)
    if x == "nan" or y == "nan":
        return np.nan, np.nan
    elif len(x) == 0 or len(y) == 0:
        return np.nan, np.nan
    else:
        x, y = apply_to_xy(preprocess_author, x, y)
        x = x.split("#")
        y = y.split("#")
        x = set(x)
        y = set(y)
        intersect = x.intersection(y)
        return len(intersect) / (min(len(x), len(y))), len(intersect) / (
            max(len(x), len(y))
        )


@labeling_function
def author_overlap(row):
    score, score_max = author_score(row)
    return decision_by_score(score, 0.1, 0.8)


def preprocess_venue(x):
    x = to_str_lower(x)
    x = re.sub("[\[\(].*[\]\)]", "", x)
    dic = {
        "International Conference on Management of Data": "SIGMOD Conference",
        "very large data bases": "vldb",
        " J.": " Journal",
        "ACM Trans. Database Syst.": "TODS",
        "ACM Transactions on Database Systems": "TODS",
    }
    for key, val in dic.items():
        x = x.replace(key.lower(), val.lower())
    return x


@labeling_function
def venue_overlap(row):
    x, y = apply_to_xy(preprocess_venue, row.venue_l, row.venue_r)
    if x == "nan" or y == "nan":
        return 0
    if len(x) == 0 or len(y) == 0:
        return 0
    x = re.sub("[^\w\d\s]", " ", x)
    y = re.sub("[^\w\d\s]", " ", y)
    x, y = apply_to_xy(split_by_space, x, y)
    x = set(x)
    y = set(y)
    score = len(x.intersection(y)) / (min(len(x), len(y)))
    return decision_by_score(score, 0.1, 0.7)


@labeling_function
def title_equal_year_equal(row):
    x, y = apply_to_xy(to_str_lower, row.title_l, row.title_r)
    if x == y and row.year_l == row.year_r:
        return 1
    else:
        return 0


def preprocess_title(x):
    x = to_str_lower(x)
    x = x.replace("- book review", "")
    x = re.sub("[\[\(].*[\]\)]", "", x)
    x = re.sub("[^\w\d\s]", " ", x)
    return x


def title_overlap_score(row):
    x, y = apply_to_xy(preprocess_title, row.title_l, row.title_r)
    x, y = apply_to_xy(split_by_space, x, y)
    x = set(x)
    y = set(y)
    score = len(x.intersection(y)) / (min(len(x), len(y)))
    return score, x, y


@labeling_function
def title_overlap(row):
    score, x, y = title_overlap_score(row)
    if score < 0.5:
        return -1
    elif score > 0.9 and (min(len(x), len(y))) > 2:
        return 1
    else:
        return 0


@labeling_function
def title_overlap_year_equal(row):
    score, x, y = title_overlap_score(row)
    if score < 0.6:
        return -1
    elif score > 0.8 and (min(len(x), len(y))) > 2 and row.year_l == row.year_r:
        return 1
    else:
        if row.year_l != row.year_r:
            return -1
        return 0

@labeling_function
def title_edit(row):
    def preprocess_title(x):
        x = to_str_lower(x)
        x = x.replace("- book review", "")
        x = re.sub("[\[\(].*[\]\)]", "", x)
        x = re.sub("[^\w\d\s]", "", x)
        return x
    x, y = apply_to_xy(preprocess_title, row.title_l, row.title_r)
    if len(x) == 0 or len(y) == 0:
        return 0
    dist = lev.distance(x, y) / min(len(x), len(y))
    dist2 = lev.distance(x, y) / max(len(x), len(y))
    if dist < 0.1:
        return 1
    else:
        if dist2 > 0.7:
            return -1
        return 0


@labeling_function
def year_unmatch(row):
    return decision_by_equal(row.year_l, row.year_r)


@labeling_function
def book_review(row):
    x, y = apply_to_xy(to_str_lower, row.title_l, row.title_r)
    if "book review column" in x and "book review column" in y:
        if row.year_l != row.year_r:
            return -1
    return 0


@labeling_function
def venue_unmatch(row):
    x, y = apply_to_xy(preprocess_venue, row.venue_l, row.venue_r)
    if x == "nan" or y == "nan":
        return 0
    if len(x) == 0 or len(y) == 0:
        return 0
    for item in ["journal", "record"]:
        if sum([item in x, item in y]) == 1:
            return -1
    if x in y or y in x:
        return 1
    else:
        return -1


@labeling_function
def author_overlap(row):
    x, y = apply_to_xy(preprocess_title, row.authors_l, row.authors_r)
    if x == "nan" or y == "nan":
        return 0
    elif len(x) == 0 or len(y) == 0:
        return 0
    x, y = apply_to_xy(split_by_space, x, y)
    x, y = set(x), set(y)
    score = len(x.intersection(y)) / (min(len(x), len(y)))
    return decision_by_score(score, 0.2, 0.7)


@labeling_function
def year_small_diff(row):
    if abs(row.year_l - row.year_r) > 3:
        return -1
    else:
        return 1


@labeling_function
def year_small_diff2(row):
    if abs(row.year_l - row.year_r) > 2:
        return -1
    else:
        return 1

@labeling_function
def author_unmatch(row):
    x, y = apply_to_xy(to_str_lower, row.authors_l, row.authors_r)
    if x == "nan" or y == "nan":
        return 0
    elif len(x) == 0 or len(y) == 0:
        return 0
    x, y = apply_to_xy(preprocess_author, x, y)
    if lev.distance(x, y) < 0.1 / min(len(x), len(y)):
        return 1
    else:
        x = x.split("#")
        y = y.split("#")
        x = set(x)
        y = set(y)
        intersect = x.intersection(y)
        score = len(intersect) / (max(len(x), len(y)))
        if score < 0.1:
            return -1
        elif score > 0.7:
            return 1
        else:
            return 0
