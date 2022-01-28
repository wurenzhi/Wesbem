import Levenshtein as lev
from .utils import *

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


def year_score(row):
    x = row.year_l
    y = row.year_r
    return abs(x - y)


def preprocess_title(x):
    x = to_str_lower(x)
    x = re.sub("[\[\(].*[\]\)]", "", x)
    x = re.sub("[^\w\d\s]", "", x)
    return x


@labeling_function
def title_equal(row):
    x, y = apply_to_xy(preprocess_title, row.title_l, row.title_r)
    if x == y:
        if len(x.split()) > 3 and len(y.split()) > 3:
            return 1
    else:
        return 0


@labeling_function
def title_contain(row):
    x, y = apply_to_xy(preprocess_title, row.title_l, row.title_r)
    if x in y or y in x:
        if len(x.split()) > 3 and len(y.split()) > 3:
            return 1
    else:
        return 0


@labeling_function
def title_contain_edit_distance(row):
    x, y = apply_to_xy(to_str_lower, row.title_l, row.title_r)
    if x == "nan" or y == "nan":
        return 0
    elif len(x) == 0 or len(y) == 0:
        return 0
    x, y = apply_to_xy(preprocess_title, x, y)
    if len(x.split()) > 3 and len(y.split()) > 3:
        if x in y or y in x:
            return 1
        if lev.distance(x, y) < 0.1 * min(len(x), len(y)):
            return 1
    return 0


@labeling_function
def title_contain_year_close(row):
    x, y = apply_to_xy(preprocess_title, row.title_l, row.title_r)
    if len(x.split()) > 3 and len(y.split()) > 3:
        if x in y or y in x:
            if abs(row.year_l - row.year_r) <= 3:
                return 1
    return 0


@labeling_function
def title_overlap(row):
    score, score1, lenx, leny = overlap_score(row, "title")
    if score < 0.5:
        return -1
    elif score > 0.7 and score1 > 0.5:
        if lenx > 3 and leny > 3:
            return 1
    else:
        return 0


@labeling_function
def title_author_overlap(row):
    score, score1, lenx, leny = overlap_score(row, "title")
    author, author1 = author_score(row)
    if score < 0.5 and author < 0.5:
        return -1
    elif score1 > 0.6:
        if author1 > 0.7:
            return 1
        return 0
    else:
        return 0


@labeling_function
def title_contain_other_unmatch(row):
    title, title1, lenx, leny = overlap_score(row, "title")
    author, author1 = author_score(row)
    venue, venue1 = venue_score(row)
    year = year_score(row)
    if year == -1:
        return -1
    if title == 1 and title1 < 1:
        if np.isnan(author) and np.isnan(venue) and np.isnan(year):
            return -1
        if author < 0.1:
            return -1
        if venue1 < 0.1:
            return -1
    return 0


@labeling_function
def title_overlap_year_equal(row):
    score, score1, lenx, leny = overlap_score(row, "title")
    year = year_score(row)
    if year > 3:
        return -1
    if score < 0.5:
        return -1
    elif score > 0.7 and score1 > 0.3:
        if year < 2 and lenx > 3 and leny > 3:
            return 1
        return 0
    else:
        return 0


@labeling_function
def title_edit(row):
    x, y = apply_to_xy(to_str_lower, row.title_l, row.title_r)
    x = x.replace("- book review", "")
    y = y.replace("- book review", "")
    if len(x) == 0 or len(y) == 0:
        return 0
    x, y = apply_to_xy(preprocess_title, x, y)
    dist = lev.distance(x, y) / min(len(x), len(y))
    dist2 = lev.distance(x, y) / max(len(x), len(y))
    if dist < 0.2 and min(len(x.split()), len(y.split())) > 2:
        return 1
    else:
        if dist2 > 0.6:
            return -1
        return 0


@labeling_function
def book_review(row):
    x, y = apply_to_xy(to_str_lower, row.title_l, row.title_r)
    if "book review column" in x and "book review column" in y:
        if row.year_l != row.year_r:
            return -1
        if row.year_l == row.year_r:
            x, y = apply_to_xy(to_str_lower, row.venue_l, row.venue_r)
            if x in y or y in x:
                return 1
    return 0


@labeling_function
def year_unmatch(row):
    score = year_score(row)
    return decision_by_score(score, 3.5, 3.5, inverse=True)


@labeling_function
def year_unmatch2(row):  # adversarial LF
    score = year_score(row)
    return decision_by_score(score, 2.5, 2.5, inverse=True)


@labeling_function
def year_unmatch3(row):  # adversarial LF
    score = year_score(row)
    return decision_by_score(score, 4.5, 4.5, inverse=True)


@labeling_function
def year_unmatch4(row):  # adversarial LF
    score = year_score(row)
    return decision_by_score(score, 4.5, 4.5, inverse=True)


@labeling_function
def title_author_venue_year(row):
    title, title1, lenx, leny = overlap_score(row, "title")
    author, author1 = author_score(row)
    venue, venue1 = venue_score(row)
    year = year_score(row)
    if year > 3:
        return -1
    if (title > 0.5 and author > 0.5 and venue > 0.5 and venue1 > 0.1):
        return 1
    if title > 0.8 and author > 0.7:
        return 1
    if title < 0.1 or author1 < 0.3:
        return -1
    if np.isnan(author) and np.isnan(venue) and np.isnan(year) and title1 < 1:
        return -1
    return 0


@labeling_function
def venue_overlap(row):
    score, score1 = venue_score(row)
    if score < 0.5:
        return -1
    if score1 < 0.2:
        return -1
    elif score > 0.7 and score > 0.7:
        return 1
    else:
        return 0


def preprocess_venue(x):
    x = to_str_lower(x)
    x = re.sub("[\[\(].*[\]\)]", "", x)
    dic = {
        "vldb": "proceedings of the international conference on very large Database Systems",
        "int.": "international",
        "J.": "journal",
        " J ": " journal ",
        "Conf.": "conference",
        "Conf ": "conference ",
        "Trans.": "Transactions",
        "Syst.": "Systems",
        "SIGMOD": "International Conference on Management of Data",
        "Procs.": "proceedings",
        "Proc ": "proceedings ",
        "Tech.": "technical",
        "TODS": "ACM Transactions on Database Systems",
    }
    for key, val in dic.items():
        x = x.replace(key.lower(), val.lower())
    return x


venue_unk_keywords = [
    "publication",
    "stanford",
    "report",
    "university",
    "dallas",
    "http://www",
    "working",
    "description",
    "ftp",
    "new york",
    "institut",
    "canada",
    "thesis",
    "ibm",
    "submitted",
    "switzerland",
]
venue_keywords = [
    "workshop",
    "very large",
    "management of data",
    "transactions",
    "journal",
]


def venue_score(row):
    x, y = apply_to_xy(to_str_lower, row.venue_l, row.venue_r)
    if x == "nan" or y == "nan":
        return np.nan, np.nan
    elif len(x) == 0 or len(y) == 0:
        return np.nan, np.nan
    else:
        x, y = apply_to_xy(preprocess_venue, x, y)
        for wrd in venue_unk_keywords:
            if wrd in x or wrd in y:
                return np.nan, np.nan
        for key in venue_keywords:
            if sum([key in x, key in y]) == 1:
                return -1, -1

        def process(x):
            x = re.sub("[^\w\d\s]", " ", x)
            words2remove = [
                "proceedings",
                "of",
                "the",
                "acm",
                "conference",
                "international",
            ]
            for wrd in words2remove:
                x = x.replace(wrd, "")
            x = x.strip()
            x = x.split()
            x = set(x)
            return x

        x, y = apply_to_xy(process, x, y)

        if len(x) == 0 or len(y) == 0:
            return np.nan, np.nan
        intersect = x.intersection(y)
        return len(intersect) / (min(len(x), len(y))), len(intersect) / (
            max(len(x), len(y))
        )


@labeling_function
def venue_unmatch(row):
    x, y = apply_to_xy(to_str_lower, row.venue_l, row.venue_r)
    if x == "nan" or y == "nan":
        return 0
    if len(x) == 0 or len(y) == 0:
        return 0
    x, y = apply_to_xy(preprocess_venue, x, y)
    for wrd in venue_unk_keywords:
        if wrd in x or wrd in y:
            return 0
    for key in venue_keywords:
        if sum([key in x, key in y]) == 1:
            return -1
    return 0


@labeling_function
def author_overlap_unmatch(row):
    author, author1 = author_score(row)
    if author < 0.3:
        return -1
    elif author > 0.7 and author1 > 0.7:
        return 1
    return 0


@labeling_function
def author_unmatch(row):
    x, y = apply_to_xy(to_str_lower, row.authors_l, row.authors_r)
    if x == "nan" or y == "nan":
        return 0
    if len(x) == 0 or len(y) == 0:
        return 0
    x, y = apply_to_xy(preprocess_author, x, y)
    lev_dist = lev.distance(x, y)

    title, title1, lenx, leny = overlap_score(row, "title")
    if lev_dist < 0.1 * min(len(x), len(y)):
        if title > 0.5:
            return 1
    elif lev_dist > 0.9 * min(len(x), len(y)):
        return -1
    else:
        x = x.split("#")
        y = y.split("#")
        x, y = set(x),set(y)
        score = len(x.intersection(y)) / (max(len(x), len(y)))
        if score < 0.1:
            return -1
        elif score > 0.7:
            if title > 0.5:
                return 1
        else:
            return 0
    return 0
