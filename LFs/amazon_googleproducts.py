import Levenshtein as lev
from .LF_utils import *

LF_dict = {}


def labeling_function(f):
    lf_name = f.__name__
    LF_dict[lf_name] = f
    return f


def preprocess_title(x):
    x = to_str_lower(x)
    x = re.sub("[\[\(].*[\]\)]", "", x)
    x = re.sub("[^\w\d\s]", " ", x)
    return x


@labeling_function
def title_overlap(row):
    x, y = apply_to_xy(preprocess_title, row.title_l, row.title_r)
    x, y = apply_to_xy(split_by_space, x, y)
    x, y = set(x), set(y)
    score = len(x.intersection(y)) / (min(len(x), len(y)))
    if score < 0.1:
        return -1
    elif score > 0.7 and (min(len(x), len(y))) > 2:
        return 1
    else:
        return 0


@labeling_function
def title_word_overlap(row):
    x, y = apply_to_xy(to_str_lower, row.title_l, row.title_r)
    x = re.findall("\w{3,}", x)
    y = re.findall("\w{3,}", y)
    x, y = set(x), set(y)
    if len(x) == 0 or len(y) == 0:
        return 0
    score = len(x.intersection(y)) / (min(len(x), len(y)))
    if score < 0.1:
        return -1
    elif score > 0.8 and min(len(x), len(y)) >= 2:
        return 1
    else:
        return 0


@labeling_function
def title_contain(row):
    x, y = apply_to_xy(to_str_lower, row.title_l, row.title_r)
    x = re.sub("\.0", "", x)
    y = re.sub("\.0", "", y)
    x, y = apply_to_xy(preprocess_title, x, y)
    return decision_contain(x, y,val_no=0)


@labeling_function
def title_edit(row):
    x, y = apply_to_xy(preprocess_title, row.title_l, row.title_r)
    if lev.distance(x, y) / min(len(x), len(y)) < 0.1:
        return 1
    else:
        return 0


@labeling_function
def des_overlap(row):
    def preprocess_des(x):
        x = to_str_lower(x)
        x = re.sub("[^\w\d\s]", " ", x)
        return x
    x, y = apply_to_xy(preprocess_des, row.description_l, row.description_r)
    if x == "nan" or y == "nan":
        return 0
    x, y = apply_to_xy(split_by_space, x, y)
    x, y = set(x), set(y)
    score = len(x.intersection(y)) / (min(len(x), len(y)))
    if score > 0.8:
        return 1
    else:
        return 0


@labeling_function
def version_unmatch(row):
    x, y = apply_to_xy(preprocess_title, row.title_l, row.title_r)
    vx, vy = find_pattern_xy(x, y, "\sv\s*(\d+)")
    vx = vx + re.findall("\s(\d+)\s", x)
    vy = vy + re.findall("\s(\d+)\s", y)
    x, y = set(vx), set(vy)
    if len(x) == 0 or len(y) == 0:
        return 0
    if len(x.intersection(y)) == 0:
        return -1
    else:
        if (
                len(x.intersection(y)) / min(len(x), len(y)) > 0.7
                and title_word_overlap(row) == 1
        ):
            return 1
        else:
            return 0


def preprocess_manu(x):
    x = to_str_lower(x)
    x = x.replace("hewlett packard", "hp")
    x = x.replace("inc", "")
    x = re.sub("[^\w\d\s]", "", x)
    x = re.sub("\s", "", x)
    return x


@labeling_function
def manu_unmatch(row):
    x, y = apply_to_xy(preprocess_manu, row.manufacturer_l, row.manufacturer_r)
    if x == "nan" or y == "nan":
        return 0
    return decision_contain(x, y)


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
    else:
        return 0


@labeling_function
def os_unmatch(row):
    x, y = apply_to_xy(to_str_lower, row.title_l, row.title_r)
    os_x, os_y = [], []
    for it in ["win", "mac"]:
        if it in x:
            os_x.append(it)
        if it in y:
            os_y.append(it)
    if len(os_x) == 0 or len(os_y) == 0:
        return 0
    if len(set(os_x).intersection(os_y)) == 0:
        return -1
    else:
        return 0


@labeling_function
def win_version_unmatch(row):
    x, y = apply_to_xy(to_str_lower, row.title_l, row.title_r)
    edi_x, edi_y = [], []
    for it in ["windows xp", "vista"]:
        if it in x:
            edi_x.append(it)
        if it in y:
            edi_y.append(it)
    if len(edi_x) == 0 or len(edi_y) == 0:
        return 0
    if len(set(edi_x).intersection(edi_y)) == 0:
        return -1
    else:
        return 0


@labeling_function
def is_upgrade(row):
    x, y = apply_to_xy(to_str_lower, row.title_l, row.title_r)
    cnt = 0
    if "upgrade" in x:
        cnt += 1
    if "upgrade" in y:
        cnt += 1
    if cnt == 1:
        return -1
    else:
        return 0


@labeling_function
def is_old(row):
    x, y = apply_to_xy(to_str_lower, row.title_l, row.title_r)
    cnt = 0
    if "old version" in x:
        cnt += 1
    if "old version" in y:
        cnt += 1
    if cnt == 1:
        return -1
    else:
        return 0


@labeling_function
def is_academic(row):
    x, y = apply_to_xy(to_str_lower, row.title_l, row.title_r)
    cnt = 0
    if "academic" in x or "education " in x:
        cnt += 1
    if "academic" in y or "education " in y:
        cnt += 1
    if cnt == 1:
        return -1
    else:
        return 0


@labeling_function
def office_type_unmatch(row):
    x, y = apply_to_xy(to_str_lower, row.title_l, row.title_r)
    edi_x = []
    edi_y = []
    for it in ["word", "onenote", "excel", "powerpoint"]:
        if it in x:
            edi_x.append(it)
        if it in y:
            edi_y.append(it)
    if len(edi_x) == 0 or len(edi_y) == 0:
        return 0
    if len(set(edi_x).intersection(edi_y)) == 0:
        return -1
    else:
        return 0


@labeling_function
def manu_overlap(row):
    x, y = apply_to_xy(preprocess_title, row.manufacturer_l, row.manufacturer_r)
    x, y = apply_to_xy(split_by_space, x, y)
    x, y = set(x), set(y)
    score = len(x.intersection(y)) / (min(len(x), len(y)))
    if score < 0.1:
        return -1
    elif score > 0.7 and (min(len(x), len(y))) > 2:
        return 1
    else:
        return 0


@labeling_function
def manu_overlap2(row):
    x, y = apply_to_xy(preprocess_title, row.manufacturer_l, row.manufacturer_r)
    x, y = apply_to_xy(split_by_space, x, y)
    x, y = set(x), set(y)
    score = len(x.intersection(y)) / (min(len(x), len(y)))
    if score < 0.1:
        return -1
    elif score > 0.8 and (min(len(x), len(y))) > 2:
        return 1
    else:
        return 0
