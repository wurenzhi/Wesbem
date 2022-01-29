import multiprocessing
from functools import partial

from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from tqdm import tqdm


def eval_score(y_pred, gt):
    cutoff = 0.5
    pred = (y_pred > cutoff).astype(int)
    score = precision_recall_fscore_support(
        np.squeeze(gt), np.squeeze(pred), average="binary"
    )
    score_list = [it for it in score]
    return score_list[:-1]

def get_gt(cand_pairs, matches):
    matches_set = set(list(map(tuple, matches.values)))
    gt = [pair in matches_set for pair in cand_pairs]
    gt = np.array(gt).astype(int)
    return gt

def apply_lf(lf, LR):
    preds = []
    for i in tqdm(range(LR.shape[0]), desc="[INFO] Applying LF {}".format(lf.__name__)):
        i_row = LR.iloc[i, :]
        pred = lf(i_row)
        preds.append(pred)
    return preds


def apply_parallel(LR, lf_name_2_func):
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    lf_names = [lf_name for lf_name in lf_name_2_func]
    lf_funcs = [lf_name_2_func[lf_name] for lf_name in lf_name_2_func]
    pred_list = pool.map(partial(apply_lf, LR=LR), lf_funcs)
    pool.close()
    pool.terminate()
    pool.join()
    pred_cols = []
    for lf_name, pred in zip(lf_names, pred_list):
        LR[lf_name + "_pred"] = pred
        pred_cols.append(lf_name + "_pred")
    return LR[pred_cols]