import os

import numpy as np
from LFs.dblp_acm import LF_dict as lfs_da
from LFs.abt_buy import LF_dict as lfs_ab
from LFs.amazon_googleproducts import LF_dict as lfs_ag
from LFs.camera import LF_dict as lfs_c
from LFs.fodors_zagats import LF_dict as lfs_fz
from LFs.dblp_scholar import LF_dict as lfs_ds
from LFs.monitor import LF_dict as lfs_m
import pandas as pd

from helpers import apply_parallel, get_gt, eval_score

dataset_lf = {"fodors_zagats":lfs_fz,"dblp_acm":lfs_da, "dblp_scholar":lfs_ds,
              "amazon_googleproducts":lfs_ag,"abt_buy":lfs_ab,"camera":lfs_c, "monitor":lfs_m}


def majority_vote(l_matrix):
    preds_prob_mv = np.mean(l_matrix, axis=1)
    y_pred = (1 + preds_prob_mv) / 2
    return y_pred


dataset = "fodors_zagats"
#read the candidate set
cand = pd.read_csv(os.path.join("data",dataset,"cache_cand.csv"))

#apply lfs to candidate set
from LFs.fodors_zagats import LF_dict
preds =  apply_parallel(cand,LF_dict)

#infer gt labels with a model
y_pred = majority_vote(preds.values)

#get the ground labels
matches = pd.read_csv(os.path.join("data", dataset, "matches.csv"))
cand_pairs = list(map(tuple, cand[["id_l", "id_r"]].values))
y_gt = get_gt(cand_pairs, matches)

#evaluate the inferred labels
scores = eval_score(y_pred, y_gt)
print(scores)
