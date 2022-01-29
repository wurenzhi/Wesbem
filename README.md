# Wesbem: Weak Supervision Benchmark for Entity Matching
This repository provides labeling functions for seven common entity matching datasets and helper functions to evaluate different labeling models (truth inference methods).

## How to use
1. Read the candidate set of one dataset e.g. fodors_zagats.
```python
dataset = "fodors_zagats"
cand = pd.read_csv(os.path.join("data",dataset,"cache_cand.csv"))
```

2. Apply lfs to candidate set
```python
from LFs.fodors_zagats import LF_dict
preds =  apply_parallel(cand,LF_dict)
```

3. Infer gt labels with a labeling model/truth inference method
```python
y_pred = majority_vote(preds.values)
```
You can replace majority_vote with the model you want to evaluate.

4. Get the ground labels
```python
matches = pd.read_csv(os.path.join("data", dataset, "matches.csv"))
cand_pairs = list(map(tuple, cand[["id_l", "id_r"]].values))
y_gt = get_gt(cand_pairs, matches)
```

5. Evaluate the infered labels
```python
scores = eval_score(y_pred, y_gt)
print(scores)
```

