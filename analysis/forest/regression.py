import random

from analysis.forest.forest import bootstrap_sample, predict, random_attributes

def variance(targets):
    n = len(targets)
    mean = sum(targets) / n
    return sum((y - mean) * (y - mean) for y in targets) / n

def variance_reduction(parent_targets, children_targets):
    v = variance(parent_targets)
    n = len(parent_targets)
    for x in children_targets:
        if x:
            v -= len(x) / n * variance(x)
    return v

def best_split_point(values, targets, min_leaf=1):
    pairs = sorted(zip(values, targets))
    values, targets = zip(*pairs)
    values = list(values)
    targets = list(targets)
    max_gain = 0
    max_c = None
    for i in range(len(values) - 1):
        if i + 1 >= min_leaf and len(values) - i - 1 >= min_leaf and values[i] != values[i + 1]:
            c = (values[i] + values[i + 1]) / 2
            gain = variance_reduction(targets, [targets[:(i + 1)], targets[(i + 1):]])
            if gain > max_gain:
                max_gain = gain
                max_c = c
    return (max_c, max_gain)

def best_split_point_fast(values, targets, min_leaf=1):
    pairs = sorted(zip(values, targets))
    values, targets = zip(*pairs)
    values = list(values)
    targets = list(targets)
    below_n = 0
    below_sum = 0
    above_n = len(targets)
    above_sum = sum(targets)
    k = above_sum * above_sum / above_n
    max_gain = 0
    max_c = None
    for i in range(len(values) - 1):
        below_n += 1
        above_n -= 1
        below_sum += targets[i]
        above_sum -= targets[i]
        if i + 1 >= min_leaf and len(values) - i - 1 >= min_leaf and values[i] != values[i + 1]:
            c = (values[i] + values[i + 1]) / 2
            gain = (below_sum * below_sum / below_n + above_sum * above_sum / above_n - k) / len(targets)
            if gain > max_gain:
                max_gain = gain
                max_c = c
    return (max_c, max_gain)

def best_attribute(rows, targets, attributes, min_leaf=1):
    best_attribute = None
    best_split = None
    best_gain = 0
    for x in attributes:
        values = [row[x] for row in rows]
        (split, gain) = best_split_point_fast(values, targets, min_leaf)
        if gain > best_gain:
            best_attribute = x
            best_split = split
            best_gain = gain
    return (best_attribute, best_split, best_gain)

def build_tree(rows, targets, pool, m, rng, max_depth, min_leaf, depth=0):
    mean = sum(targets) / len(targets)
    if all(target == targets[0] for target in targets):
        return {"label": mean}
    if max_depth is not None and depth >= max_depth:
        return {"label": mean}
    if m == len(pool):
        attributes = pool
    else:
        attributes = [pool[j] for j in random_attributes(len(pool), m, rng)]
    (best_attr, best_split, best_gain) = best_attribute(rows, targets, attributes, min_leaf)
    if best_attr is None:
        return {"label": mean}
    below_rows, below_targets = [], []
    above_rows, above_targets = [], []
    for i in range(len(rows)):
        if rows[i][best_attr] <= best_split:
            below_rows.append(rows[i])
            below_targets.append(targets[i])
        else:
            above_rows.append(rows[i])
            above_targets.append(targets[i])
    return {"attribute": best_attr,
            "split": best_split,
            "below": build_tree(below_rows, below_targets, pool, m, rng, max_depth, min_leaf, depth + 1),
            "above": build_tree(above_rows, above_targets, pool, m, rng, max_depth, min_leaf, depth + 1)}

def build_forest(rows, targets, n_trees, m, max_depth, min_leaf, sampling, seed):
    ans = []
    rng = random.Random(seed)
    n_attributes = len(rows[0])
    for i in range(n_trees):
        (sample_rows, sample_targets) = bootstrap_sample(rows, targets, rng)
        if sampling == "split":
            pool = list(range(n_attributes))
        elif sampling == "tree":
            pool = random_attributes(n_attributes, m, rng)
        else:
            raise ValueError(f"sampling is {sampling!r}, but it must be \"split\" or \"tree\"")
        ans.append(build_tree(sample_rows, sample_targets, pool, m, rng, max_depth, min_leaf))
    return ans

def predict_forest(forest, row):
    predictions = [predict(tree, row) for tree in forest]
    return sum(predictions) / len(predictions)
