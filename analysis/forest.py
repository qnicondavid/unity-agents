# The comments in the code are based on my (ND Milandru) iai lecture notes.

import math
import random

def count_labels(labels):
    ans = {}
    for x in labels:
        ans[x] = ans.get(x, 0) + 1
    return ans

# Entropy (measures the impurity of S, between 0 and 1)  [Lecture 04, slides 14-15]
# - S: sample of training examples
# For n classes:
#     E(S) = -sum_{i=1..n} p_i log_n p_i
def entropy(counts, base):
    ans = 0
    total = sum(counts)
    for x in counts:
        if x:
            p = x / total
            ans = ans - p * (math.log(p) / math.log(base))
    return ans

# Information gain is the expected reduction in entropy caused by partitioning
# the instances from S according to a given attribute A.  [Lecture 04, slide 18]
#     Gain(S, A) = E(S) - sum over v in Values(A) of |S_v|/|S| * E(S_v)
# - |S_v|/|S|: the fraction of S that goes down branch v
# In words: gain = entropy before the split minus the weighted average entropy after it.
def information_gain(parent_counts, children_counts, base):
    ans = entropy(parent_counts, base)
    total = sum(parent_counts)
    for x in children_counts:
        if x:
            ans = ans - (sum(x) / total) * entropy(x, base)
    return ans

# Decision trees with real values  [Lecture 04, slides 30-31]
# - if we want to build a tree with numerical values, we first need to turn that into discrete data
# - we generally use 2 classes, x > a and x <= a
# Candidate split points = midpoints between neighbouring values
# Pick the highest gain
# Reference version: recounts both sides for every candidate split (slow).
# Kept to check best_split_point_fast; not used by the tree.
def best_split_point(values, labels, base):
    pairs = sorted(zip(values, labels))
    values, labels = zip(*pairs)
    values = list(values)
    labels = list(labels)

    parent_counts = list(count_labels(labels).values())

    best_split = None
    best_gain = -1.0

    for i in range(len(values) - 1):
        if values[i] != values[i + 1]:
            split = (values[i] + values[i + 1]) / 2

            children_counts = []
            children_counts.append(list(count_labels(labels[:(i + 1)]).values()))
            children_counts.append(list(count_labels(labels[(i + 1):]).values()))

            ig = information_gain(parent_counts, children_counts, base)
            if ig > best_gain:
                best_gain = ig
                best_split = split

    if best_split is None:
        return (None, 0)
    return (best_split, best_gain)


# Same split as best_split_point, found in one sweep over the sorted values:
# running class counts below / above the candidate (+1 below, -1 above), about 100 times faster.
def best_split_point_fast(values, labels, base):
    pairs = sorted(zip(values, labels))
    values, labels = zip(*pairs)
    values = list(values)
    labels = list(labels)

    parent_counts = list(count_labels(labels).values())

    best_split = None
    best_gain = -1.0

    above_counts = count_labels(labels)
    below_counts = {c: 0 for c in above_counts}

    for i in range(len(values) - 1):
        below_counts[labels[i]] += 1
        above_counts[labels[i]] -= 1

        if values[i] != values[i + 1]:
            split = (values[i] + values[i + 1]) / 2

            children_counts = []
            children_counts.append(list(below_counts.values()))
            children_counts.append(list(above_counts.values()))

            ig = information_gain(parent_counts, children_counts, base)
            if ig > best_gain:
                best_gain = ig
                best_split = split

    if best_split is None:
        return (None, 0)
    return (best_split, best_gain)

# Decision Tree Learning Algorithm  [Lecture 04, slide 12]
# 1. A <- the best decision attribute for a node N
# The attribute with the highest gain is the "best" attribute in step 1 of the algorithm.  [slide 18]
def best_attribute(rows, labels, attributes, base):
    best_attr = None
    best_split = None
    best_gain = 0
    for attribute in attributes:
        current_values = []
        for x in rows:
            current_values.append(x[attribute])
        (best_current_split, best_current_gain) = best_split_point_fast(current_values, labels, base)
        if best_current_gain > best_gain:
            best_attr = attribute
            best_split = best_current_split
            best_gain = best_current_gain
    return (best_attr, best_split, best_gain)



# Decision Tree Learning Algorithm  [Lecture 04, slide 12]
# 1. A <- the best decision attribute for a node N
# 2. A becomes the current question
# 3. Create a child for each value of A
# 4. Sort training examples to leaf nodes
# 5. If training examples perfectly classified: then stop
#    else: iterate over new leaf nodes
def build_tree(rows, labels, attributes, base, max_depth=None, depth=0):
    d = count_labels(labels)
    if len(d) == 1:
        return {"label": labels[0]}

    # Problems with trees  [Lecture 04, slide 39]
    # - single trees can easily overfit: specific and won't generalise to new data
    # - errors in the data get their own special rules, because the algorithm keeps
    #   splitting until all the training data is classified correctly
    if max_depth is not None and depth >= max_depth:
        return {"label": max(d, key=d.get)}

    (best_attr, best_split, best_gain) = best_attribute(rows, labels, attributes, base)

    if best_attr is None:
        return {"label": max(d, key=d.get)}

    below_rows, below_labels = [], []
    above_rows, above_labels = [], []

    for i in range(len(rows)):
        if rows[i][best_attr] <= best_split:
            below_rows.append(rows[i])
            below_labels.append(labels[i])
        else:
            above_rows.append(rows[i])
            above_labels.append(labels[i])

    return {"attribute": best_attr,
            "split": best_split,
            "below": build_tree(below_rows, below_labels, attributes, base, max_depth, depth + 1),
            "above": build_tree(above_rows, above_labels, attributes, base, max_depth, depth + 1)}

# Decision Tree  [Lecture 04, slide 7]
# - each internal node tests an attribute
# - each branch corresponds to an attribute value
# - leaf node = class node
def predict(tree, row):
    if "label" in tree:
        return tree["label"]
    curr_attr = tree["attribute"]
    curr_split = tree["split"]

    if row[curr_attr] <= curr_split:
        return predict(tree["below"], row)
    return predict(tree["above"], row)

# Problems with trees  [Lecture 04, slide 39]
# - our tree building algorithm is deterministic (same data gives the same tree)
# From trees to forests  [Lecture 04, slide 41]
# - to make many trees from one dataset, use subsampling, called bagging or bootstrapping
def bootstrap_sample(rows, labels, rng):
    n = len(rows)
    sample_rows, sample_labels = [], []
    for i in range(n):
        drawn = rng.randrange(n)
        sample_rows.append(rows[drawn])
        sample_labels.append(labels[drawn])
    return (sample_rows, sample_labels)

# - each tree is built from a random subset of the examples and a random subset
#   of the attributes  [Lecture 04, slide 41]
def random_attributes(n_attributes, m, rng):
    attributes = []
    for i in range(m):
        drawn = rng.randrange(n_attributes)
        while drawn in attributes:
            drawn = rng.randrange(n_attributes)
        attributes.append(drawn)
    attributes.sort()
    return attributes

# - repeat this many times to get a forest  [Lecture 04, slide 41]
def build_forest(rows, labels, n_trees, m, max_depth, seed):
    ans = []
    rng = random.Random(seed)
    n_attributes = len(rows[0])
    base = len(set(labels))
    for i in range(n_trees):
        (sample_rows, sample_labels) = bootstrap_sample(rows, labels, rng)
        attributes = random_attributes(n_attributes, m, rng)
        ans.append(build_tree(sample_rows, sample_labels, attributes, base, max_depth))
    return ans

# Ensemble classifiers  [Lecture 04, slide 40]
# - combining a set of weak classifiers often improves classification
def predict_forest(forest, row):
    ans = {}
    for tree in forest:
        curr_ans = predict(tree, row)
        ans[curr_ans] = ans.get(curr_ans, 0) + 1
    return max(ans, key=ans.get)
