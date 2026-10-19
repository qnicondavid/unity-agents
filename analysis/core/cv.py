# The comments in the code are based on my (ND Milandru) iai lecture notes.

import random

# - split the data into k parts, called folds  [Lecture 04, slide 35]
def make_folds(n, k, seed):
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    base = n // k
    remainder = n % k

    ans = []
    i = 0

    for _ in range(k):
        curr = indices[i: i + base]
        i += base
        if remainder:
            curr.append(indices[i])
            i += 1
            remainder -= 1
        ans.append(curr)
    return ans


def make_group_folds(groups, k, seed):
    positions = {}
    for i in range(len(groups)):
        if groups[i] not in positions:
            positions[groups[i]] = []
        positions[groups[i]].append(i)

    distinct = list(positions)
    rng = random.Random(seed)
    rng.shuffle(distinct)

    ans = []
    for _ in range(k):
        ans.append([])
    for group in distinct:
        smallest = 0
        for f in range(k):
            if len(ans[f]) < len(ans[smallest]):
                smallest = f
        ans[smallest].extend(positions[group])

    for fold in ans:
        fold.sort()
    return ans

# Imbalance & Accuracy  [Lecture 04, slide 36]
# - when one class is much more common, always predicting that class gives high
#   accuracy without learning anything
# - balanced accuracy is the average of the accuracy on each class
def accuracy(true_labels, predicted_labels):
    total = len(true_labels)
    positives = 0
    for i in range(total):
        if true_labels[i] == predicted_labels[i]:
            positives += 1
    return positives / total


def balanced_accuracy(true_labels, predicted_labels):
    totals = {}
    trues = {}

    total = len(true_labels)

    for i in range(total):
        totals[true_labels[i]] = totals.get(true_labels[i], 0) + 1
        if true_labels[i] == predicted_labels[i]:
            trues[true_labels[i]] = trues.get(true_labels[i], 0) + 1

    total_accuracy = 0

    for label in totals:
        total_accuracy += trues.get(label, 0) / totals[label]

    return total_accuracy / len(totals)


def confusion_matrix(true_labels, predicted_labels):
    labels = sorted(set(true_labels) | set(predicted_labels))

    matrix = {
        true_label: {predicted_label: 0 for predicted_label in labels}
        for true_label in labels
    }

    for i in range(len(true_labels)):
        matrix[true_labels[i]][predicted_labels[i]] += 1
    return matrix


def mean_absolute_error(true_values, predicted_values):
    total = 0
    for i in range(len(true_values)):
        total += abs(true_values[i] - predicted_values[i])
    return total / len(true_values)


# Why not measure on the training set?  [Lecture 04, slide 34]
# - performance on the training set only shows what the algorithm remembers
# - the best estimate of future performance comes from an independent test set
#
# Cross-validation (CV)  [Lecture 04, slide 35]
# - estimates test performance using only the training data
# - split the data into k parts, called folds
# - train on k-1 folds and test on the remaining one
# - repeat k times so each fold is the test fold once
# - average the k scores
def cross_validation(rows, targets, folds, train, predict):
    predictions = [None] * len(rows)
    k = len(folds)
    for f in range(k):
        training_rows, training_targets = [], []
        test_rows = []
        for pos in folds[f]:
            test_rows.append(rows[pos])
        for j in range(k):
            if j != f:
                for pos in folds[j]:
                    training_rows.append(rows[pos])
                    training_targets.append(targets[pos])
        model = train(training_rows, training_targets)
        predicted = predict(model, test_rows)
        for i in range(len(test_rows)):
            predictions[folds[f][i]] = predicted[i]

    return predictions
