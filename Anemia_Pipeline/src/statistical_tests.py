"""
McNemar's test for comparing two classifiers' predictions on the SAME test set.

McNemar's test is the right tool here (rather than an unpaired t-test) because
we're comparing two models' predictions on identical samples — it only looks
at the samples where the two models disagree, which is exactly what matters
for "is model A significantly better than model B on this data".

Usage as a library (e.g., to compare Stage-1 hybrid vs. the ablation's
Config B/C, or Stage-2 vs. a future revision):

    from src.statistical_tests import mcnemar_test
    result = mcnemar_test(y_true, y_pred_model_a, y_pred_model_b)

Usage from the command line, given two saved prediction arrays (.npy files
of 0/1 predictions, same order/length as each other and as y_true):

    python -m src.statistical_tests \
        --y-true y_true.npy --pred-a model_a_preds.npy --pred-b model_b_preds.npy
"""

import argparse

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar


def mcnemar_test(y_true, y_pred_a, y_pred_b, alpha=0.05):
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    a_correct = (y_pred_a == y_true)
    b_correct = (y_pred_b == y_true)

    # Contingency table:
    #                B correct   B incorrect
    # A correct         n00          n01
    # A incorrect       n10          n11
    n00 = int(np.sum(a_correct & b_correct))
    n01 = int(np.sum(a_correct & ~b_correct))
    n10 = int(np.sum(~a_correct & b_correct))
    n11 = int(np.sum(~a_correct & ~b_correct))
    table = [[n00, n01], [n10, n11]]

    # exact=True uses the binomial test, recommended when n01 + n10 < 25;
    # otherwise the chi-square approximation with continuity correction is used.
    use_exact = (n01 + n10) < 25
    result = mcnemar(table, exact=use_exact, correction=not use_exact)

    significant = result.pvalue < alpha
    print(f"Contingency table: {table}")
    print(f"McNemar statistic: {result.statistic:.4f}")
    print(f"p-value: {result.pvalue:.6f}")
    print(f"Significant at alpha={alpha}: {significant}")

    return {
        "contingency_table": table,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant": bool(significant),
        "alpha": alpha,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--y-true", required=True, help=".npy file of ground-truth labels")
    parser.add_argument("--pred-a", required=True, help=".npy file of model A's predictions")
    parser.add_argument("--pred-b", required=True, help=".npy file of model B's predictions")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    y_true = np.load(args.y_true)
    pred_a = np.load(args.pred_a)
    pred_b = np.load(args.pred_b)
    mcnemar_test(y_true, pred_a, pred_b, args.alpha)
