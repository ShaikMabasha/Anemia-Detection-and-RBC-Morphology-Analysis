"""
Disease-mapping module v2 — uses REAL predicted morphology classes from the
Chula-RBC-12-trained multi-class model (stage3_morphology_multiclass.py),
replacing the old score-threshold heuristic in inference_pipeline.py's
MORPHOLOGY_RULES, which was an ad-hoc approximation because Stage-3 only had
a binary abnormal/normal score to work with.

Each predicted class now maps directly and transparently to known
associated conditions, based on standard hematology references. This
mapping is still NOT independently validated by a clinician — see the
paper's Limitations section — but it is at least now grounded in a real,
specific, independently-annotated cell classification rather than an
arbitrary score range.
"""

# Class name -> set of possible disease/condition associations.
# Sources: standard hematology references (e.g., Wintrobe's Clinical
# Hematology); the same clinical associations already used in the paper's
# Section 3.4.3, extended to match the full Chula-RBC-12 class set.
DISEASE_MAP = {
    "Normal_cell": set(),
    "Macrocyte": {"Vitamin B12 Deficiency", "Folate Deficiency", "Liver Disease"},
    "Microcyte": {"Iron Deficiency Anemia", "Thalassemia"},
    "Spherocyte": {"Hereditary Spherocytosis", "Autoimmune Hemolytic Anemia"},
    "Target_cell": {"Thalassemia", "Liver Disease", "Hemoglobin C Disease"},
    "Stomatocyte": {"Liver Disease", "Alcoholism", "Hereditary Stomatocytosis"},
    "Ovalocyte": {"Hereditary Elliptocytosis", "Iron Deficiency Anemia"},
    "Teardrop": {"Myelofibrosis", "Thalassemia", "Bone Marrow Infiltration"},
    "Burr_cell": {"Uremia", "Pyruvate Kinase Deficiency", "Artifact (consider re-staining)"},
    "Schistocyte": {"Microangiopathic Hemolytic Anemia", "DIC", "Mechanical Heart Valve"},
    "Hypochromia": {"Iron Deficiency Anemia", "Thalassemia"},
    "Elliptocyte": {"Hereditary Elliptocytosis", "Iron Deficiency Anemia"},
}


def map_diseases(predicted_classes):
    """predicted_classes: iterable of class-name strings (e.g., from majority-vote
    across all cells detected in an RBC image, or the single top prediction)."""
    diseases = set()
    for cls in predicted_classes:
        diseases |= DISEASE_MAP.get(cls, set())
    return sorted(diseases)


def summarize_cell_predictions(class_predictions, min_fraction=0.05):
    """
    Given a list of per-cell predicted class names for one RBC image, returns
    a summary: {class_name: fraction_of_cells}, filtered to classes present
    above min_fraction, sorted by prevalence descending. This is what should
    be reported in the diagnostic report instead of a single score.
    """
    from collections import Counter

    total = len(class_predictions)
    if total == 0:
        return {}
    counts = Counter(class_predictions)
    summary = {
        cls: count / total
        for cls, count in counts.items()
        if count / total >= min_fraction
    }
    return dict(sorted(summary.items(), key=lambda kv: kv[1], reverse=True))


if __name__ == "__main__":
    # Quick self-test with a synthetic example
    example_predictions = (
        ["Normal_cell"] * 40 + ["Microcyte"] * 12 + ["Target_cell"] * 8 + ["Hypochromia"] * 5
    )
    summary = summarize_cell_predictions(example_predictions)
    print("Morphology summary:", summary)
    print("Possible disease indications:", map_diseases(summary.keys()))
