# Hierarchical Classifier Methodology Fix

## Problem Identified

The hierarchical classifier was evaluating on **training data** instead of using proper **cross-validation**, leading to artificially inflated accuracy scores (100%).

### Original Buggy Code (taxonomy_analysis.py)
```python
if use_hierarchical:
    # Train hierarchical classifier
    hier_clf.fit(X_scaled, y, feature_names=feature_names)

    # Evaluate on SAME training data ❌
    eval_results = hier_clf.evaluate(X_scaled, y)  # BUG!
    multiclass_accuracy = eval_results['overall_accuracy']  # 100% false positive
```

This is methodologically unsound because:
1. **Data leakage**: Testing on training data
2. **Overfitting**: Models memorize training data perfectly
3. **No generalization test**: Doesn't measure real-world performance
4. **Invalid comparison**: Can't compare to flat classifier using CV

## Solution Implemented

Replaced training-set evaluation with **proper k-fold cross-validation**, matching the methodology used for the flat classifier.

### Fixed Code (taxonomy_analysis.py)
```python
if use_hierarchical:
    # Use StratifiedKFold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, test_idx in skf.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train Stage 1: Binary (aligned vs misaligned)
        stage1_clf = RandomForestClassifier(**hier_clf.stage1_params)
        stage1_clf.fit(X_train, y_train_binary)
        stage1_pred_test = stage1_clf.predict(X_test)  # ✅ Test on held-out data

        # Train Stage 2: Misalignment subtype (6-way)
        stage2_clf = RandomForestClassifier(**hier_clf.stage2_params)
        stage2_clf.fit(X_train_mis, y_train_mis)
        stage2_pred = stage2_clf.predict(X_test_mis)  # ✅ Test on held-out data

        # Combine predictions
        y_pred[test_idx] = fold_pred

    # Calculate accuracy on cross-validated predictions
    multiclass_accuracy = (y_pred == y).mean()  # ✅ Valid
```

## Key Methodological Improvements

### 1. Cross-Validation
- **Before:** Train on all data, test on all data (invalid)
- **After:** Train on 80%, test on 20%, repeat 5 times (valid)

### 2. Fair Comparison
- **Flat classifier:** Uses `cross_val_predict()` → CV predictions
- **Hierarchical (fixed):** Uses manual k-fold CV → CV predictions
- **Now comparable!** Both use same methodology

### 3. Stage-Specific Metrics
Proper calculation of each stage's accuracy:
```python
# Stage 1: Aligned vs Misaligned (binary)
stage1_accuracy = (stage1_preds == y_binary).mean()

# Stage 2: Misalignment subtype (6-way, only on misaligned)
stage2_accuracy = (y_pred[misaligned_mask] == y[misaligned_mask]).mean()
```

### 4. No Data Leakage
Each fold:
- Trains on fold training data
- Tests on fold test data (never seen during training)
- Predictions aggregated across all folds

## Expected Impact

### Before Fix:
- **Reported accuracy:** 100% (on training data)
- **Real accuracy:** Unknown (likely much lower)
- **Comparison to flat:** Invalid (different methodologies)

### After Fix:
- **Reported accuracy:** Actual cross-validated performance
- **Real accuracy:** Same as reported (valid generalization)
- **Comparison to flat:** Valid (same methodology)

## Test Results

### Hierarchical CV Fix Test (N=5)
Running: `test_hierarchical_fix.py`
- **Purpose:** Verify fix doesn't give 100% accuracy on small sample
- **Status:** In progress...

### Full Experiment (N=100, Flat Classifier)
Running: `run0008` (background)
- **Purpose:** Baseline comparison with enhanced probes
- **Status:** In progress...

## Files Modified

1. **`taxonomy_analysis.py`** (lines 140-212)
   - Replaced `hier_clf.fit()` + `hier_clf.evaluate()` with manual k-fold CV
   - Now generates proper cross-validated predictions
   - Calculates stage1_accuracy and stage2_accuracy correctly

2. **`exp10_probe_enhanced_v2.py`** (line 617)
   - Fixed embeddings save method
   - Changed from `save_embeddings()` to `np.savez()`

## Research Integrity

This fix is critical for:
- **Valid conclusions:** Can't claim 92%+ accuracy with flawed methodology
- **Reproducibility:** Other researchers can validate our approach
- **Fair comparison:** Can properly compare flat vs hierarchical
- **Scientific rigor:** Meets standards for ML evaluation

## Next Steps

1. ✅ Implement fix
2. ⏳ Verify with N=5 test (running)
3. ⏳ Run full N=100 with flat classifier (running)
4. ⬜ Run full N=100 with hierarchical classifier (after verification)
5. ⬜ Compare results and determine which approach is better

## Summary

**Problem:** Hierarchical classifier used training set evaluation → 100% false accuracy
**Solution:** Implement proper k-fold cross-validation → true accuracy
**Impact:** Now have scientifically sound methodology for comparison
**Status:** Fix implemented, tests running to verify

---

**Date:** 2025-11-26
**Author:** Claude
**Reviewed:** User confirmed need for methodological soundness
