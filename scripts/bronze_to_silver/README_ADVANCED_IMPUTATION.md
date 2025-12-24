# Advanced Imputation: K-NN and MICE

## Overview

This notebook (`3b_b2s_advanced_imputation_knn_mice.ipynb`) implements advanced imputation techniques (K-NN and MICE) for specific columns that benefit from relationship-aware imputation.

## When to Use

Use this notebook when you want to:
- Improve imputation accuracy for columns with complex inter-feature relationships
- Compare advanced methods with the baseline group median approach
- Leverage correlations between features for better imputation

## Target Columns

### MICE Imputation
- **BP_systolic, BP_diastolic**: Strong correlation between the pair makes MICE ideal

### K-NN Imputation
- **Vital Signs Cluster**: temperature, heart_rate, resp_rate, o2sat (highly correlated)
- **Lab Values**: creatinine, blood_wbc (multi-factor relationships)

## Usage

### Option 1: Standalone Execution
1. Run the main missing value handling notebook first (`3_b2s_handle_missing_values.ipynb`) up to temporal filling
2. Run this notebook to apply K-NN/MICE
3. Compare results with baseline approach

### Option 2: Integrated Pipeline
1. Modify `3_b2s_handle_missing_values.ipynb` to call functions from this notebook
2. Replace group median steps with K-NN/MICE for target columns

## Performance Considerations

- **Runtime**: 10-30 minutes for 158K rows
- **Memory**: Requires 8GB+ RAM
- **Scalability**: K-NN scales as O(n²), consider sampling for very large datasets

## Configuration

### K-NN Parameters
- `n_neighbors`: Default 5, adjust based on data density
- `weights`: 'distance' (default) or 'uniform'
- `metric`: 'euclidean' (default)

### MICE Parameters
- `max_iter`: Default 10 iterations
- `estimator`: RandomForestRegressor (handles non-linear relationships)
- `n_estimators`: 50 trees per iteration

## Output Files

- `bronze_missing_values_handled_advanced.csv`: Dataset with K-NN/MICE imputation
- `imputation_comparison_summary.csv`: Summary of missing values after imputation

## Comparison with Baseline

The baseline approach uses:
- Temporal filling (forward/backward fill within patients)
- Group medians by (y, age_group, gender/icu_admission)

K-NN/MICE advantages:
- Captures complex feature relationships
- Uses multiple features simultaneously
- Better for correlated variables

Baseline advantages:
- Much faster (seconds vs minutes)
- More interpretable
- Handles temporal structure well

## Recommendations

1. **Start with baseline**: The current approach is efficient and appropriate for most cases
2. **Use K-NN/MICE selectively**: Only for columns where relationships matter
3. **Compare results**: Evaluate if improved imputation translates to better model performance
4. **Monitor performance**: Track computational cost vs. accuracy gains

## Troubleshooting

### Memory Issues
- Reduce `n_neighbors` for K-NN
- Process columns in batches
- Use sampling for initial testing

### Convergence Issues (MICE)
- Increase `max_iter`
- Check for highly correlated features
- Ensure sufficient non-missing data

### Slow Performance
- Reduce `n_estimators` in MICE
- Use fewer features in K-NN
- Consider parallel processing (already enabled for RandomForest)

## Next Steps

After running this notebook:
1. Compare imputed values with baseline
2. Evaluate impact on downstream models
3. Decide whether to integrate into main pipeline
4. Document any improvements in model performance

