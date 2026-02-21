# compensation_buddy
An MLP model to predict SWE compensation based on real-world dataset

## Data Pre-Processing

### Data Cleaning
- WAGE_UNIT_OF_PAY: convert into annual wages
- OUTLIER_FILTER: keep only range 50000-400000
- Normalize using log1p for bell-shaped, symmetric data

![Comparison Plot](media/log1p_comparison.png)

### Data Preprocessing
- Categorize JOB_TITLE into role_category
- Categorize EMPLOYER_NAME into is_top_tier, industry
- Add new field seniority based on JOB_TITLE
- Categorize WORKERSITE_CITY into metro_tier
- One-Hot Encode the categorical columns
