# =============================================================================
# CAP4611 - Programming Assignment 1: Exploratory Data Analysis
# Author: Richard Magiday
# Dataset: telco_churn2.csv (ConnectIQ Telecom Customer Churn)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # change to 'TkAgg' or remove if running interactively
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# =============================================================================
# A. BASIC SETUP
# =============================================================================

# --- A1: Load dataset ---
df = pd.read_csv('telco_churn2.csv')
print('=== A1: Dataset Shape ===')
print(f'Number of rows   : {df.shape[0]}')
print(f'Number of columns: {df.shape[1]}')

# --- A2: Summary statistics ---
print('\n=== A2: Summary Statistics (describe) ===')
print(df.describe())

# --- A3: Interpretation (written in markdown in the notebook) ---
# Key observations from describe():
# * tenure   : ranges 0–72 months; mean ~32, std ~24 — spread across the full
#              subscription lifecycle.
# * MonthlyCharges: ranges ~18–119; mean ~65, std ~30 — three broad service
#              tiers (phone-only, DSL, Fiber optic) drive the wide spread.
# * TotalCharges  : high std relative to mean; right-skewed (many short-tenure
#              customers with low totals, few long-tenure with high totals).
# * SeniorCitizen : mean ~0.162 → only ~16% of customers are seniors.

# --- A4: First 5 and last 5 rows ---
print('\n=== A4: First 5 rows ===')
print(df.head())
print('\n=== A4: Last 5 rows ===')
print(df.tail())

# --- A5: Numerical columns ---
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print('\n=== A5: Numerical Columns ===')
for c in num_cols:
    print(f'  - {c}')
print(f'  Total: {len(num_cols)}')
# Note: SeniorCitizen is stored as int (0/1) but is conceptually categorical.

# --- A6: Categorical columns ---
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print('\n=== A6: Categorical Columns ===')
for c in cat_cols:
    print(f'  - {c}  (unique values: {df[c].nunique()})')
print(f'  Total: {len(cat_cols)}')


# =============================================================================
# B. MISSING VALUES ANALYSIS
# =============================================================================

print('\n=== B1: Missing Value Count (descending) ===')
mv_count = df.isnull().sum().sort_values(ascending=False)
print(mv_count[mv_count > 0])

print('\n=== B2: Missing Value Percentage (descending) ===')
mv_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
print(mv_pct[mv_pct > 0].round(2))

# --- B3: Convert TotalCharges blank strings to NaN ---
print('\n=== B3: TotalCharges Conversion ===')
print(f'dtype before: {df["TotalCharges"].dtype}')
print('Blank-string rows:', (df['TotalCharges'].str.strip() == '').sum())
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(f'dtype after : {df["TotalCharges"].dtype}')
print('\nMissing values after conversion:')
mv_after = df.isnull().sum().sort_values(ascending=False)
print(mv_after[mv_after > 0])

# --- B4: Bar plot — columns with missing values (least → most) ---
mv_plot = df.isnull().sum()
mv_plot = mv_plot[mv_plot > 0].sort_values()   # ascending = least left

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(mv_plot.index, mv_plot.values, color='steelblue')
ax.set_title('Missing Values per Column (Least → Most)', fontsize=13)
ax.set_xlabel('Column')
ax.set_ylabel('Missing Count')
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 5,
            str(int(b.get_height())), ha='center', va='bottom', fontsize=10)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('B4_missing_bar.png', dpi=150)
plt.close()
print('Saved: B4_missing_bar.png')

# --- B5a: Missingno matrix (200-row sample) ---
sample_df = df.sample(200, random_state=42)
fig, ax = plt.subplots(figsize=(12, 6))
msno.matrix(sample_df, ax=ax, sparkline=True)
ax.set_title('Missingno Matrix — 200-row sample', fontsize=13)
plt.tight_layout()
plt.savefig('B5a_msno_matrix.png', dpi=150)
plt.close()
print('Saved: B5a_msno_matrix.png')

# Sparkline interpretation (written in markdown in the notebook):
# * Left-most value  : minimum number of non-null columns found in any single
#                      row of the sample (worst-case row completeness).
# * Right-most value : maximum number of non-null columns found in any single
#                      row (best-case row completeness).
# With only 2 columns missing (Education and TotalCharges), the sparkline
# will range from 22 (rows missing both) to 24 (fully complete rows).

# --- B5b: Missingno heatmap ---
fig, ax = plt.subplots(figsize=(8, 6))
msno.heatmap(df, ax=ax)
ax.set_title('Missingno Heatmap', fontsize=13)
plt.tight_layout()
plt.savefig('B5b_msno_heatmap.png', dpi=150)
plt.close()
print('Saved: B5b_msno_heatmap.png')

# Why the heatmap is not interesting here (written in markdown in the notebook):
# * Only 2 columns have missing values (Education and TotalCharges), so the
#   heatmap produces a near-empty 2×2 matrix — there is almost nothing to
#   compare across columns.
# * Value  1 : the two columns are ALWAYS missing together (perfect positive
#              co-missingness — when one is NaN, the other is too).
# * Value -1 : the two columns NEVER miss at the same time (perfect negative
#              co-missingness — mutually exclusive missingness).

# --- B6: How to handle missing values ---
# Education (80.6% missing, ~5 679 NaN):
#   → DROP the column. More than 80 % of rows lack a value; imputation would
#     manufacture the majority of the feature, introducing noise. The remaining
#     four education levels (HS / Assoc / Bach / Master) would cover < 20 % of
#     the data — far too sparse to be a reliable predictor.
#
# TotalCharges (0.16% missing, 11 NaN):
#   → These 11 rows all have tenure == 0 (brand-new customers never billed).
#     Two reasonable options:
#     (a) Impute with 0 — logically correct (no bill yet).
#     (b) Impute with tenure × MonthlyCharges — also valid and captures the
#         relationship between the three numerical features.
#     We will use option (a): df['TotalCharges'].fillna(0).

df['TotalCharges'] = df['TotalCharges'].fillna(0)
print('\nTotalCharges NaN filled with 0.')
print('Remaining missing values:')
print(df.isnull().sum()[df.isnull().sum() > 0])


# =============================================================================
# C. UNDERSTANDING CATEGORICAL ATTRIBUTES
# =============================================================================

# Columns included in categorical analysis.
# customerID and SupportTicketID are unique identifiers — not plotted
# individually but discussed below.
# SeniorCitizen is stored as int but is binary/categorical.
cat_analysis = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod',
    'Education', 'ActiveCountryCode', 'Churn',
]

# --- C: Bar plots — category counts ---
n_cols_plot = 2
n_rows_plot = (len(cat_analysis) + 1) // 2

fig, axes = plt.subplots(n_rows_plot, n_cols_plot,
                         figsize=(16, n_rows_plot * 4))
axes = axes.flatten()

for i, col in enumerate(cat_analysis):
    vc = df[col].value_counts(dropna=False)
    sns.barplot(x=vc.index.astype(str), y=vc.values,
                ax=axes[i], palette='Set2')
    axes[i].set_title(f'Counts: {col}', fontsize=11)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
    axes[i].tick_params(axis='x', rotation=45)
    for p in axes[i].patches:
        axes[i].annotate(f'{int(p.get_height())}',
                         (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha='center', va='bottom', fontsize=8)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Category Counts — All Categorical Attributes', fontsize=14)
plt.tight_layout()
plt.savefig('C_bar_counts.png', dpi=150)
plt.close()
print('Saved: C_bar_counts.png')

# --- C: Countplots — each feature vs Churn (target) ---
# Churn itself is excluded from the hue comparison (trivial self-comparison).
cat_vs_churn = [c for c in cat_analysis if c != 'Churn']

n_rows_plot2 = (len(cat_vs_churn) + 1) // 2
fig, axes = plt.subplots(n_rows_plot2, n_cols_plot,
                         figsize=(16, n_rows_plot2 * 4))
axes = axes.flatten()

for i, col in enumerate(cat_vs_churn):
    sns.countplot(data=df, x=col, hue='Churn',
                  ax=axes[i], palette='Set1')
    axes[i].set_title(f'{col} vs Churn', fontsize=11)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].legend(title='Churn', fontsize=8)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Categorical Features vs Churn (Target)', fontsize=14)
plt.tight_layout()
plt.savefig('C_countplot_vs_churn.png', dpi=150)
plt.close()
print('Saved: C_countplot_vs_churn.png')

# --- C: Interpretation (written as markdown in the notebook) ---
# Decisions & reasoning:
#
# * Churn (target): 26.5% Yes vs 73.5% No — significant class imbalance.
#   The dataset is imbalanced; resampling (SMOTE oversampling or random
#   undersampling) or class-weighted models will be needed before training.
#
# * customerID: unique identifier (7 043 unique values). Zero predictive value.
#   → DROP before modelling.
#
# * SupportTicketID: near-unique ticket reference (TK-XXXXXX format).
#   Zero predictive value. → DROP before modelling.
#
# * ActiveCountryCode: CONSTANT — all 7 043 rows = "+1". Zero variance.
#   A zero-variance feature adds no information. → DROP before modelling.
#
# * Education: 80.6% missing (already decided to drop in Section B).
#   → DROP.
#
# * SeniorCitizen: only ~16% are seniors (1), yet seniors show a noticeably
#   higher churn rate. → KEEP; consider encoding as binary 0/1 (already is).
#
# * Contract: Month-to-month customers make up the largest group AND have
#   the highest churn rate by far. This is likely one of the strongest
#   predictors. → KEEP.
#
# * InternetService: Fiber optic users churn the most. DSL churns less.
#   No-internet customers barely churn. → KEEP; high predictive signal.
#
# * PaymentMethod: Electronic check users churn far more than automatic
#   payment users (credit card / bank transfer). → KEEP.
#
# * OnlineSecurity, TechSupport, OnlineBackup, DeviceProtection,
#   StreamingTV, StreamingMovies: customers with "No internet service" form
#   a distinct third category. These columns have high correlation with
#   InternetService. Consider combining or dropping redundant ones to reduce
#   multicollinearity / dimensionality after checking feature importance.
#
# * gender: roughly 50/50 split; churn rates look similar for both genders.
#   May have low predictive power → monitor feature importance after modelling.
#
# * PaperlessBilling: customers on paperless billing churn more.
#   → KEEP (informative).
#
# Non-technical finding:
# Customers who pay by electronic check are much more likely to leave the
# company than those who use automatic payment methods like credit cards or
# bank transfers. This could suggest that customers who never set up autopay
# are less committed or are more actively looking for better deals elsewhere.


# =============================================================================
# D. UNDERSTANDING NUMERICAL ATTRIBUTES
# =============================================================================

num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

# --- D: Histograms ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(num_features):
    axes[i].hist(df[col].dropna(), bins=30, color='steelblue',
                 edgecolor='black', alpha=0.75)
    axes[i].axvline(df[col].mean(),   color='red',   linestyle='--',
                    label=f'Mean: {df[col].mean():.1f}')
    axes[i].axvline(df[col].median(), color='green', linestyle='--',
                    label=f'Median: {df[col].median():.1f}')
    axes[i].set_title(f'Histogram: {col}', fontsize=13)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    axes[i].legend(fontsize=9)

plt.suptitle('Histograms of Numerical Features', fontsize=14)
plt.tight_layout()
plt.savefig('D_histograms.png', dpi=150)
plt.close()
print('Saved: D_histograms.png')

# --- D: Seaborn distplot (histplot + KDE — modern equivalent of distplot) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(num_features):
    sns.histplot(df[col].dropna(), kde=True, bins=30,
                 ax=axes[i], color='purple', alpha=0.6)
    axes[i].set_title(f'Distplot: {col}', fontsize=13)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Density')

plt.suptitle('Seaborn Distplot (histplot + KDE) of Numerical Features', fontsize=14)
plt.tight_layout()
plt.savefig('D_distplots.png', dpi=150)
plt.close()
print('Saved: D_distplots.png')

# Skewness check
print('\n=== D: Skewness ===')
for col in num_features:
    skew = df[col].skew()
    print(f'  {col}: skewness = {skew:.3f}')

# --- D: Interpretation (markdown in notebook) ---
# Decisions & reasoning:
#
# * tenure: roughly uniform distribution across 1–72 months with slight
#   peaks at very short (new customers) and very long (loyal customers)
#   tenures. No strong skew. The bimodal tendency supports the categorical
#   grouping in Section G. → Consider the tenure_group feature as a simpler
#   representation.
#
# * MonthlyCharges: roughly uniform / slightly right-skewed. Multiple modes
#   visible (~20, ~50, ~70, ~100) corresponding to phone-only, DSL, and
#   Fiber optic bundles. No transformation strictly needed, but could be
#   normalised (MinMax or StandardScaler) before distance-based models.
#
# * TotalCharges: right-skewed (many low values for short-tenure customers,
#   long tail for high-tenure customers). Highly correlated with
#   tenure × MonthlyCharges. → Consider log-transformation to reduce skew,
#   or consider dropping it if tenure and MonthlyCharges are retained
#   (multicollinearity — see Section E).
#
# Non-technical findings:
# 1. Many customers have been with ConnectIQ for only a few months — the
#    spike of low-tenure customers suggests the company has been growing
#    quickly (or losing customers quickly).
# 2. Monthly bills tend to cluster around three price points (~$20, $50–70,
#    $90–100), reflecting the three main service tiers: phone-only, DSL
#    internet, and fast Fiber optic internet.
# 3. Most customers have paid a relatively small total to date (right-skewed
#    TotalCharges) — the bulk are newer; the very high totals belong to a
#    small group of long-loyal customers who are the least likely to churn.


# =============================================================================
# E. CORRELATION ANALYSIS
# =============================================================================

# --- E1: Correlation heatmap (include Churn as numeric) ---
df_corr = df.copy()
df_corr['Churn_num'] = (df_corr['Churn'] == 'Yes').astype(int)

corr_cols = ['tenure', 'MonthlyCharges', 'TotalCharges',
             'SeniorCitizen', 'Churn_num']
corr_labels = ['tenure', 'MonthlyCharges', 'TotalCharges',
               'SeniorCitizen', 'Churn']
corr_matrix = df_corr[corr_cols].corr()
corr_matrix.columns = corr_labels
corr_matrix.index   = corr_labels

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5,
            annot_kws={'size': 12}, ax=ax)
ax.set_title('Correlation Heatmap — Numerical Features + Churn', fontsize=13)
plt.tight_layout()
plt.savefig('E1_heatmap.png', dpi=150)
plt.close()
print('Saved: E1_heatmap.png')

print('\n=== E1: Correlation Matrix ===')
print(corr_matrix.round(3))

# --- E2: Boxplots — numerical features vs Churn ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, col in enumerate(num_features):
    sns.boxplot(data=df, x='Churn', y=col, ax=axes[i], palette='Set2')
    axes[i].set_title(f'{col} vs Churn', fontsize=13)
    axes[i].set_xlabel('Churn')
    axes[i].set_ylabel(col)

plt.suptitle('Numerical Features vs Churn', fontsize=14)
plt.tight_layout()
plt.savefig('E2_boxplot_vs_churn.png', dpi=150)
plt.close()
print('Saved: E2_boxplot_vs_churn.png')

# --- E: Interpretation (markdown in notebook) ---
# Feature relationships & decisions:
#
# * TotalCharges vs tenure  (r ≈ 0.83): very high positive correlation —
#   multicollinearity. TotalCharges is essentially tenure × MonthlyCharges.
#   → Consider DROPPING TotalCharges and retaining tenure + MonthlyCharges
#     to avoid redundancy and unstable coefficients in linear models.
#
# * TotalCharges vs MonthlyCharges (r ≈ 0.65): moderate positive correlation,
#   confirming the above — higher monthly bills accumulate to higher totals.
#
# * Churn vs tenure (r ≈ −0.35): moderate negative correlation — longer-
#   tenured customers are less likely to churn. Tenure is a strong predictor.
#
# * Churn vs MonthlyCharges (r ≈ +0.19): weak positive correlation — higher
#   bills slightly increase churn probability.
#
# * Churn vs SeniorCitizen (r ≈ +0.15): weak positive — seniors churn a bit
#   more. Not enough to drop, but not a dominant predictor on its own.
#
# Non-technical findings:
# 1. Customers who eventually left (Churn = Yes) had noticeably SHORTER tenure
#    on average — they tended to leave within the first 1–2 years, while loyal
#    customers stayed 4–6 years.
# 2. Churned customers paid HIGHER monthly bills on average — suggesting that
#    customers who feel the price is too high for the value they receive are
#    more likely to cancel their service.


# =============================================================================
# F. OUTLIERS
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, col in enumerate(num_features):
    q1  = df[col].quantile(0.25)
    q3  = df[col].quantile(0.75)
    iqr = q3 - q1
    lower  = q1 - 1.5 * iqr
    upper  = q3 + 1.5 * iqr
    n_out  = ((df[col] < lower) | (df[col] > upper)).sum()

    sns.boxplot(data=df, y=col, ax=axes[i], color='lightblue',
                flierprops=dict(marker='o', color='red', alpha=0.4, markersize=4))
    axes[i].set_title(f'Boxplot: {col}', fontsize=12)
    axes[i].set_xlabel(f'IQR outliers: {n_out} ({n_out/len(df)*100:.1f}%)')
    print(f'{col}: lower fence={lower:.2f}, upper fence={upper:.2f}, '
          f'outliers={n_out} ({n_out/len(df)*100:.1f}%)')

plt.suptitle('Outlier Detection via Boxplots', fontsize=14)
plt.tight_layout()
plt.savefig('F_outliers.png', dpi=150)
plt.close()
print('Saved: F_outliers.png')

# --- F: Discussion (markdown in notebook) ---
# * tenure: bounded 0–72 by design; no IQR outliers. Distribution is spread
#   but not extreme — no action needed.
#
# * MonthlyCharges: a small number of very high charges fall above the upper
#   fence (~$115+). These likely correspond to customers with every add-on
#   service active. They are plausible business values, NOT data errors.
#   → Retain; they represent a real high-value customer segment.
#
# * TotalCharges: right tail extends to ~$8 000+ (long-tenure, high-bill
#   customers). These are legitimate and expected. No data errors.
#   → Retain, but apply log-transformation before feeding into distance-based
#     or linear models to reduce the influence of the long tail.


# =============================================================================
# G. TRANSFORMING TENURE
# =============================================================================

# --- G1: Identify unique values ---
print('\n=== G1: Tenure — Unique Values ===')
tenure_vals = sorted(df['tenure'].unique())
print(f'Range  : {min(tenure_vals)} – {max(tenure_vals)} months')
print(f'Unique count: {len(tenure_vals)}')
print(f'Values : {tenure_vals}')

# --- G2: Categorize into 3 groups ---
def categorize_tenure(t):
    if t <= 24:
        return 0   # Short-term  (0–2 years)
    elif t <= 48:
        return 1   # Mid-term    (2–4 years)
    else:
        return 2   # Long-term   (4–6 years)

df['tenure_group'] = df['tenure'].apply(categorize_tenure)

print('\n=== G2: Tenure Group Distribution ===')
print(df['tenure_group'].value_counts().sort_index())
print('\nGroup legend:')
print('  0 = Short-term  (tenure  0–24 months,  0–2 years)')
print('  1 = Mid-term    (tenure 25–48 months,  2–4 years)')
print('  2 = Long-term   (tenure 49–72 months,  4–6 years)')

# Verification
for g in [0, 1, 2]:
    sub = df[df['tenure_group'] == g]['tenure']
    print(f'  Group {g}: min={sub.min()}, max={sub.max()}, n={len(sub)}')

# Plot
group_counts = df['tenure_group'].value_counts().sort_index()
labels = ['Short-term\n(0–24 mo)', 'Mid-term\n(25–48 mo)', 'Long-term\n(49–72 mo)']
colors = ['#ff9999', '#99c2ff', '#99ff99']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, group_counts.values, color=colors, edgecolor='grey')
ax.set_title('Tenure Group Distribution', fontsize=13)
ax.set_ylabel('Count')
for bar, count in zip(bars, group_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30, str(count), ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('G_tenure_groups.png', dpi=150)
plt.close()
print('Saved: G_tenure_groups.png')

# --- G3: Explanation (markdown in notebook) ---
# Categorization logic:
# The tenure feature spans 0–72 months (6 years).  Dividing this into three
# equal 24-month windows creates natural, business-meaningful segments:
#   * Group 0 (0–24 months): "Short-term" — new/recent customers who have not
#     yet built loyalty.  Churn risk is highest in this window.
#   * Group 1 (25–48 months): "Mid-term" — established customers who have
#     survived the first wave of churn.  Moderate risk.
#   * Group 2 (49–72 months): "Long-term" — loyal customers unlikely to churn.
# This 3-level ordinal feature is simpler for some models (e.g., decision
# trees) to exploit than the raw continuous value.


# =============================================================================
# H. SUMMARY AND DISCUSSION
# =============================================================================

print('\n=== H: Summary ===')
summary = """
FINDINGS & NEXT STEPS
=====================
1. Class imbalance (Churn):
   - 73.5% No / 26.5% Yes (~3:1 ratio).
   - Must rebalance before training:
       * Oversampling : SMOTE (Synthetic Minority Oversampling Technique)
       * Undersampling: Random Under-Sampler
       * Class weights: class_weight='balanced' in sklearn estimators
   - Evaluation metric: prefer F1-score, ROC-AUC, or Precision-Recall AUC
     over raw Accuracy (which is misleading with imbalanced classes).

2. Columns to DROP:
   - customerID      : unique identifier, no predictive value.
   - SupportTicketID : unique ticket reference, no predictive value.
   - ActiveCountryCode: zero variance (constant "+1").
   - Education       : 80.6% missing — too sparse to impute reliably.
   - TotalCharges    : highly collinear with tenure × MonthlyCharges (r ≈ 0.83);
                       retaining tenure + MonthlyCharges avoids multicollinearity.

3. Missing values handled:
   - TotalCharges (11 rows, tenure=0): filled with 0 (not yet billed).
   - Education    (5 679 rows)       : column dropped.

4. Distribution / transformation needs:
   - TotalCharges is right-skewed → log-transform before distance-based or
     linear models (already recommended to drop, but if kept: log1p).
   - MonthlyCharges and tenure are reasonably distributed; apply StandardScaler
     or MinMaxScaler before models sensitive to feature scale (SVM, KNN, etc.).

5. Key predictors identified:
   - Contract type (Month-to-month → very high churn).
   - Tenure (short tenure → high churn risk).
   - InternetService (Fiber optic → higher churn than DSL or None).
   - PaymentMethod (Electronic check → higher churn).
   - MonthlyCharges (higher bill → slightly higher churn).
   - SeniorCitizen (seniors churn more).
   - PaperlessBilling (paperless billing users churn more).

6. Feature engineering suggestions:
   - tenure_group (created in G): ordinal categorical grouping.
   - Consider one-hot encoding for nominal features
     (InternetService, PaymentMethod, Contract).
   - Binary encode Yes/No features as 0/1.

7. Redundant / correlated feature groups to watch:
   - OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
     StreamingTV, StreamingMovies all share the "No internet service"
     level (tied to InternetService). After one-hot encoding, check VIF
     (Variance Inflation Factor) and prune if needed.
"""
print(summary)
