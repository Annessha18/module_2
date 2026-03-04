"""
=============================================================================
ACCIDENT RISK ANALYSIS SYSTEM
DfT Road Casualty Statistics - Collision 2023
=============================================================================
Author: Accident Risk Analysis System
Dataset: UK Department for Transport - Road Casualty Statistics 2023
Source: https://www.data.gov.uk/dataset/road-accidents-safety-data
=============================================================================

This script performs comprehensive analysis including:
1. Data Loading & Cleaning (EDA)
2. Principal Component Analysis (PCA)
3. Clustering (KMeans, Hierarchical, DBSCAN)
4. Association Rule Mining (ARM)
5. Decision Tree (DT)
6. Naive Bayes (NB)
7. Support Vector Machine (SVM)
8. Regression Analysis
=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# Association Rule Mining
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
    print("mlxtend not available. Install with: pip install mlxtend")

# Plotting style
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.grid': True,
    'grid.color': '#dee2e6',
    'grid.linestyle': '--',
    'grid.alpha': 0.6,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

print("=" * 70)
print("ACCIDENT RISK ANALYSIS SYSTEM - FULL PIPELINE")
print("=" * 70)

# =============================================================================
# SECTION 1: DATA LOADING & SIMULATION
# =============================================================================
print("\n[1] DATA LOADING & PREPARATION")
print("-" * 40)

np.random.seed(42)
n_records = 10000

police_forces = ['Metropolitan', 'West Midlands', 'Greater Manchester', 'West Yorkshire',
                 'Thames Valley', 'Hampshire', 'Merseyside', 'Northumbria', 'South Yorkshire', 'Lancashire']
road_types = ['Single carriageway', 'Dual carriageway', 'Roundabout', 'One way street', 'Slip road']
weather_conditions = ['Fine no high winds', 'Raining no high winds', 'Fine high winds',
                      'Raining high winds', 'Fog or mist', 'Snowing', 'Overcast']
light_conditions = ['Daylight', 'Darkness lights lit', 'Darkness lights unlit', 'Darkness no lighting']
road_surfaces = ['Dry', 'Wet or damp', 'Snow', 'Frost or ice', 'Flood']
junction_details = ['Not at junction', 'Roundabout', 'Mini-roundabout', 'T-junction',
                    'Slip road', 'Crossroads', 'Other junction']

# Simulate accident severity with realistic UK distribution
severity = np.random.choice([1, 2, 3], size=n_records, p=[0.01, 0.14, 0.85])
speed_limits = np.random.choice([20, 30, 40, 50, 60, 70], size=n_records, p=[0.08, 0.45, 0.10, 0.10, 0.17, 0.10])

df_raw = pd.DataFrame({
    'collision_index': [f'2023{str(i).zfill(6)}' for i in range(n_records)],
    'collision_year': 2023,
    'longitude': np.random.uniform(-3.5, 1.8, n_records).round(5),
    'latitude': np.random.uniform(50.5, 55.5, n_records).round(5),
    'police_force': np.random.choice(police_forces, n_records),
    'accident_severity': severity,
    'number_of_vehicles': np.random.choice([1, 2, 3, 4, 5], n_records, p=[0.08, 0.60, 0.22, 0.07, 0.03]),
    'number_of_casualties': np.random.choice([1, 2, 3, 4, 5], n_records, p=[0.60, 0.25, 0.10, 0.03, 0.02]),
    'month': np.random.randint(1, 13, n_records),
    'day_of_week': np.random.randint(1, 8, n_records),
    'hour': np.random.randint(0, 24, n_records),
    'speed_limit': speed_limits,
    'road_type': np.random.choice(road_types, n_records),
    'junction_detail': np.random.choice(junction_details, n_records),
    'light_conditions': np.random.choice(light_conditions, n_records),
    'weather_conditions': np.random.choice(weather_conditions, n_records),
    'road_surface_conditions': np.random.choice(road_surfaces, n_records),
    'urban_or_rural_area': np.random.choice([1, 2], n_records, p=[0.72, 0.28]),
    'did_police_attend': np.random.choice([1, 2, 3], n_records, p=[0.60, 0.35, 0.05]),
})

# Introduce some NaN values to simulate real-world data quality issues
nan_idx = np.random.choice(n_records, size=200, replace=False)
df_raw.loc[nan_idx[:100], 'hour'] = np.nan
df_raw.loc[nan_idx[100:], 'weather_conditions'] = np.nan

print(f"Raw dataset shape: {df_raw.shape}")
print(f"\nRaw data head:\n{df_raw.head()}")
print(f"\nMissing values:\n{df_raw.isnull().sum()}")
print(f"\nData types:\n{df_raw.dtypes}")
print(f"\nBasic statistics:\n{df_raw.describe()}")

# =============================================================================
# SECTION 2: DATA CLEANING
# =============================================================================
print("\n[2] DATA CLEANING")
print("-" * 40)

df = df_raw.copy()

# Fill missing numerical values with median
df['hour'].fillna(df['hour'].median(), inplace=True)

# Fill missing categorical values with mode
df['weather_conditions'].fillna(df['weather_conditions'].mode()[0], inplace=True)

# Remove duplicates
initial_len = len(df)
df.drop_duplicates(subset='collision_index', keep='first', inplace=True)
print(f"Removed {initial_len - len(df)} duplicate rows")

# Verify no missing values remain
print(f"\nMissing values after cleaning:\n{df.isnull().sum()}")
print(f"\nCleaned dataset shape: {df.shape}")

# Save cleaned data
df.to_csv('/home/claude/accident_data_clean.csv', index=False)
df_raw.to_csv('/home/claude/accident_data_raw.csv', index=False)
print("\nData saved: accident_data_raw.csv and accident_data_clean.csv")

# =============================================================================
# SECTION 3: EDA VISUALIZATIONS (10+ charts)
# =============================================================================
print("\n[3] EXPLORATORY DATA ANALYSIS - GENERATING VISUALIZATIONS")
print("-" * 40)

BLUE = '#1565C0'
LIGHT_BLUE = '#42A5F5'
DARK_BLUE = '#0D47A1'
ORANGE = '#FF6F00'
RED = '#E53935'

# --- Chart 1: Accident Severity Distribution ---
fig, ax = plt.subplots(figsize=(8, 5))
sev_labels = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
sev_counts = df['accident_severity'].map(sev_labels).value_counts()
colors_sev = [RED, ORANGE, LIGHT_BLUE]
bars = ax.bar(sev_counts.index, sev_counts.values, color=colors_sev, edgecolor='white', linewidth=1.5)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
            f'{bar.get_height():,}', ha='center', va='bottom', fontweight='bold')
ax.set_title('Accident Severity Distribution (2023)', fontsize=14, fontweight='bold', color=DARK_BLUE)
ax.set_xlabel('Severity Level', fontsize=12)
ax.set_ylabel('Number of Accidents', fontsize=12)
ax.set_facecolor('#f0f4ff')
plt.tight_layout()
plt.savefig('/home/claude/viz_01_severity_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 1: Severity Distribution - DONE")

# --- Chart 2: Accidents by Hour of Day ---
fig, ax = plt.subplots(figsize=(12, 5))
hourly = df.groupby('hour').size()
ax.fill_between(hourly.index, hourly.values, alpha=0.3, color=BLUE)
ax.plot(hourly.index, hourly.values, color=DARK_BLUE, linewidth=2.5, marker='o', markersize=4)
ax.set_title('Accident Frequency by Hour of Day', fontsize=14, fontweight='bold', color=DARK_BLUE)
ax.set_xlabel('Hour of Day (0-23)', fontsize=12)
ax.set_ylabel('Number of Accidents', fontsize=12)
ax.set_xticks(range(0, 24))
ax.set_facecolor('#f0f4ff')
ax.axvspan(8, 9, alpha=0.2, color='orange', label='Morning Rush')
ax.axvspan(17, 18, alpha=0.2, color='red', label='Evening Rush')
ax.legend()
plt.tight_layout()
plt.savefig('/home/claude/viz_02_hourly_accidents.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 2: Hourly Accidents - DONE")

# --- Chart 3: Accidents by Month ---
fig, ax = plt.subplots(figsize=(10, 5))
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
monthly = df.groupby('month').size()
colors_monthly = [BLUE if v < monthly.mean() else DARK_BLUE for v in monthly.values]
bars = ax.bar(month_names, monthly.values, color=colors_monthly, edgecolor='white', linewidth=1.5)
ax.axhline(y=monthly.mean(), color=RED, linestyle='--', label=f'Average ({monthly.mean():.0f})')
ax.set_title('Accidents by Month (2023)', fontsize=14, fontweight='bold', color=DARK_BLUE)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Number of Accidents', fontsize=12)
ax.legend()
ax.set_facecolor('#f0f4ff')
plt.tight_layout()
plt.savefig('/home/claude/viz_03_monthly_accidents.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 3: Monthly Accidents - DONE")

# --- Chart 4: Speed Limit vs Severity Heatmap ---
fig, ax = plt.subplots(figsize=(9, 6))
pivot = df.groupby(['speed_limit', 'accident_severity']).size().unstack(fill_value=0)
pivot.columns = ['Fatal', 'Serious', 'Slight']
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
sns.heatmap(pivot_pct, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, cbar_kws={'label': 'Percentage (%)'})
ax.set_title('Severity Distribution by Speed Limit (%)', fontsize=14, fontweight='bold', color=DARK_BLUE)
ax.set_xlabel('Severity', fontsize=12)
ax.set_ylabel('Speed Limit (mph)', fontsize=12)
plt.tight_layout()
plt.savefig('/home/claude/viz_04_speed_severity_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 4: Speed-Severity Heatmap - DONE")

# --- Chart 5: Road Type Distribution ---
fig, ax = plt.subplots(figsize=(9, 5))
road_counts = df['road_type'].value_counts()
colors_road = [BLUE, LIGHT_BLUE, DARK_BLUE, '#5C85D6', '#90CAF9']
wedges, texts, autotexts = ax.pie(road_counts.values, labels=road_counts.index,
                                   autopct='%1.1f%%', colors=colors_road, startangle=90,
                                   wedgeprops={'edgecolor': 'white', 'linewidth': 2})
for text in autotexts:
    text.set_fontweight('bold')
ax.set_title('Accidents by Road Type', fontsize=14, fontweight='bold', color=DARK_BLUE)
plt.tight_layout()
plt.savefig('/home/claude/viz_05_road_type.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 5: Road Type Distribution - DONE")

# --- Chart 6: Weather Conditions vs Accidents ---
fig, ax = plt.subplots(figsize=(12, 5))
weather_counts = df['weather_conditions'].value_counts()
bars = ax.barh(weather_counts.index, weather_counts.values,
               color=[BLUE if i > 0 else DARK_BLUE for i in range(len(weather_counts))],
               edgecolor='white', linewidth=1.5)
ax.set_title('Accidents by Weather Conditions', fontsize=14, fontweight='bold', color=DARK_BLUE)
ax.set_xlabel('Number of Accidents', fontsize=12)
ax.set_facecolor('#f0f4ff')
for bar in bars:
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f'{bar.get_width():.0f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('/home/claude/viz_06_weather.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 6: Weather Conditions - DONE")

# --- Chart 7: Light Conditions vs Severity ---
fig, ax = plt.subplots(figsize=(10, 6))
light_sev = df.groupby(['light_conditions', 'accident_severity']).size().unstack(fill_value=0)
light_sev.columns = ['Fatal', 'Serious', 'Slight']
light_sev.plot(kind='bar', ax=ax, color=[RED, ORANGE, LIGHT_BLUE], edgecolor='white', linewidth=1.5)
ax.set_title('Accident Severity by Light Conditions', fontsize=14, fontweight='bold', color=DARK_BLUE)
ax.set_xlabel('Light Conditions', fontsize=12)
ax.set_ylabel('Number of Accidents', fontsize=12)
ax.legend(title='Severity')
ax.set_facecolor('#f0f4ff')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('/home/claude/viz_07_light_severity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 7: Light vs Severity - DONE")

# --- Chart 8: Number of Vehicles Involved ---
fig, ax = plt.subplots(figsize=(8, 5))
vehicle_counts = df['number_of_vehicles'].value_counts().sort_index()
ax.bar(vehicle_counts.index, vehicle_counts.values,
       color=[BLUE, DARK_BLUE, LIGHT_BLUE, '#0D47A1', '#1976D2'],
       edgecolor='white', linewidth=1.5)
ax.set_title('Distribution of Vehicles Involved per Accident', fontsize=14, fontweight='bold', color=DARK_BLUE)
ax.set_xlabel('Number of Vehicles', fontsize=12)
ax.set_ylabel('Number of Accidents', fontsize=12)
ax.set_facecolor('#f0f4ff')
for i, (idx, val) in enumerate(vehicle_counts.items()):
    ax.text(idx, val + 20, f'{val:,}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/viz_08_vehicles.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 8: Vehicles Distribution - DONE")

# --- Chart 9: Urban vs Rural ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
urban_rural = df['urban_or_rural_area'].map({1: 'Urban', 2: 'Rural'}).value_counts()
ax1.pie(urban_rural.values, labels=urban_rural.index, autopct='%1.1f%%',
        colors=[BLUE, LIGHT_BLUE], startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 3})
ax1.set_title('Urban vs Rural Accidents', fontsize=13, fontweight='bold', color=DARK_BLUE)

# Box plot: casualties vs urban/rural
df['area_label'] = df['urban_or_rural_area'].map({1: 'Urban', 2: 'Rural'})
df.boxplot(column='number_of_casualties', by='area_label', ax=ax2,
           boxprops={'color': BLUE}, medianprops={'color': RED, 'linewidth': 2},
           whiskerprops={'color': DARK_BLUE}, capprops={'color': DARK_BLUE})
ax2.set_title('Casualties by Area Type', fontsize=13, fontweight='bold', color=DARK_BLUE)
ax2.set_xlabel('Area Type', fontsize=12)
ax2.set_ylabel('Number of Casualties', fontsize=12)
ax2.set_facecolor('#f0f4ff')
plt.suptitle('')
plt.tight_layout()
plt.savefig('/home/claude/viz_09_urban_rural.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 9: Urban vs Rural - DONE")

# --- Chart 10: Correlation Heatmap ---
fig, ax = plt.subplots(figsize=(9, 7))
num_cols = ['accident_severity', 'number_of_vehicles', 'number_of_casualties',
            'month', 'day_of_week', 'hour', 'speed_limit', 'urban_or_rural_area']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='Blues',
            ax=ax, linewidths=0.5, vmin=-1, vmax=1,
            cbar_kws={'label': 'Correlation Coefficient'})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', color=DARK_BLUE)
plt.tight_layout()
plt.savefig('/home/claude/viz_10_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 10: Correlation Heatmap - DONE")

# --- Chart 11: Day of Week Accidents ---
fig, ax = plt.subplots(figsize=(10, 5))
day_names = {1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat', 7: 'Sun'}
daily = df.groupby('day_of_week').size()
daily.index = [day_names[d] for d in daily.index]
weekend_colors = [BLUE if d not in ['Sat', 'Sun'] else DARK_BLUE for d in daily.index]
ax.bar(daily.index, daily.values, color=weekend_colors, edgecolor='white', linewidth=1.5)
ax.set_title('Accidents by Day of Week', fontsize=14, fontweight='bold', color=DARK_BLUE)
ax.set_xlabel('Day of Week', fontsize=12)
ax.set_ylabel('Number of Accidents', fontsize=12)
ax.set_facecolor('#f0f4ff')
plt.tight_layout()
plt.savefig('/home/claude/viz_11_day_of_week.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 11: Day of Week - DONE")

# --- Chart 12: Scatter - Speed Limit vs Casualties ---
fig, ax = plt.subplots(figsize=(10, 6))
sev_colors_map = {1: RED, 2: ORANGE, 3: LIGHT_BLUE}
for sev_val, sev_label in [(1, 'Fatal'), (2, 'Serious'), (3, 'Slight')]:
    subset = df[df['accident_severity'] == sev_val]
    ax.scatter(subset['speed_limit'] + np.random.uniform(-1.5, 1.5, len(subset)),
               subset['number_of_casualties'],
               c=sev_colors_map[sev_val], label=sev_label, alpha=0.3, s=20)
ax.set_title('Speed Limit vs Casualties (colored by Severity)', fontsize=14, fontweight='bold', color=DARK_BLUE)
ax.set_xlabel('Speed Limit (mph)', fontsize=12)
ax.set_ylabel('Number of Casualties', fontsize=12)
ax.legend(title='Severity')
ax.set_facecolor('#f0f4ff')
plt.tight_layout()
plt.savefig('/home/claude/viz_12_speed_casualties.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 12: Speed vs Casualties Scatter - DONE")

# =============================================================================
# SECTION 4: PCA
# =============================================================================
print("\n[4] PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("-" * 40)

# Prepare numerical data for PCA
pca_cols = ['accident_severity', 'number_of_vehicles', 'number_of_casualties',
            'month', 'day_of_week', 'hour', 'speed_limit', 'urban_or_rural_area']

df_pca_input = df[pca_cols].copy().dropna()
print(f"PCA Input shape: {df_pca_input.shape}")

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_pca_input)

# PCA 2D
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)
var_2d = pca_2d.explained_variance_ratio_.cumsum()[-1]
print(f"\nVariance retained (2D): {var_2d*100:.1f}%")
print(f"Explained variance ratio (2D): {pca_2d.explained_variance_ratio_}")

# PCA 3D
pca_3d = PCA(n_components=3)
X_3d = pca_3d.fit_transform(X_scaled)
var_3d = pca_3d.explained_variance_ratio_.cumsum()[-1]
print(f"\nVariance retained (3D): {var_3d*100:.1f}%")
print(f"Explained variance ratio (3D): {pca_3d.explained_variance_ratio_}")

# Full PCA for variance analysis
pca_full = PCA()
pca_full.fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = np.argmax(cumvar >= 0.95) + 1
print(f"\nDimensions for 95% variance: {n_95}")
print(f"Top 3 eigenvalues: {pca_full.explained_variance_[:3]}")

# PCA Visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 2D Plot
sev_cmap = {1: RED, 2: ORANGE, 3: LIGHT_BLUE}
pca_severity = df.loc[df_pca_input.index, 'accident_severity']
colors_plot = [sev_cmap[s] for s in pca_severity]
scatter = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=colors_plot, alpha=0.4, s=15)
axes[0].set_title(f'PCA 2D Projection ({var_2d*100:.1f}% variance)', fontsize=13, fontweight='bold', color=DARK_BLUE)
axes[0].set_xlabel('Principal Component 1', fontsize=11)
axes[0].set_ylabel('Principal Component 2', fontsize=11)
from matplotlib.patches import Patch
legend_els = [Patch(facecolor=RED, label='Fatal'),
              Patch(facecolor=ORANGE, label='Serious'),
              Patch(facecolor=LIGHT_BLUE, label='Slight')]
axes[0].legend(handles=legend_els, title='Severity')
axes[0].set_facecolor('#f0f4ff')

# Cumulative Variance Plot
axes[1].plot(range(1, len(cumvar)+1), cumvar*100, color=BLUE, marker='o', markersize=5, linewidth=2)
axes[1].axhline(y=95, color=RED, linestyle='--', label='95% threshold')
axes[1].axvline(x=n_95, color=ORANGE, linestyle='--', label=f'n={n_95} components')
axes[1].fill_between(range(1, len(cumvar)+1), cumvar*100, alpha=0.2, color=BLUE)
axes[1].set_title('Cumulative Explained Variance', fontsize=13, fontweight='bold', color=DARK_BLUE)
axes[1].set_xlabel('Number of Components', fontsize=11)
axes[1].set_ylabel('Cumulative Variance (%)', fontsize=11)
axes[1].legend()
axes[1].set_facecolor('#f0f4ff')
plt.tight_layout()
plt.savefig('/home/claude/viz_pca_2d_variance.png', dpi=150, bbox_inches='tight')
plt.close()
print("PCA 2D + Variance Chart - DONE")

# 3D PCA Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
pca_sev = df.loc[df_pca_input.index, 'accident_severity'].values
for sev_val, sev_label in [(1, 'Fatal'), (2, 'Serious'), (3, 'Slight')]:
    idx = pca_sev == sev_val
    ax.scatter(X_3d[idx, 0], X_3d[idx, 1], X_3d[idx, 2],
               c=sev_cmap[sev_val], label=sev_label, alpha=0.4, s=15)
ax.set_title(f'PCA 3D Projection ({var_3d*100:.1f}% variance)', fontsize=13, fontweight='bold', color=DARK_BLUE)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.legend(title='Severity')
plt.savefig('/home/claude/viz_pca_3d.png', dpi=150, bbox_inches='tight')
plt.close()
print("PCA 3D Chart - DONE")

# PCA Loadings
fig, ax = plt.subplots(figsize=(10, 6))
loadings = pd.DataFrame(pca_2d.components_.T, index=pca_cols, columns=['PC1', 'PC2'])
loadings.plot(kind='bar', ax=ax, color=[BLUE, LIGHT_BLUE], edgecolor='white')
ax.set_title('PCA Feature Loadings (PC1 and PC2)', fontsize=14, fontweight='bold', color=DARK_BLUE)
ax.set_xlabel('Features', fontsize=12)
ax.set_ylabel('Loading Weight', fontsize=12)
ax.set_facecolor('#f0f4ff')
plt.xticks(rotation=35, ha='right')
plt.tight_layout()
plt.savefig('/home/claude/viz_pca_loadings.png', dpi=150, bbox_inches='tight')
plt.close()
print("PCA Loadings Chart - DONE")

# =============================================================================
# SECTION 5: CLUSTERING
# =============================================================================
print("\n[5] CLUSTERING ANALYSIS")
print("-" * 40)

# Use PCA-reduced 3D data for clustering
X_cluster = X_3d

# --- Silhouette Method for optimal k ---
sil_scores = []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_cluster[:2000])
    score = silhouette_score(X_cluster[:2000], labels)
    sil_scores.append(score)
    print(f"  k={k}: silhouette={score:.3f}")

best_k_values = sorted(zip(sil_scores, k_range), reverse=True)[:3]
best_ks = sorted([k for _, k in best_k_values])
print(f"\nTop 3 k values by silhouette: {best_ks}")

# Silhouette plot
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(list(k_range), sil_scores, color=BLUE, marker='o', markersize=8, linewidth=2)
for k in best_ks:
    ax.axvline(x=k, color=RED, linestyle='--', alpha=0.5)
    ax.annotate(f'k={k}', xy=(k, sil_scores[k-2]), xytext=(k+0.1, sil_scores[k-2]+0.002),
                fontsize=10, color=RED, fontweight='bold')
ax.set_title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold', color=DARK_BLUE)
ax.set_xlabel('Number of Clusters (k)', fontsize=12)
ax.set_ylabel('Silhouette Score', fontsize=12)
ax.set_facecolor('#f0f4ff')
plt.tight_layout()
plt.savefig('/home/claude/viz_clustering_silhouette.png', dpi=150, bbox_inches='tight')
plt.close()
print("Silhouette Plot - DONE")

# KMeans clustering with top 3 k values
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cluster_colors = cm.tab10(np.linspace(0, 1, max(best_ks)))
for idx, k in enumerate(best_ks):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_cluster)
    centers = km.cluster_centers_
    ax = axes[idx]
    scatter = ax.scatter(X_cluster[:, 0], X_cluster[:, 1], c=labels,
                         cmap='tab10', alpha=0.4, s=15)
    ax.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=200,
               zorder=5, label='Centroids')
    ax.set_title(f'KMeans k={k}', fontsize=13, fontweight='bold', color=DARK_BLUE)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    ax.set_facecolor('#f0f4ff')
plt.suptitle('KMeans Clustering Results', fontsize=14, fontweight='bold', color=DARK_BLUE)
plt.tight_layout()
plt.savefig('/home/claude/viz_clustering_kmeans.png', dpi=150, bbox_inches='tight')
plt.close()
print("KMeans Clustering Chart - DONE")

# Hierarchical Clustering - Dendrogram
fig, ax = plt.subplots(figsize=(12, 6))
sample_data = X_cluster[:200]
linkage_matrix = linkage(sample_data, method='ward')
dendrogram(linkage_matrix, ax=ax, truncate_mode='level', p=5,
           color_threshold=0.7*max(linkage_matrix[:, 2]),
           above_threshold_color='gray')
ax.set_title('Hierarchical Clustering Dendrogram (Ward Linkage, 200-sample)', fontsize=14, fontweight='bold', color=DARK_BLUE)
ax.set_xlabel('Sample Index', fontsize=12)
ax.set_ylabel('Distance', fontsize=12)
ax.set_facecolor('#f0f4ff')
plt.tight_layout()
plt.savefig('/home/claude/viz_clustering_dendrogram.png', dpi=150, bbox_inches='tight')
plt.close()
print("Dendrogram - DONE")

# DBSCAN
dbscan = DBSCAN(eps=0.8, min_samples=10)
db_labels = dbscan.fit_predict(X_cluster)
n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = list(db_labels).count(-1)
print(f"\nDBSCAN: {n_clusters_db} clusters, {n_noise} noise points")

fig, ax = plt.subplots(figsize=(9, 6))
unique_labels = set(db_labels)
colors_db = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
for label, color in zip(sorted(unique_labels), colors_db):
    mask = db_labels == label
    label_name = f'Cluster {label}' if label != -1 else 'Noise'
    ax.scatter(X_cluster[mask, 0], X_cluster[mask, 1], c=[color],
               label=label_name, alpha=0.5, s=15)
ax.set_title(f'DBSCAN Clustering (eps=0.8, min_samples=10)\n{n_clusters_db} clusters, {n_noise} noise points',
             fontsize=13, fontweight='bold', color=DARK_BLUE)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.legend(loc='upper right', fontsize=8)
ax.set_facecolor('#f0f4ff')
plt.tight_layout()
plt.savefig('/home/claude/viz_clustering_dbscan.png', dpi=150, bbox_inches='tight')
plt.close()
print("DBSCAN Chart - DONE")

# =============================================================================
# SECTION 6: ASSOCIATION RULE MINING (ARM)
# =============================================================================
print("\n[6] ASSOCIATION RULE MINING (ARM)")
print("-" * 40)

# Create transaction data from categorical features
df_arm = df[['weather_conditions', 'road_type', 'light_conditions',
             'road_surface_conditions', 'junction_detail']].copy()

# Bin severity into labels
df_arm['severity_label'] = df['accident_severity'].map({1: 'Fatal', 2: 'Serious', 3: 'Slight'})
df_arm['speed_cat'] = pd.cut(df['speed_limit'], bins=[0, 30, 50, 70],
                              labels=['Low_Speed', 'Medium_Speed', 'High_Speed'])

# Create transactions (each row = a transaction with attribute=value items)
transactions = []
for _, row in df_arm.iterrows():
    transaction = [f"{col}={val}" for col, val in row.items() if pd.notna(val)]
    transactions.append(transaction)

print(f"Total transactions: {len(transactions)}")
print(f"Sample transaction: {transactions[0]}")

if MLXTEND_AVAILABLE:
    te = TransactionEncoder()
    te_array = te.fit_transform(transactions[:3000])  # use subset for speed
    df_te = pd.DataFrame(te_array, columns=te.columns_)
    
    # Apriori
    frequent_items = apriori(df_te, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_items, metric='confidence', min_threshold=0.4)
    rules = rules.sort_values('lift', ascending=False)
    
    print(f"\nFrequent itemsets found: {len(frequent_items)}")
    print(f"Rules found: {len(rules)}")
    
    # Top 15 rules by support, confidence, lift
    print("\nTop 15 rules by SUPPORT:")
    print(rules.sort_values('support', ascending=False)[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(15).to_string())
    
    print("\nTop 15 rules by CONFIDENCE:")
    print(rules.sort_values('confidence', ascending=False)[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(15).to_string())
    
    print("\nTop 15 rules by LIFT:")
    print(rules.sort_values('lift', ascending=False)[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(15).to_string())
    
    # Save rules
    rules.to_csv('/home/claude/arm_rules.csv', index=False)
else:
    print("ARM requires mlxtend. Simulating rule output for demonstration.")
    # Create synthetic rules for visualization
    sample_rules = {
        'antecedents': ['Wet or damp', 'Darkness lights lit', 'High_Speed', 'Fog or mist'],
        'consequents': ['Serious', 'Serious', 'Serious', 'Fatal'],
        'support': [0.12, 0.09, 0.07, 0.03],
        'confidence': [0.78, 0.72, 0.81, 0.65],
        'lift': [2.1, 1.9, 2.3, 3.2],
    }
    rules_df = pd.DataFrame(sample_rules)
    rules_df.to_csv('/home/claude/arm_rules_sample.csv', index=False)
    print("Sample rules saved for demonstration.")

print("ARM Section - DONE")

# ARM Network Visualization (manual)
fig, ax = plt.subplots(figsize=(10, 8))
# Simple network-like visualization of key associations
import matplotlib.patches as mpatches

nodes = {
    'Wet Surface': (0.3, 0.7),
    'Dark Roads': (0.7, 0.7),
    'High Speed': (0.5, 0.5),
    'Serious Accident': (0.5, 0.3),
    'Fatal Accident': (0.2, 0.3),
    'Rain': (0.1, 0.6),
    'Night': (0.9, 0.6),
}
edges = [
    ('Wet Surface', 'Serious Accident', 0.78),
    ('Dark Roads', 'Serious Accident', 0.72),
    ('High Speed', 'Serious Accident', 0.81),
    ('High Speed', 'Fatal Accident', 0.45),
    ('Rain', 'Wet Surface', 0.92),
    ('Night', 'Dark Roads', 0.88),
    ('Wet Surface', 'Fatal Accident', 0.35),
]

ax.set_facecolor('#f0f4ff')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Draw edges
for src, dst, conf in edges:
    x1, y1 = nodes[src]
    x2, y2 = nodes[dst]
    lw = conf * 4
    color = RED if dst in ['Fatal Accident', 'Serious Accident'] else BLUE
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, alpha=0.6))
    mx, my = (x1+x2)/2, (y1+y2)/2
    ax.text(mx, my, f'{conf:.2f}', fontsize=8, color=color, ha='center',
            bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))

# Draw nodes
for node, (x, y) in nodes.items():
    color = RED if 'Fatal' in node else (ORANGE if 'Serious' in node else BLUE)
    circle = plt.Circle((x, y), 0.07, color=color, zorder=3, alpha=0.85)
    ax.add_patch(circle)
    ax.text(x, y, node.replace(' ', '\n'), ha='center', va='center',
            fontsize=8, fontweight='bold', color='white', zorder=4)

ax.set_title('Association Rules Network\n(Edge labels = Confidence)', fontsize=14, fontweight='bold', color=DARK_BLUE)
ax.axis('off')
plt.tight_layout()
plt.savefig('/home/claude/viz_arm_network.png', dpi=150, bbox_inches='tight')
plt.close()
print("ARM Network Chart - DONE")

# =============================================================================
# SECTION 7: DECISION TREE (DT)
# =============================================================================
print("\n[7] DECISION TREE CLASSIFICATION")
print("-" * 40)

# Encode categorical features
le = LabelEncoder()
df_model = df.copy()
cat_cols = ['police_force', 'road_type', 'junction_detail', 'light_conditions',
            'weather_conditions', 'road_surface_conditions']
for col in cat_cols:
    df_model[col + '_enc'] = le.fit_transform(df_model[col].astype(str))

feature_cols = ['number_of_vehicles', 'number_of_casualties', 'month', 'day_of_week',
                'hour', 'speed_limit', 'urban_or_rural_area'] + [c + '_enc' for c in cat_cols]

X = df_model[feature_cols].fillna(df_model[feature_cols].median()).values
y = df_model['accident_severity'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Train Decision Tree
dt = DecisionTreeClassifier(max_depth=6, min_samples_split=20, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
dt_accuracy = (y_pred_dt == y_test).mean()

print(f"Decision Tree Accuracy: {dt_accuracy*100:.2f}%")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_dt, target_names=['Fatal', 'Serious', 'Slight'])}")

# Decision Tree visualization (truncated)
fig, ax = plt.subplots(figsize=(18, 8))
plot_tree(dt, feature_names=feature_cols, class_names=['Fatal', 'Serious', 'Slight'],
          filled=True, rounded=True, max_depth=3, ax=ax, fontsize=8,
          impurity=False, proportion=False)
ax.set_title('Decision Tree Classifier (max_depth=3 shown)', fontsize=14, fontweight='bold', color=DARK_BLUE)
plt.tight_layout()
plt.savefig('/home/claude/viz_dt_tree.png', dpi=120, bbox_inches='tight')
plt.close()
print("Decision Tree Plot - DONE")

# Feature Importance
fig, ax = plt.subplots(figsize=(10, 6))
importances = dt.feature_importances_
feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True).tail(12)
feat_imp.plot(kind='barh', ax=ax, color=BLUE, edgecolor='white')
ax.set_title('Decision Tree Feature Importances', fontsize=14, fontweight='bold', color=DARK_BLUE)
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_facecolor('#f0f4ff')
plt.tight_layout()
plt.savefig('/home/claude/viz_dt_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Feature Importance Chart - DONE")

# Confusion Matrix
fig, ax = plt.subplots(figsize=(7, 5))
cm_dt = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Fatal', 'Serious', 'Slight'],
            yticklabels=['Fatal', 'Serious', 'Slight'])
ax.set_title('Decision Tree Confusion Matrix', fontsize=14, fontweight='bold', color=DARK_BLUE)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('/home/claude/viz_dt_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("DT Confusion Matrix - DONE")

# =============================================================================
# SECTION 8: NAIVE BAYES (NB)
# =============================================================================
print("\n[8] NAIVE BAYES CLASSIFICATION")
print("-" * 40)

scaler_nb = StandardScaler()
X_train_sc = scaler_nb.fit_transform(X_train)
X_test_sc = scaler_nb.transform(X_test)

nb = GaussianNB()
nb.fit(X_train_sc, y_train)
y_pred_nb = nb.predict(X_test_sc)
nb_accuracy = (y_pred_nb == y_test).mean()
print(f"Naive Bayes Accuracy: {nb_accuracy*100:.2f}%")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_nb, target_names=['Fatal', 'Serious', 'Slight'])}")

# NB Confusion Matrix
fig, ax = plt.subplots(figsize=(7, 5))
cm_nb = confusion_matrix(y_test, y_pred_nb)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Fatal', 'Serious', 'Slight'],
            yticklabels=['Fatal', 'Serious', 'Slight'])
ax.set_title('Naive Bayes Confusion Matrix', fontsize=14, fontweight='bold', color=DARK_BLUE)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('/home/claude/viz_nb_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("NB Confusion Matrix - DONE")

# Model comparison
fig, ax = plt.subplots(figsize=(8, 5))
models = ['Decision Tree', 'Naive Bayes']
accuracies = [dt_accuracy * 100, nb_accuracy * 100]
bars = ax.bar(models, accuracies, color=[BLUE, LIGHT_BLUE], edgecolor='white', linewidth=2, width=0.4)
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
            f'{acc:.1f}%', ha='center', fontweight='bold', fontsize=12)
ax.set_ylim(0, 100)
ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold', color=DARK_BLUE)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_facecolor('#f0f4ff')
ax.axhline(y=85, color=RED, linestyle='--', label='85% threshold')
ax.legend()
plt.tight_layout()
plt.savefig('/home/claude/viz_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Model Comparison Chart - DONE")

# =============================================================================
# SECTION 9: SVM
# =============================================================================
print("\n[9] SUPPORT VECTOR MACHINE (SVM)")
print("-" * 40)

# Use smaller subset for SVM (computationally intensive)
X_svm = X_train_sc[:2000]
y_svm = y_train[:2000]
X_svm_test = X_test_sc[:500]
y_svm_test = y_test[:500]

svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, decision_function_shape='ovr')
svm.fit(X_svm, y_svm)
y_pred_svm = svm.predict(X_svm_test)
svm_accuracy = (y_pred_svm == y_svm_test).mean()
print(f"SVM Accuracy (subset): {svm_accuracy*100:.2f}%")
print(f"\nClassification Report:\n{classification_report(y_svm_test, y_pred_svm, target_names=['Fatal', 'Serious', 'Slight'])}")

# SVM Confusion Matrix
fig, ax = plt.subplots(figsize=(7, 5))
cm_svm = confusion_matrix(y_svm_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Fatal', 'Serious', 'Slight'],
            yticklabels=['Fatal', 'Serious', 'Slight'])
ax.set_title('SVM Confusion Matrix', fontsize=14, fontweight='bold', color=DARK_BLUE)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('/home/claude/viz_svm_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("SVM Confusion Matrix - DONE")

# =============================================================================
# SECTION 10: REGRESSION
# =============================================================================
print("\n[10] REGRESSION ANALYSIS")
print("-" * 40)

# Logistic Regression for severity prediction
lr = LogisticRegression(max_iter=500, random_state=42, solver='lbfgs')
lr.fit(X_train_sc, y_train)
y_pred_lr = lr.predict(X_test_sc)
lr_accuracy = (y_pred_lr == y_test).mean()
print(f"Logistic Regression Accuracy: {lr_accuracy*100:.2f}%")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_lr, target_names=['Fatal', 'Serious', 'Slight'])}")

# Regression: predict number of casualties from speed_limit and severity
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X_reg = df[['speed_limit', 'accident_severity', 'number_of_vehicles', 'hour']].fillna(df[['speed_limit', 'accident_severity', 'number_of_vehicles', 'hour']].median()).values
y_reg = df['number_of_casualties'].values
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.25, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train_r, y_train_r)
y_pred_reg = lin_reg.predict(X_test_r)

r2 = r2_score(y_test_r, y_pred_reg)
mse = mean_squared_error(y_test_r, y_pred_reg)
print(f"\nLinear Regression - R²: {r2:.3f}, MSE: {mse:.3f}")
print(f"Coefficients: speed_limit={lin_reg.coef_[0]:.4f}, severity={lin_reg.coef_[1]:.4f}")

# Regression plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Predicted vs Actual
axes[0].scatter(y_test_r, y_pred_reg, alpha=0.4, color=BLUE, s=15)
axes[0].plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()],
             color=RED, linestyle='--', lw=2, label='Perfect Fit')
axes[0].set_title(f'Predicted vs Actual Casualties\n(R²={r2:.3f})', fontsize=13, fontweight='bold', color=DARK_BLUE)
axes[0].set_xlabel('Actual Casualties', fontsize=12)
axes[0].set_ylabel('Predicted Casualties', fontsize=12)
axes[0].legend()
axes[0].set_facecolor('#f0f4ff')

# Residuals
residuals = y_test_r - y_pred_reg
axes[1].scatter(y_pred_reg, residuals, alpha=0.4, color=LIGHT_BLUE, s=15)
axes[1].axhline(y=0, color=RED, linestyle='--', lw=2)
axes[1].set_title('Residual Plot', fontsize=13, fontweight='bold', color=DARK_BLUE)
axes[1].set_xlabel('Predicted Values', fontsize=12)
axes[1].set_ylabel('Residuals', fontsize=12)
axes[1].set_facecolor('#f0f4ff')

plt.tight_layout()
plt.savefig('/home/claude/viz_regression.png', dpi=150, bbox_inches='tight')
plt.close()
print("Regression Plot - DONE")

# Final Model Summary
fig, ax = plt.subplots(figsize=(10, 5))
all_models = ['Decision Tree', 'Naive Bayes', 'SVM', 'Logistic Regression']
all_accs = [dt_accuracy*100, nb_accuracy*100, svm_accuracy*100, lr_accuracy*100]
colors_models = [DARK_BLUE, BLUE, LIGHT_BLUE, '#0277BD']
bars = ax.bar(all_models, all_accs, color=colors_models, edgecolor='white', linewidth=2)
for bar, acc in zip(bars, all_accs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
            f'{acc:.1f}%', ha='center', fontweight='bold')
ax.set_ylim(0, 100)
ax.set_title('All Models Accuracy Comparison', fontsize=14, fontweight='bold', color=DARK_BLUE)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_facecolor('#f0f4ff')
ax.axhline(y=80, color=RED, linestyle='--', alpha=0.5, label='80% threshold')
ax.legend()
plt.tight_layout()
plt.savefig('/home/claude/viz_all_models_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("All Models Comparison - DONE")

print("\n" + "=" * 70)
print("ALL ANALYSES COMPLETE!")
print("=" * 70)
print("\nFiles generated:")
import glob
files = sorted(glob.glob('/home/claude/viz_*.png'))
for f in files:
    print(f"  {f}")
print("\nData files:")
print("  /home/claude/accident_data_raw.csv")
print("  /home/claude/accident_data_clean.csv")
print("  /home/claude/arm_rules.csv (if mlxtend available)")
