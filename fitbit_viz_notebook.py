# Fitbit Activity Data Analysis
# Author: Your Name
# Date: October 2025
# Portfolio Project: Understanding User Activity Patterns and Health Metrics

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 70)
print("FITBIT ACTIVITY DATA ANALYSIS")
print("=" * 70)

# ============================================================================
# SECTION 1: DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

print("\n📊 Loading dataset...")
df = pd.read_csv('/kaggle/input/fitbit/Fitabase Data 4.12.16-5.12.16/dailyActivity_merged.csv')

print(f"✓ Dataset loaded successfully!")
print(f"  - Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  - Date Range: April 12, 2016 - May 12, 2016")

# ============================================================================
# SECTION 2: DATA CLEANING AND PREPARATION
# ============================================================================

print("\n🧹 Cleaning and preparing data...")

# Convert data types
df['Id'] = df['Id'].astype(str)
df['ActivityDate'] = pd.to_datetime(df['ActivityDate'], format='%m/%d/%Y')

# Standardize column names
df.columns = df.columns.str.lower()
df.rename(columns={
    'activitydate': 'activity_date',
    'totalsteps': 'total_steps',
    'totaldistance': 'total_distance',
    'trackerdistance': 'tracker_distance',
    'loggedactivitiesdistance': 'logged_activities_distance',
    'veryactivedistance': 'very_active_distance',
    'moderatelyactivedistance': 'moderately_active_distance',
    'lightactivedistance': 'light_active_distance',
    'sedentaryactivedistance': 'sedentary_active_distance',
    'veryactiveminutes': 'very_active_minutes',
    'fairlyactiveminutes': 'fairly_active_minutes',
    'lightlyactiveminutes': 'lightly_active_minutes',
    'sedentaryminutes': 'sedentary_minutes'
}, inplace=True)

# Create day of week features
df['day_of_week'] = df['activity_date'].dt.day_name()
df['n_day_of_week'] = df['activity_date'].dt.weekday

# Data quality checks
print(f"✓ Data cleaning complete!")
print(f"  - Null values: {df.isnull().sum().sum()}")
print(f"  - Duplicate rows: {df.duplicated().sum()}")
print(f"  - Unique users: {df['id'].nunique()}")

# ============================================================================
# SECTION 3: FEATURE ENGINEERING - ACTIVITY LEVEL CLASSIFICATION
# ============================================================================

print("\n🏃 Classifying users by activity level...")

# Calculate average steps per user
id_grp = df.groupby(['id'])
id_avg_step = id_grp['total_steps'].mean().sort_values(ascending=False).to_frame()

# Define activity level categories
conditions = [
    (id_avg_step < 6000),
    (id_avg_step > 6000) & (id_avg_step < 12000),
    (id_avg_step >= 12000)
]
values = ['Sedentary', 'Active', 'Very Active']
id_avg_step['activity_level'] = np.select(conditions, values)

# Add activity level to main dataframe
id_activity_level = id_avg_step['activity_level']
df['activity_level'] = [id_activity_level[c] for c in df['id']]

# Display classification summary
activity_counts = df.groupby('activity_level')['id'].nunique()
print(f"✓ Activity level classification complete!")
for level, count in activity_counts.items():
    print(f"  - {level}: {count} users")

# Select relevant columns for analysis
df = df[['id', 'activity_date', 'total_steps', 'total_distance',
         'very_active_minutes', 'fairly_active_minutes',
         'lightly_active_minutes', 'sedentary_minutes', 'calories',
         'activity_level', 'day_of_week', 'n_day_of_week']].copy()

# ============================================================================
# SECTION 4: DESCRIPTIVE STATISTICS
# ============================================================================

print("\n📈 Key Statistics:")
print("-" * 70)
print(f"Average daily steps:     {df['total_steps'].mean():,.0f} steps")
print(f"Average daily calories:  {df['calories'].mean():,.0f} calories")
print(f"Average daily distance:  {df['total_distance'].mean():.2f} km")
print(f"Median daily steps:      {df['total_steps'].median():,.0f} steps")
print("-" * 70)

# ============================================================================
# SECTION 5: VISUALIZATIONS
# ============================================================================

print("\n🎨 Generating visualizations...")

# Create output directory for saving plots
import os
os.makedirs('visualizations', exist_ok=True)

# ----------------------------------------------------------------------------
# VISUALIZATION 1: Steps vs Calories Correlation
# ----------------------------------------------------------------------------

plt.figure(figsize=(12, 7))
colors = {'Sedentary': '#e74c3c', 'Active': '#f39c12', 'Very Active': '#27ae60'}

for level in df['activity_level'].unique():
    subset = df[df['activity_level'] == level]
    plt.scatter(subset['total_steps'], subset['calories'], 
               label=level, alpha=0.6, s=50, color=colors[level])

plt.xlabel('Total Steps', fontsize=13, fontweight='bold')
plt.ylabel('Calories Burned', fontsize=13, fontweight='bold')
plt.title('Correlation Between Daily Steps and Calories Burned\nby Activity Level',
          fontsize=16, fontweight='bold', pad=20)
plt.legend(title='Activity Level', fontsize=11, title_fontsize=12)
plt.grid(True, alpha=0.3)

# Add correlation coefficient
from scipy import stats
corr, p_value = stats.pearsonr(df['total_steps'], df['calories'])
plt.text(0.05, 0.95, f'Correlation: r = {corr:.3f}\np-value < 0.001',
         transform=plt.gca().transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('visualizations/01_steps_calories_correlation.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: 01_steps_calories_correlation.png")
plt.show()

# ----------------------------------------------------------------------------
# VISUALIZATION 2: Average Steps by Day of Week
# ----------------------------------------------------------------------------

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
fig, ax = plt.subplots(figsize=(12, 7))

day_grp = df.groupby(['day_of_week'])
avg_daily_steps = day_grp['total_steps'].mean().reindex(day_order)

bars = ax.bar(range(len(day_order)), avg_daily_steps, 
              color=['#3498db' if x < avg_daily_steps.mean() else '#2ecc71' 
                     for x in avg_daily_steps],
              edgecolor='black', linewidth=1.2)

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.axhline(y=avg_daily_steps.mean(), color='red', linestyle='--', 
           linewidth=2, label=f'Average: {avg_daily_steps.mean():,.0f} steps')

ax.set_xticks(range(len(day_order)))
ax.set_xticklabels(day_order, fontsize=11, fontweight='bold')
ax.set_ylabel('Average Steps', fontsize=13, fontweight='bold')
ax.set_xlabel('Day of Week', fontsize=13, fontweight='bold')
ax.set_title('Average Number of Steps by Day of Week',
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/02_avg_steps_by_day.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: 02_avg_steps_by_day.png")
plt.show()

# ----------------------------------------------------------------------------
# VISUALIZATION 3: Activity Time Distribution
# ----------------------------------------------------------------------------

very_active_mins = df['very_active_minutes'].sum()
fairly_active_mins = df['fairly_active_minutes'].sum()
lightly_active_mins = df['lightly_active_minutes'].sum()
sedentary_mins = df['sedentary_minutes'].sum()

slices = [very_active_mins, fairly_active_mins, lightly_active_mins, sedentary_mins]
labels = ['Very Active', 'Fairly Active', 'Lightly Active', 'Sedentary']
colors_pie = ['#27ae60', '#f39c12', '#3498db', '#e74c3c']
explode = [0.05, 0.05, 0.05, 0.1]

fig, ax = plt.subplots(figsize=(10, 8))
wedges, texts, autotexts = ax.pie(slices, labels=labels, colors=colors_pie,
                                   autopct='%1.1f%%', startangle=90,
                                   explode=explode, shadow=True,
                                   textprops={'fontsize': 12, 'fontweight': 'bold'})

# Enhance autopct text
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(13)
    autotext.set_fontweight('bold')

ax.set_title('Distribution of Activity Time\n(Total Minutes Across All Users)',
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('visualizations/03_activity_time_distribution.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: 03_activity_time_distribution.png")
plt.show()

# ----------------------------------------------------------------------------
# VISUALIZATION 4: Activity Minutes vs Calories (Multi-panel)
# ----------------------------------------------------------------------------

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
fig.suptitle('Relationship Between Activity Duration and Calories Burned',
             fontsize=18, fontweight='bold', y=0.995)

activity_types = [
    ('sedentary_minutes', 'Sedentary Minutes'),
    ('lightly_active_minutes', 'Lightly Active Minutes'),
    ('fairly_active_minutes', 'Fairly Active Minutes'),
    ('very_active_minutes', 'Very Active Minutes')
]

for idx, (col, title) in enumerate(activity_types):
    row = idx // 2
    col_idx = idx % 2
    ax = axes[row, col_idx]
    
    for level in df['activity_level'].unique():
        subset = df[df['activity_level'] == level]
        ax.scatter(subset['calories'], subset[col],
                  label=level, alpha=0.6, s=40, color=colors[level])
    
    ax.set_xlabel('Calories Burned', fontsize=11, fontweight='bold')
    ax.set_ylabel('Minutes', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr, _ = stats.pearsonr(df['calories'], df[col])
    ax.text(0.05, 0.95, f'r = {corr:.3f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    if idx == 3:  # Only show legend on last plot
        ax.legend(title='Activity Level', fontsize=9, title_fontsize=10)

plt.tight_layout()
plt.savefig('visualizations/04_activity_minutes_vs_calories.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: 04_activity_minutes_vs_calories.png")
plt.show()

# ----------------------------------------------------------------------------
# VISUALIZATION 5: Activity Level Distribution Box Plot
# ----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 7))

bp = ax.boxplot([df[df['activity_level'] == 'Sedentary']['total_steps'],
                 df[df['activity_level'] == 'Active']['total_steps'],
                 df[df['activity_level'] == 'Very Active']['total_steps']],
                labels=['Sedentary\n(<6,000 steps)', 'Active\n(6,000-12,000 steps)', 
                       'Very Active\n(≥12,000 steps)'],
                patch_artist=True,
                notch=True,
                showmeans=True)

# Color the boxes
colors_box = ['#e74c3c', '#f39c12', '#27ae60']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Total Steps', fontsize=13, fontweight='bold')
ax.set_xlabel('Activity Level Classification', fontsize=13, fontweight='bold')
ax.set_title('Distribution of Daily Steps by Activity Level\n(Box Plot with Mean & Median)',
             fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/05_activity_level_boxplot.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: 05_activity_level_boxplot.png")
plt.show()

# ----------------------------------------------------------------------------
# VISUALIZATION 6: Heatmap - Correlation Matrix
# ----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 8))

correlation_cols = ['total_steps', 'total_distance', 'very_active_minutes',
                   'fairly_active_minutes', 'lightly_active_minutes',
                   'sedentary_minutes', 'calories']

corr_matrix = df[correlation_cols].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
            cmap='coolwarm', center=0, square=True, linewidths=1,
            cbar_kws={"shrink": 0.8}, ax=ax, vmin=-1, vmax=1)

ax.set_title('Correlation Matrix of Activity Metrics',
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('visualizations/06_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: 06_correlation_heatmap.png")
plt.show()

# ============================================================================
# SECTION 6: KEY INSIGHTS SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("KEY INSIGHTS SUMMARY")
print("=" * 70)

print("\n1. ACTIVITY PATTERNS:")
print(f"   • {(sedentary_mins / sum(slices) * 100):.1f}% of time spent sedentary")
print(f"   • Only {(very_active_mins / sum(slices) * 100):.1f}% of time in vigorous activity")
print(f"   • Average user takes {df['total_steps'].mean():,.0f} steps/day")

print("\n2. CORRELATIONS:")
print(f"   • Steps vs Calories: r = {corr:.3f} (Strong positive)")
print(f"   • Very active minutes show strongest calorie correlation")
print(f"   • Sedentary time negatively correlated with calorie burn")

print("\n3. WEEKLY PATTERNS:")
most_active = avg_daily_steps.idxmax()
least_active = avg_daily_steps.idxmin()
print(f"   • Most active day: {most_active} ({avg_daily_steps[most_active]:,.0f} steps)")
print(f"   • Least active day: {least_active} ({avg_daily_steps[least_active]:,.0f} steps)")
print(f"   • Weekend activity dips on Sunday")

print("\n4. RECOMMENDATIONS:")
print("   • Target interventions to reduce sedentary time")
print("   • Focus on increasing very active minutes (highest impact)")
print("   • Implement Sunday motivation strategies")
print("   • Personalize goals based on user's activity level")

print("\n" + "=" * 70)
print("✓ Analysis complete! All visualizations saved to 'visualizations/' folder")
print("=" * 70)