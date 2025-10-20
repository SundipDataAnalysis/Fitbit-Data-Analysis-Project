# Fitbit Activity Data Analysis 📊

A comprehensive analysis of Fitbit user activity data to uncover patterns in daily movement, calorie expenditure, and activity levels. This project demonstrates data cleaning, exploratory data analysis, and visualization techniques using Python.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)
- [Analysis Methodology](#analysis-methodology)
- [Visualizations](#visualizations)
- [Business Recommendations](#business-recommendations)
- [Installation & Usage](#installation--usage)
- [Future Enhancements](#future-enhancements)
- [Contact](#contact)

## 🎯 Project Overview

This project analyzes Fitbit activity tracker data from 33 users over a one-month period (April 12 - May 12, 2016) to understand:
- Daily activity patterns and step counts
- Correlation between physical activity and calorie expenditure
- User behavior across different days of the week
- Time allocation across various activity intensity levels

**Key Question:** How do different activity levels impact calorie burn, and what patterns emerge in user behavior?

## 📊 Dataset

**Source:** Kaggle Fitbit Fitness Tracker Data
- **Users:** 33 unique participants
- **Time Period:** April 12, 2016 - May 12, 2016 (31 days)
- **Records:** 940 daily activity entries
- **Files Used:** `dailyActivity_merged.csv`

### Features Analyzed
- Total daily steps
- Distance traveled
- Active minutes (very active, fairly active, lightly active)
- Sedentary minutes
- Calories burned
- Day of week patterns

## 🔍 Key Findings

### 1. Activity Level Distribution
Users were categorized into three groups based on average daily steps:
- **Sedentary:** < 6,000 steps/day
- **Active:** 6,000 - 12,000 steps/day
- **Very Active:** ≥ 12,000 steps/day

### 2. Time Allocation
- **81.3%** of time spent sedentary (⚠️ Major health concern)
- **16.0%** lightly active
- **1.1%** fairly active
- **1.7%** very active

### 3. Steps vs Calories Burned
- **Strong positive correlation** (r > 0.6) between total steps and calories
- Very active users consistently burn more calories
- Average: 7,638 steps/day, 2,304 calories/day

### 4. Weekly Patterns
- **Most Active Days:** Tuesday and Saturday
- **Least Active Day:** Sunday (below average)
- Opportunity for targeted Sunday engagement strategies

### 5. Activity-Calorie Relationships
- **Very active minutes:** Strong positive correlation with calorie burn
- **Fairly active minutes:** Moderate positive correlation
- **Sedentary minutes:** Negative correlation with calorie expenditure
- **Insight:** Even small increases in activity level drive meaningful calorie burn

## 🛠️ Technologies Used

```python
- Python 3.10
- pandas - Data manipulation and analysis
- NumPy - Numerical computing
- Matplotlib - Data visualization
- Seaborn - Statistical graphics
- datetime - Date/time handling
```

## 📈 Analysis Methodology

### 1. Data Cleaning & Preparation
```python
- Converted ID to string type for categorical treatment
- Parsed dates to datetime format (MM/DD/YYYY)
- Standardized column names (snake_case)
- Created derived features (day_of_week, activity_level)
- Verified data integrity (no nulls, no duplicates)
```

### 2. Feature Engineering
- **Activity Level Classification:** Categorized users based on average daily steps
- **Day of Week:** Extracted weekday names and numeric values
- **Distance Validation:** Calculated difference between total and tracker distance

### 3. Exploratory Data Analysis
- Correlation analysis between activity metrics
- Time series patterns across days of the week
- Group-level comparisons by activity level

### 4. Visualization
- Scatter plots for correlation analysis
- Bar charts for day-of-week comparisons
- Pie charts for time allocation
- Multi-panel plots for comprehensive activity-calorie relationships

## 📸 Visualizations

### Steps vs Calories Correlation
Demonstrates clear positive relationship between daily steps and calorie expenditure, with activity level segments showing distinct patterns.

### Average Steps by Day of Week
Bar chart revealing weekly activity patterns, with reference line showing overall average (7,638 steps).

### Activity Time Distribution
Pie chart highlighting the alarming 81.3% sedentary time allocation.

### Activity Minutes vs Calories (4-Panel)
Comprehensive view showing how different intensity levels correlate with calorie burn.

## 💡 Business Recommendations

### For Product Teams
1. **Reduce Sedentary Time:**
   - Implement hourly movement reminders
   - Gamify "breaking up sitting time"
   - Create micro-challenges (e.g., "Move for 2 minutes")

2. **Weekend Engagement:**
   - Target Sunday with motivational notifications
   - Suggest weekend-specific activities
   - Create "Sunday Funday" challenge campaigns

3. **Personalized Goal Setting:**
   - Adaptive goals based on user baseline
   - Tiered challenges for different activity levels
   - Celebrate small wins (10-minute improvements)

### For Users/Health Professionals
1. Start with achievable goal: 7,400 steps/day (dataset average)
2. Focus on consistency over intensity
3. Break up sedentary time with light activity
4. Plan active Sunday routines to maintain weekly momentum


### Running the Analysis
1. Ensure the dataset is in the correct path: `/kaggle/input/fitbit/`
2. Run all cells sequentially
3. View generated visualizations and insights

## 🔮 Future Enhancements

### Planned Improvements
- [ ] Integrate sleep data analysis (`sleepDay_merged.csv`)
- [ ] Analyze weight trends (`weightLogInfo_merged.csv`)
- [ ] Hourly activity patterns using minute-level data
- [ ] Heart rate analysis (`heartrate_seconds_merged.csv`)
- [ ] Statistical significance testing (t-tests, ANOVA)
- [ ] Predictive modeling for user engagement/churn
- [ ] Interactive dashboard with Plotly or Streamlit
- [ ] Time series forecasting for activity trends
- [ ] Correlation heatmap for all metrics
- [ ] User segmentation with clustering (K-means)

### Advanced Analytics
- Machine learning model to predict calorie burn
- Anomaly detection for unusual activity patterns
- Cohort analysis for user retention
- A/B test framework for intervention effectiveness

-- Further analysis to consider in future applications

## 📝 Project Structure

```
fitbit-analysis/
│
├── README.md
├── fitbit_analysis.ipynb          # Main analysis notebook
├── requirements.txt               # Python dependencies
│
├── data/
│   └── dailyActivity_merged.csv   # Primary dataset
│
├── visualizations/
│   ├── steps_calories_correlation.png
│   ├── avg_steps_by_day.png
│   ├── activity_time_distribution.png
│   └── activity_calories_multiplot.png
│
└── docs/
    └── analysis_recommendations.md
```

## 📧 Contact

**Your Name**
- GitHub: [@SundipDataAnalysis](https://github.com/SundipDataAnalysis)
- LinkedIn:[ https://linkedin.com/in/yourprofile](https://www.linkedin.com/in/sundip-sharma-837515123/)
- Email: sundipsharma@hotmail.co.uk
- Portfolio:

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Dataset provided by [Kaggle Fitbit Fitness Tracker Data](https://www.kaggle.com/datasets/arashnic/fitbit)


