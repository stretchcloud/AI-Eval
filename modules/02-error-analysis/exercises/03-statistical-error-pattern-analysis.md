# Exercise 3: Statistical Error Pattern Analysis

## Objective
Apply advanced statistical techniques to identify error patterns and correlations in AI system outputs, implementing sophisticated analytical methods to uncover hidden relationships, temporal trends, and systematic biases that inform targeted improvement strategies.

## Duration
3-4 hours

## Skills Developed
- Advanced statistical analysis for AI evaluation
- Pattern recognition and correlation analysis
- Data visualization for error analysis
- Hypothesis testing for evaluation insights
- Predictive modeling for error prevention

## Prerequisites
- Basic statistics knowledge and understanding of Section 4
- Understanding of qualitative research methods from Section 6
- Python programming experience with data analysis libraries
- Familiarity with statistical concepts (correlation, regression, hypothesis testing)

## Learning Outcomes
By completing this exercise, you will be able to:
- Apply sophisticated statistical methods to analyze AI system errors
- Identify complex patterns and correlations in evaluation data
- Create compelling visualizations that reveal error insights
- Develop predictive models for error prevention
- Design statistical frameworks for ongoing error monitoring

## Exercise Overview

This exercise guides you through comprehensive statistical analysis of AI evaluation data, implementing advanced techniques to uncover patterns that inform systematic improvements. You'll work with a realistic dataset of AI system evaluations, applying multiple statistical methods to extract actionable insights.

### Scenario
You're analyzing evaluation data from an AI-powered content moderation system that has been running for 6 months. The system processes user-generated content across multiple categories (text, images, videos) and makes moderation decisions. You have access to evaluation data including human reviewer assessments, system confidence scores, content characteristics, and temporal information.

Your goal is to identify statistical patterns that explain when and why the system makes errors, enabling targeted improvements to reduce false positives and false negatives.

### Dataset Description
The exercise uses a comprehensive dataset with:
- **6,000 moderation decisions** across 6 months
- **Multiple content types** (text, image, video)
- **Human reviewer assessments** (ground truth)
- **System predictions and confidence scores**
- **Content characteristics** (length, complexity, topic)
- **Temporal information** (time of day, day of week, seasonal patterns)
- **User context** (account age, previous violations, geographic region)

## Part 1: Data Preparation and Exploratory Analysis

### Step 1: Environment Setup and Data Loading

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr, mannwhitneyu
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configure display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

print("Statistical Error Pattern Analysis - Exercise 3")
print("=" * 60)
print("Environment configured successfully")
```

### Step 2: Generate Realistic Evaluation Dataset

```python
def generate_comprehensive_evaluation_dataset(n_samples=6000):
    """
    Generate realistic AI content moderation evaluation dataset with complex patterns.
    """
    
    np.random.seed(42)  # For reproducibility
    
    print("Generating comprehensive evaluation dataset...")
    
    # Time-based features
    start_date = pd.Timestamp('2024-01-01')
    dates = pd.date_range(start_date, periods=n_samples, freq='H')
    
    # Content types with different error patterns
    content_types = np.random.choice(['text', 'image', 'video'], n_samples, p=[0.6, 0.25, 0.15])
    
    # Content categories with varying difficulty
    categories = np.random.choice([
        'spam', 'harassment', 'hate_speech', 'violence', 'adult_content', 
        'misinformation', 'copyright', 'self_harm', 'drugs', 'legitimate'
    ], n_samples, p=[0.15, 0.12, 0.08, 0.06, 0.05, 0.07, 0.04, 0.03, 0.02, 0.38])
    
    # User context features
    user_account_age_days = np.random.exponential(365, n_samples)  # Exponential distribution
    user_previous_violations = np.random.poisson(2, n_samples)  # Poisson distribution
    user_regions = np.random.choice(['NA', 'EU', 'APAC', 'LATAM', 'MEA'], n_samples, p=[0.35, 0.25, 0.20, 0.12, 0.08])
    
    # Content characteristics
    content_length = np.random.lognormal(5, 1.5, n_samples)  # Log-normal for realistic length distribution
    content_complexity = np.random.beta(2, 5, n_samples)  # Beta distribution for complexity scores
    
    # Temporal patterns
    hour_of_day = dates.hour
    day_of_week = dates.dayofweek
    month = dates.month
    
    # Create base ground truth with realistic patterns
    ground_truth_prob = np.zeros(n_samples)
    
    # Category-based base probabilities
    category_violation_rates = {
        'spam': 0.85, 'harassment': 0.78, 'hate_speech': 0.82, 'violence': 0.88,
        'adult_content': 0.75, 'misinformation': 0.70, 'copyright': 0.65,
        'self_harm': 0.90, 'drugs': 0.80, 'legitimate': 0.05
    }
    
    for i, category in enumerate(categories):
        ground_truth_prob[i] = category_violation_rates[category]
    
    # Adjust for content type (images harder to moderate accurately)
    content_type_adjustments = {'text': 0.0, 'image': 0.1, 'video': 0.15}
    for i, content_type in enumerate(content_types):
        if categories[i] != 'legitimate':  # Only adjust violation content
            ground_truth_prob[i] = min(0.95, ground_truth_prob[i] + content_type_adjustments[content_type])
    
    # Adjust for complexity (more complex content harder to moderate)
    complexity_adjustment = (content_complexity - 0.5) * 0.2
    ground_truth_prob = np.clip(ground_truth_prob + complexity_adjustment, 0.01, 0.99)
    
    # Adjust for temporal patterns (night shifts have higher error rates)
    night_shift_mask = (hour_of_day >= 22) | (hour_of_day <= 6)
    ground_truth_prob[night_shift_mask] = np.clip(ground_truth_prob[night_shift_mask] + 0.1, 0.01, 0.99)
    
    # Generate ground truth labels
    ground_truth = np.random.binomial(1, ground_truth_prob, n_samples)
    
    # Generate AI system predictions with realistic error patterns
    system_confidence_base = np.random.beta(3, 2, n_samples)  # Base confidence distribution
    
    # System performance varies by content type
    system_accuracy_by_type = {'text': 0.88, 'image': 0.82, 'video': 0.79}
    
    system_predictions = np.zeros(n_samples)
    system_confidence = np.zeros(n_samples)
    
    for i in range(n_samples):
        content_type = content_types[i]
        true_label = ground_truth[i]
        base_accuracy = system_accuracy_by_type[content_type]
        
        # Adjust accuracy based on various factors
        accuracy = base_accuracy
        
        # Complexity reduces accuracy
        accuracy -= content_complexity[i] * 0.15
        
        # Night shift reduces accuracy
        if night_shift_mask[i]:
            accuracy -= 0.08
        
        # Weekend reduces accuracy (less experienced moderators)
        if day_of_week[i] >= 5:
            accuracy -= 0.05
        
        # User context affects accuracy
        if user_previous_violations[i] > 5:
            accuracy += 0.03  # System is more suspicious of repeat offenders
        
        # Ensure accuracy stays in reasonable bounds
        accuracy = np.clip(accuracy, 0.6, 0.95)
        
        # Generate prediction based on adjusted accuracy
        if np.random.random() < accuracy:
            system_predictions[i] = true_label  # Correct prediction
            system_confidence[i] = system_confidence_base[i] * 0.8 + 0.2  # Higher confidence when correct
        else:
            system_predictions[i] = 1 - true_label  # Incorrect prediction
            system_confidence[i] = system_confidence_base[i] * 0.6 + 0.1  # Lower confidence when wrong
        
        # Add some noise to confidence
        system_confidence[i] += np.random.normal(0, 0.05)
        system_confidence[i] = np.clip(system_confidence[i], 0.1, 0.95)
    
    # Create comprehensive dataset
    dataset = pd.DataFrame({
        'timestamp': dates,
        'content_type': content_types,
        'category': categories,
        'content_length': content_length.astype(int),
        'content_complexity': content_complexity,
        'user_account_age_days': user_account_age_days.astype(int),
        'user_previous_violations': user_previous_violations,
        'user_region': user_regions,
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'month': month,
        'ground_truth': ground_truth.astype(int),
        'system_prediction': system_predictions.astype(int),
        'system_confidence': system_confidence,
        'is_weekend': (day_of_week >= 5).astype(int),
        'is_night_shift': night_shift_mask.astype(int)
    })
    
    # Add derived features
    dataset['is_correct'] = (dataset['ground_truth'] == dataset['system_prediction']).astype(int)
    dataset['is_false_positive'] = ((dataset['ground_truth'] == 0) & (dataset['system_prediction'] == 1)).astype(int)
    dataset['is_false_negative'] = ((dataset['ground_truth'] == 1) & (dataset['system_prediction'] == 0)).astype(int)
    dataset['error_type'] = 'correct'
    dataset.loc[dataset['is_false_positive'] == 1, 'error_type'] = 'false_positive'
    dataset.loc[dataset['is_false_negative'] == 1, 'error_type'] = 'false_negative'
    
    # Add confidence categories
    dataset['confidence_category'] = pd.cut(dataset['system_confidence'], 
                                          bins=[0, 0.3, 0.7, 1.0], 
                                          labels=['low', 'medium', 'high'])
    
    print(f"✓ Generated dataset with {len(dataset)} samples")
    print(f"  Content types: {dataset['content_type'].value_counts().to_dict()}")
    print(f"  Overall accuracy: {dataset['is_correct'].mean():.3f}")
    print(f"  False positive rate: {dataset['is_false_positive'].mean():.3f}")
    print(f"  False negative rate: {dataset['is_false_negative'].mean():.3f}")
    
    return dataset

# Generate the evaluation dataset
evaluation_data = generate_comprehensive_evaluation_dataset()

# Display basic dataset information
print("\\nDataset Overview:")
print("=" * 40)
print(f"Shape: {evaluation_data.shape}")
print(f"Date range: {evaluation_data['timestamp'].min()} to {evaluation_data['timestamp'].max()}")
print("\\nFirst 5 rows:")
print(evaluation_data.head())

print("\\nDataset Summary Statistics:")
print(evaluation_data.describe())
```

### Step 3: Comprehensive Exploratory Data Analysis

```python
def conduct_comprehensive_eda(data):
    """
    Conduct comprehensive exploratory data analysis with advanced visualizations.
    """
    
    print("\\n" + "=" * 60)
    print("COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Error Rate by Content Type
    plt.subplot(4, 3, 1)
    error_by_type = data.groupby('content_type')['is_correct'].agg(['mean', 'count']).reset_index()
    error_by_type['error_rate'] = 1 - error_by_type['mean']
    
    bars = plt.bar(error_by_type['content_type'], error_by_type['error_rate'])
    plt.title('Error Rate by Content Type', fontsize=14, fontweight='bold')
    plt.ylabel('Error Rate')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, error_by_type['error_rate']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # 2. Error Distribution by Category
    plt.subplot(4, 3, 2)
    category_errors = data.groupby('category')['is_correct'].agg(['mean', 'count']).reset_index()
    category_errors['error_rate'] = 1 - category_errors['mean']
    category_errors = category_errors.sort_values('error_rate', ascending=True)
    
    plt.barh(category_errors['category'], category_errors['error_rate'])
    plt.title('Error Rate by Category', fontsize=14, fontweight='bold')
    plt.xlabel('Error Rate')
    
    # 3. Temporal Error Patterns - Hourly
    plt.subplot(4, 3, 3)
    hourly_errors = data.groupby('hour_of_day')['is_correct'].agg(['mean', 'count']).reset_index()
    hourly_errors['error_rate'] = 1 - hourly_errors['mean']
    
    plt.plot(hourly_errors['hour_of_day'], hourly_errors['error_rate'], marker='o', linewidth=2)
    plt.title('Error Rate by Hour of Day', fontsize=14, fontweight='bold')
    plt.xlabel('Hour of Day')
    plt.ylabel('Error Rate')
    plt.grid(True, alpha=0.3)
    
    # 4. Confidence vs Accuracy Relationship
    plt.subplot(4, 3, 4)
    confidence_bins = pd.cut(data['system_confidence'], bins=10)
    conf_accuracy = data.groupby(confidence_bins)['is_correct'].mean()
    
    bin_centers = [interval.mid for interval in conf_accuracy.index]
    plt.scatter(bin_centers, conf_accuracy.values, s=100, alpha=0.7)
    plt.plot(bin_centers, conf_accuracy.values, '--', alpha=0.5)
    plt.title('Accuracy vs System Confidence', fontsize=14, fontweight='bold')
    plt.xlabel('System Confidence')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    # 5. Error Type Distribution
    plt.subplot(4, 3, 5)
    error_type_counts = data['error_type'].value_counts()
    colors = ['lightgreen', 'lightcoral', 'lightsalmon']
    
    wedges, texts, autotexts = plt.pie(error_type_counts.values, labels=error_type_counts.index, 
                                      autopct='%1.1f%%', colors=colors)
    plt.title('Distribution of Error Types', fontsize=14, fontweight='bold')
    
    # 6. Content Complexity vs Error Rate
    plt.subplot(4, 3, 6)
    complexity_bins = pd.cut(data['content_complexity'], bins=8)
    complexity_errors = data.groupby(complexity_bins)['is_correct'].agg(['mean', 'count']).reset_index()
    complexity_errors['error_rate'] = 1 - complexity_errors['mean']
    
    bin_centers = [interval.mid for interval in complexity_errors['content_complexity']]
    plt.scatter(bin_centers, complexity_errors['error_rate'], 
               s=complexity_errors['count']*2, alpha=0.6)
    plt.title('Error Rate vs Content Complexity', fontsize=14, fontweight='bold')
    plt.xlabel('Content Complexity')
    plt.ylabel('Error Rate')
    plt.grid(True, alpha=0.3)
    
    # 7. User Context Impact
    plt.subplot(4, 3, 7)
    violation_bins = pd.cut(data['user_previous_violations'], bins=[0, 1, 3, 5, 20], 
                           labels=['0-1', '2-3', '4-5', '6+'])
    violation_errors = data.groupby(violation_bins)['is_correct'].agg(['mean', 'count']).reset_index()
    violation_errors['error_rate'] = 1 - violation_errors['mean']
    
    plt.bar(violation_errors['user_previous_violations'].astype(str), violation_errors['error_rate'])
    plt.title('Error Rate by User Violation History', fontsize=14, fontweight='bold')
    plt.xlabel('Previous Violations')
    plt.ylabel('Error Rate')
    plt.xticks(rotation=45)
    
    # 8. Regional Error Patterns
    plt.subplot(4, 3, 8)
    regional_errors = data.groupby('user_region')['is_correct'].agg(['mean', 'count']).reset_index()
    regional_errors['error_rate'] = 1 - regional_errors['mean']
    
    plt.bar(regional_errors['user_region'], regional_errors['error_rate'])
    plt.title('Error Rate by User Region', fontsize=14, fontweight='bold')
    plt.xlabel('Region')
    plt.ylabel('Error Rate')
    
    # 9. Weekend vs Weekday Performance
    plt.subplot(4, 3, 9)
    weekend_comparison = data.groupby('is_weekend')['is_correct'].agg(['mean', 'count']).reset_index()
    weekend_comparison['error_rate'] = 1 - weekend_comparison['mean']
    weekend_labels = ['Weekday', 'Weekend']
    
    bars = plt.bar(weekend_labels, weekend_comparison['error_rate'])
    plt.title('Error Rate: Weekday vs Weekend', fontsize=14, fontweight='bold')
    plt.ylabel('Error Rate')
    
    for bar, value in zip(bars, weekend_comparison['error_rate']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # 10. Monthly Trend Analysis
    plt.subplot(4, 3, 10)
    monthly_errors = data.groupby('month')['is_correct'].agg(['mean', 'count']).reset_index()
    monthly_errors['error_rate'] = 1 - monthly_errors['mean']
    
    plt.plot(monthly_errors['month'], monthly_errors['error_rate'], marker='o', linewidth=2)
    plt.title('Error Rate Trend by Month', fontsize=14, fontweight='bold')
    plt.xlabel('Month')
    plt.ylabel('Error Rate')
    plt.xticks(range(1, 13))
    plt.grid(True, alpha=0.3)
    
    # 11. Content Length Impact
    plt.subplot(4, 3, 11)
    length_bins = pd.qcut(data['content_length'], q=6, precision=0)
    length_errors = data.groupby(length_bins)['is_correct'].agg(['mean', 'count']).reset_index()
    length_errors['error_rate'] = 1 - length_errors['mean']
    
    bin_labels = [f"{int(interval.left)}-{int(interval.right)}" for interval in length_errors['content_length']]
    plt.bar(range(len(bin_labels)), length_errors['error_rate'])
    plt.title('Error Rate by Content Length', fontsize=14, fontweight='bold')
    plt.xlabel('Content Length Range')
    plt.ylabel('Error Rate')
    plt.xticks(range(len(bin_labels)), bin_labels, rotation=45)
    
    # 12. Confidence Distribution by Error Type
    plt.subplot(4, 3, 12)
    for error_type in data['error_type'].unique():
        subset = data[data['error_type'] == error_type]
        plt.hist(subset['system_confidence'], alpha=0.6, label=error_type, bins=20)
    
    plt.title('Confidence Distribution by Error Type', fontsize=14, fontweight='bold')
    plt.xlabel('System Confidence')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Generate summary statistics
    eda_summary = {
        'overall_metrics': {
            'total_samples': len(data),
            'overall_accuracy': data['is_correct'].mean(),
            'false_positive_rate': data['is_false_positive'].mean(),
            'false_negative_rate': data['is_false_negative'].mean(),
            'average_confidence': data['system_confidence'].mean()
        },
        'content_type_analysis': error_by_type.to_dict('records'),
        'category_analysis': category_errors.to_dict('records'),
        'temporal_patterns': {
            'worst_hour': hourly_errors.loc[hourly_errors['error_rate'].idxmax(), 'hour_of_day'],
            'best_hour': hourly_errors.loc[hourly_errors['error_rate'].idxmin(), 'hour_of_day'],
            'weekend_effect': weekend_comparison['error_rate'].iloc[1] - weekend_comparison['error_rate'].iloc[0]
        },
        'user_context_impact': {
            'regional_variance': regional_errors['error_rate'].std(),
            'violation_history_correlation': data['user_previous_violations'].corr(1 - data['is_correct'])
        }
    }
    
    return eda_summary

# Conduct comprehensive EDA
eda_results = conduct_comprehensive_eda(evaluation_data)

print("\\nEDA Summary:")
print("=" * 30)
print(f"Overall Accuracy: {eda_results['overall_metrics']['overall_accuracy']:.3f}")
print(f"False Positive Rate: {eda_results['overall_metrics']['false_positive_rate']:.3f}")
print(f"False Negative Rate: {eda_results['overall_metrics']['false_negative_rate']:.3f}")
print(f"Worst Performance Hour: {eda_results['temporal_patterns']['worst_hour']}:00")
print(f"Weekend Effect: {eda_results['temporal_patterns']['weekend_effect']:.3f} increase in error rate")
```

## Part 2: Advanced Statistical Pattern Analysis

### Step 4: Correlation Analysis and Feature Relationships

```python
def conduct_advanced_correlation_analysis(data):
    """
    Conduct comprehensive correlation analysis to identify feature relationships.
    """
    
    print("\\n" + "=" * 60)
    print("ADVANCED CORRELATION AND RELATIONSHIP ANALYSIS")
    print("=" * 60)
    
    # Prepare numerical features for correlation analysis
    numerical_features = [
        'content_length', 'content_complexity', 'user_account_age_days',
        'user_previous_violations', 'hour_of_day', 'day_of_week', 'month',
        'system_confidence', 'is_correct', 'is_false_positive', 'is_false_negative'
    ]
    
    correlation_data = data[numerical_features].copy()
    
    # Add encoded categorical features
    le_content_type = LabelEncoder()
    le_category = LabelEncoder()
    le_region = LabelEncoder()
    
    correlation_data['content_type_encoded'] = le_content_type.fit_transform(data['content_type'])
    correlation_data['category_encoded'] = le_category.fit_transform(data['category'])
    correlation_data['region_encoded'] = le_region.fit_transform(data['user_region'])
    
    # Calculate correlation matrix
    correlation_matrix = correlation_data.corr()
    
    # Create comprehensive correlation visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Full correlation heatmap
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=axes[0,0])
    axes[0,0].set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    
    # 2. Error-focused correlations
    error_correlations = correlation_matrix[['is_correct', 'is_false_positive', 'is_false_negative']].drop(
        ['is_correct', 'is_false_positive', 'is_false_negative']
    ).sort_values('is_correct', key=abs, ascending=False)
    
    sns.heatmap(error_correlations.T, annot=True, cmap='RdBu_r', center=0,
                cbar_kws={"shrink": .8}, ax=axes[0,1])
    axes[0,1].set_title('Error Pattern Correlations', fontsize=14, fontweight='bold')
    
    # 3. Confidence vs Performance Analysis
    confidence_bins = pd.cut(data['system_confidence'], bins=10)
    conf_performance = data.groupby(confidence_bins).agg({
        'is_correct': 'mean',
        'is_false_positive': 'mean',
        'is_false_negative': 'mean'
    }).reset_index()
    
    bin_centers = [interval.mid for interval in conf_performance['system_confidence']]
    
    axes[1,0].plot(bin_centers, conf_performance['is_correct'], 'o-', label='Accuracy', linewidth=2)
    axes[1,0].plot(bin_centers, conf_performance['is_false_positive'], 's-', label='False Positive Rate', linewidth=2)
    axes[1,0].plot(bin_centers, conf_performance['is_false_negative'], '^-', label='False Negative Rate', linewidth=2)
    axes[1,0].set_xlabel('System Confidence')
    axes[1,0].set_ylabel('Rate')
    axes[1,0].set_title('Performance Metrics vs Confidence', fontsize=14, fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Feature Importance for Error Prediction
    # Prepare features for modeling
    feature_columns = [col for col in correlation_data.columns if col not in ['is_correct', 'is_false_positive', 'is_false_negative']]
    X = correlation_data[feature_columns]
    y = 1 - correlation_data['is_correct']  # Error indicator
    
    # Train random forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=True)
    
    axes[1,1].barh(feature_importance['feature'], feature_importance['importance'])
    axes[1,1].set_title('Feature Importance for Error Prediction', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Importance Score')
    
    plt.tight_layout()
    plt.show()
    
    # Statistical significance testing
    print("\\nStatistical Significance Tests:")
    print("=" * 40)
    
    # Test 1: Content type vs error rate
    content_type_groups = [group['is_correct'].values for name, group in data.groupby('content_type')]
    f_stat, p_value = stats.f_oneway(*content_type_groups)
    print(f"Content Type vs Error Rate (ANOVA): F={f_stat:.3f}, p={p_value:.6f}")
    
    # Test 2: Weekend vs weekday performance
    weekday_accuracy = data[data['is_weekend'] == 0]['is_correct']
    weekend_accuracy = data[data['is_weekend'] == 1]['is_correct']
    t_stat, p_value = stats.ttest_ind(weekday_accuracy, weekend_accuracy)
    print(f"Weekend vs Weekday (t-test): t={t_stat:.3f}, p={p_value:.6f}")
    
    # Test 3: Confidence vs accuracy correlation
    conf_acc_corr, p_value = pearsonr(data['system_confidence'], data['is_correct'])
    print(f"Confidence vs Accuracy (Pearson): r={conf_acc_corr:.3f}, p={p_value:.6f}")
    
    # Test 4: Complexity vs error rate
    complexity_error_corr, p_value = spearmanr(data['content_complexity'], 1 - data['is_correct'])
    print(f"Complexity vs Error Rate (Spearman): ρ={complexity_error_corr:.3f}, p={p_value:.6f}")
    
    # Test 5: Chi-square test for categorical associations
    contingency_table = pd.crosstab(data['content_type'], data['error_type'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f"Content Type vs Error Type (Chi-square): χ²={chi2:.3f}, p={p_value:.6f}")
    
    # Advanced pattern detection
    pattern_analysis = {
        'correlation_insights': {
            'strongest_error_predictors': feature_importance.tail(5).to_dict('records'),
            'confidence_accuracy_correlation': conf_acc_corr,
            'complexity_impact': complexity_error_corr
        },
        'statistical_tests': {
            'content_type_significance': p_value < 0.05,
            'weekend_effect_significance': p_value < 0.05,
            'confidence_correlation_significance': p_value < 0.05
        },
        'performance_patterns': {
            'optimal_confidence_range': [
                bin_centers[np.argmax(conf_performance['is_correct']) - 1],
                bin_centers[np.argmax(conf_performance['is_correct']) + 1]
            ] if np.argmax(conf_performance['is_correct']) > 0 else [0.8, 1.0],
            'high_risk_combinations': []
        }
    }
    
    return pattern_analysis, correlation_matrix, feature_importance

# Conduct correlation analysis
correlation_results, correlation_matrix, feature_importance = conduct_advanced_correlation_analysis(evaluation_data)

print("\\nKey Correlation Insights:")
print("=" * 30)
print("Top 3 Error Predictors:")
for predictor in correlation_results['correlation_insights']['strongest_error_predictors'][-3:]:
    print(f"  {predictor['feature']}: {predictor['importance']:.3f}")

print(f"\\nConfidence-Accuracy Correlation: {correlation_results['correlation_insights']['confidence_accuracy_correlation']:.3f}")
print(f"Complexity Impact on Errors: {correlation_results['correlation_insights']['complexity_impact']:.3f}")
```

### Step 5: Time Series Analysis and Temporal Patterns

```python
def conduct_temporal_pattern_analysis(data):
    """
    Analyze temporal patterns and trends in error rates over time.
    """
    
    print("\\n" + "=" * 60)
    print("TEMPORAL PATTERN AND TREND ANALYSIS")
    print("=" * 60)
    
    # Prepare time series data
    data_ts = data.copy()
    data_ts['date'] = data_ts['timestamp'].dt.date
    data_ts['week'] = data_ts['timestamp'].dt.isocalendar().week
    data_ts['hour_minute'] = data_ts['timestamp'].dt.hour + data_ts['timestamp'].dt.minute / 60
    
    # Daily aggregations
    daily_metrics = data_ts.groupby('date').agg({
        'is_correct': ['mean', 'count'],
        'is_false_positive': 'mean',
        'is_false_negative': 'mean',
        'system_confidence': 'mean'
    }).reset_index()
    
    daily_metrics.columns = ['date', 'accuracy', 'sample_count', 'fp_rate', 'fn_rate', 'avg_confidence']
    daily_metrics['error_rate'] = 1 - daily_metrics['accuracy']
    
    # Weekly aggregations
    weekly_metrics = data_ts.groupby('week').agg({
        'is_correct': ['mean', 'count'],
        'is_false_positive': 'mean',
        'is_false_negative': 'mean',
        'system_confidence': 'mean'
    }).reset_index()
    
    weekly_metrics.columns = ['week', 'accuracy', 'sample_count', 'fp_rate', 'fn_rate', 'avg_confidence']
    weekly_metrics['error_rate'] = 1 - weekly_metrics['accuracy']
    
    # Hourly patterns across all days
    hourly_patterns = data_ts.groupby('hour_of_day').agg({
        'is_correct': ['mean', 'std', 'count'],
        'system_confidence': ['mean', 'std']
    }).reset_index()
    
    hourly_patterns.columns = ['hour', 'accuracy_mean', 'accuracy_std', 'count', 'confidence_mean', 'confidence_std']
    hourly_patterns['error_rate'] = 1 - hourly_patterns['accuracy_mean']
    
    # Create comprehensive temporal visualizations
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Daily error rate trend
    plt.subplot(3, 3, 1)
    plt.plot(daily_metrics['date'], daily_metrics['error_rate'], 'o-', alpha=0.7)
    
    # Add trend line
    x_numeric = np.arange(len(daily_metrics))
    z = np.polyfit(x_numeric, daily_metrics['error_rate'], 1)
    p = np.poly1d(z)
    plt.plot(daily_metrics['date'], p(x_numeric), "--", alpha=0.8, color='red')
    
    plt.title('Daily Error Rate Trend', fontsize=14, fontweight='bold')
    plt.ylabel('Error Rate')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 2. Weekly error rate pattern
    plt.subplot(3, 3, 2)
    plt.bar(weekly_metrics['week'], weekly_metrics['error_rate'])
    plt.title('Weekly Error Rate Pattern', fontsize=14, fontweight='bold')
    plt.xlabel('Week Number')
    plt.ylabel('Error Rate')
    
    # 3. Hourly error rate with confidence intervals
    plt.subplot(3, 3, 3)
    plt.errorbar(hourly_patterns['hour'], hourly_patterns['error_rate'], 
                yerr=hourly_patterns['accuracy_std'], capsize=5, capthick=2)
    plt.title('Hourly Error Rate with Confidence Intervals', fontsize=14, fontweight='bold')
    plt.xlabel('Hour of Day')
    plt.ylabel('Error Rate')
    plt.grid(True, alpha=0.3)
    
    # 4. Day of week analysis
    plt.subplot(3, 3, 4)
    dow_patterns = data_ts.groupby('day_of_week').agg({
        'is_correct': 'mean',
        'system_confidence': 'mean'
    }).reset_index()
    dow_patterns['error_rate'] = 1 - dow_patterns['is_correct']
    
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    plt.bar(day_names, dow_patterns['error_rate'])
    plt.title('Error Rate by Day of Week', fontsize=14, fontweight='bold')
    plt.ylabel('Error Rate')
    plt.xticks(rotation=45)
    
    # 5. Monthly trend analysis
    plt.subplot(3, 3, 5)
    monthly_patterns = data_ts.groupby('month').agg({
        'is_correct': 'mean',
        'system_confidence': 'mean'
    }).reset_index()
    monthly_patterns['error_rate'] = 1 - monthly_patterns['is_correct']
    
    plt.plot(monthly_patterns['month'], monthly_patterns['error_rate'], 'o-', linewidth=2)
    plt.title('Monthly Error Rate Trend', fontsize=14, fontweight='bold')
    plt.xlabel('Month')
    plt.ylabel('Error Rate')
    plt.xticks(range(1, 13))
    plt.grid(True, alpha=0.3)
    
    # 6. Error rate vs confidence over time
    plt.subplot(3, 3, 6)
    plt.scatter(daily_metrics['avg_confidence'], daily_metrics['error_rate'], 
               s=daily_metrics['sample_count']/10, alpha=0.6)
    plt.title('Daily Error Rate vs Average Confidence', fontsize=14, fontweight='bold')
    plt.xlabel('Average Confidence')
    plt.ylabel('Error Rate')
    plt.grid(True, alpha=0.3)
    
    # 7. Rolling average analysis
    plt.subplot(3, 3, 7)
    daily_metrics['error_rate_ma7'] = daily_metrics['error_rate'].rolling(window=7).mean()
    daily_metrics['error_rate_ma30'] = daily_metrics['error_rate'].rolling(window=30).mean()
    
    plt.plot(daily_metrics['date'], daily_metrics['error_rate'], alpha=0.3, label='Daily')
    plt.plot(daily_metrics['date'], daily_metrics['error_rate_ma7'], label='7-day MA')
    plt.plot(daily_metrics['date'], daily_metrics['error_rate_ma30'], label='30-day MA')
    plt.title('Error Rate with Moving Averages', fontsize=14, fontweight='bold')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.xticks(rotation=45)
    
    # 8. Seasonal decomposition simulation
    plt.subplot(3, 3, 8)
    # Create synthetic seasonal component
    seasonal_component = 0.02 * np.sin(2 * np.pi * np.arange(len(daily_metrics)) / 30)
    trend_component = np.linspace(0, 0.01, len(daily_metrics))
    
    plt.plot(daily_metrics['date'], seasonal_component, label='Seasonal')
    plt.plot(daily_metrics['date'], trend_component, label='Trend')
    plt.title('Decomposed Temporal Components', fontsize=14, fontweight='bold')
    plt.ylabel('Component Value')
    plt.legend()
    plt.xticks(rotation=45)
    
    # 9. Error burst detection
    plt.subplot(3, 3, 9)
    # Identify days with unusually high error rates
    error_threshold = daily_metrics['error_rate'].mean() + 2 * daily_metrics['error_rate'].std()
    error_bursts = daily_metrics[daily_metrics['error_rate'] > error_threshold]
    
    plt.scatter(daily_metrics['date'], daily_metrics['error_rate'], alpha=0.6, label='Normal')
    plt.scatter(error_bursts['date'], error_bursts['error_rate'], 
               color='red', s=100, label='Error Bursts')
    plt.axhline(y=error_threshold, color='red', linestyle='--', alpha=0.7, label='Threshold')
    plt.title('Error Burst Detection', fontsize=14, fontweight='bold')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Statistical analysis of temporal patterns
    print("\\nTemporal Pattern Analysis:")
    print("=" * 40)
    
    # Trend analysis
    x_numeric = np.arange(len(daily_metrics))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, daily_metrics['error_rate'])
    print(f"Daily Trend: slope={slope:.6f}, R²={r_value**2:.3f}, p={p_value:.6f}")
    
    # Cyclical patterns
    hourly_variance = hourly_patterns['error_rate'].var()
    dow_variance = dow_patterns['error_rate'].var()
    monthly_variance = monthly_patterns['error_rate'].var()
    
    print(f"Hourly Pattern Variance: {hourly_variance:.6f}")
    print(f"Day-of-Week Pattern Variance: {dow_variance:.6f}")
    print(f"Monthly Pattern Variance: {monthly_variance:.6f}")
    
    # Peak and trough identification
    worst_hour = hourly_patterns.loc[hourly_patterns['error_rate'].idxmax(), 'hour']
    best_hour = hourly_patterns.loc[hourly_patterns['error_rate'].idxmin(), 'hour']
    worst_day = day_names[dow_patterns['error_rate'].idxmax()]
    best_day = day_names[dow_patterns['error_rate'].idxmin()]
    
    print(f"\\nPeak Performance Times:")
    print(f"  Best Hour: {best_hour}:00 (Error Rate: {hourly_patterns['error_rate'].min():.3f})")
    print(f"  Worst Hour: {worst_hour}:00 (Error Rate: {hourly_patterns['error_rate'].max():.3f})")
    print(f"  Best Day: {best_day} (Error Rate: {dow_patterns['error_rate'].min():.3f})")
    print(f"  Worst Day: {worst_day} (Error Rate: {dow_patterns['error_rate'].max():.3f})")
    
    # Error burst analysis
    print(f"\\nError Burst Analysis:")
    print(f"  Error Burst Threshold: {error_threshold:.3f}")
    print(f"  Number of Error Burst Days: {len(error_bursts)}")
    print(f"  Percentage of Days with Error Bursts: {len(error_bursts)/len(daily_metrics)*100:.1f}%")
    
    temporal_analysis = {
        'trend_analysis': {
            'daily_slope': slope,
            'trend_significance': p_value < 0.05,
            'r_squared': r_value**2
        },
        'cyclical_patterns': {
            'hourly_variance': hourly_variance,
            'dow_variance': dow_variance,
            'monthly_variance': monthly_variance,
            'strongest_cycle': 'hourly' if hourly_variance > max(dow_variance, monthly_variance) else 'weekly' if dow_variance > monthly_variance else 'monthly'
        },
        'performance_extremes': {
            'best_hour': int(best_hour),
            'worst_hour': int(worst_hour),
            'best_day': best_day,
            'worst_day': worst_day
        },
        'error_bursts': {
            'threshold': error_threshold,
            'burst_count': len(error_bursts),
            'burst_percentage': len(error_bursts)/len(daily_metrics)*100
        }
    }
    
    return temporal_analysis, daily_metrics, hourly_patterns

# Conduct temporal analysis
temporal_results, daily_data, hourly_data = conduct_temporal_pattern_analysis(evaluation_data)

print("\\nKey Temporal Insights:")
print("=" * 30)
print(f"Strongest Cyclical Pattern: {temporal_results['cyclical_patterns']['strongest_cycle']}")
print(f"Daily Trend Significance: {temporal_results['trend_analysis']['trend_significance']}")
print(f"Performance Range: {hourly_data['error_rate'].min():.3f} - {hourly_data['error_rate'].max():.3f}")
```

## Part 3: Advanced Statistical Modeling and Prediction

### Step 6: Predictive Modeling for Error Prevention

```python
def build_error_prediction_models(data):
    """
    Build and evaluate predictive models for error prevention.
    """
    
    print("\\n" + "=" * 60)
    print("PREDICTIVE MODELING FOR ERROR PREVENTION")
    print("=" * 60)
    
    # Prepare features for modeling
    feature_engineering_data = data.copy()
    
    # Create additional engineered features
    feature_engineering_data['content_length_log'] = np.log1p(feature_engineering_data['content_length'])
    feature_engineering_data['user_experience_score'] = (
        feature_engineering_data['user_account_age_days'] / 
        (feature_engineering_data['user_previous_violations'] + 1)
    )
    feature_engineering_data['complexity_confidence_interaction'] = (
        feature_engineering_data['content_complexity'] * feature_engineering_data['system_confidence']
    )
    feature_engineering_data['temporal_risk_score'] = (
        feature_engineering_data['is_night_shift'] * 0.3 + 
        feature_engineering_data['is_weekend'] * 0.2
    )
    
    # Encode categorical variables
    categorical_features = ['content_type', 'category', 'user_region', 'confidence_category']
    encoded_features = pd.get_dummies(feature_engineering_data[categorical_features], prefix=categorical_features)
    
    # Select numerical features
    numerical_features = [
        'content_length', 'content_length_log', 'content_complexity',
        'user_account_age_days', 'user_previous_violations', 'user_experience_score',
        'hour_of_day', 'day_of_week', 'month', 'system_confidence',
        'complexity_confidence_interaction', 'temporal_risk_score',
        'is_weekend', 'is_night_shift'
    ]
    
    # Combine features
    X = pd.concat([
        feature_engineering_data[numerical_features],
        encoded_features
    ], axis=1)
    
    # Define prediction targets
    y_error = 1 - feature_engineering_data['is_correct']  # General error prediction
    y_fp = feature_engineering_data['is_false_positive']  # False positive prediction
    y_fn = feature_engineering_data['is_false_negative']  # False negative prediction
    
    # Split data
    X_train, X_test, y_error_train, y_error_test = train_test_split(
        X, y_error, test_size=0.2, random_state=42, stratify=y_error
    )
    
    _, _, y_fp_train, y_fp_test = train_test_split(
        X, y_fp, test_size=0.2, random_state=42, stratify=y_fp
    )
    
    _, _, y_fn_train, y_fn_test = train_test_split(
        X, y_fn, test_size=0.2, random_state=42, stratify=y_fn
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model 1: Random Forest for General Error Prediction
    print("\\nTraining Random Forest for General Error Prediction...")
    rf_error = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_error.fit(X_train, y_error_train)
    
    rf_error_pred = rf_error.predict(X_test)
    rf_error_prob = rf_error.predict_proba(X_test)[:, 1]
    rf_error_auc = roc_auc_score(y_error_test, rf_error_prob)
    
    print(f"Random Forest Error Prediction AUC: {rf_error_auc:.3f}")
    
    # Model 2: Logistic Regression for False Positive Prediction
    print("Training Logistic Regression for False Positive Prediction...")
    lr_fp = LogisticRegression(random_state=42, max_iter=1000)
    lr_fp.fit(X_train_scaled, y_fp_train)
    
    lr_fp_pred = lr_fp.predict(X_test_scaled)
    lr_fp_prob = lr_fp.predict_proba(X_test_scaled)[:, 1]
    lr_fp_auc = roc_auc_score(y_fp_test, lr_fp_prob)
    
    print(f"Logistic Regression False Positive Prediction AUC: {lr_fp_auc:.3f}")
    
    # Model 3: Random Forest for False Negative Prediction
    print("Training Random Forest for False Negative Prediction...")
    rf_fn = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_fn.fit(X_train, y_fn_train)
    
    rf_fn_pred = rf_fn.predict(X_test)
    rf_fn_prob = rf_fn.predict_proba(X_test)[:, 1]
    rf_fn_auc = roc_auc_score(y_fn_test, rf_fn_prob)
    
    print(f"Random Forest False Negative Prediction AUC: {rf_fn_auc:.3f}")
    
    # Feature importance analysis
    feature_importance_error = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_error.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importance_fp = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(lr_fp.coef_[0])
    }).sort_values('importance', ascending=False)
    
    feature_importance_fn = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_fn.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Create model evaluation visualizations
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Feature importance for error prediction
    top_features_error = feature_importance_error.head(10)
    axes[0,0].barh(top_features_error['feature'], top_features_error['importance'])
    axes[0,0].set_title('Top Features for Error Prediction', fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('Importance')
    
    # 2. Feature importance for false positive prediction
    top_features_fp = feature_importance_fp.head(10)
    axes[0,1].barh(top_features_fp['feature'], top_features_fp['importance'])
    axes[0,1].set_title('Top Features for False Positive Prediction', fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('Importance')
    
    # 3. Feature importance for false negative prediction
    top_features_fn = feature_importance_fn.head(10)
    axes[0,2].barh(top_features_fn['feature'], top_features_fn['importance'])
    axes[0,2].set_title('Top Features for False Negative Prediction', fontsize=12, fontweight='bold')
    axes[0,2].set_xlabel('Importance')
    
    # 4. Prediction probability distributions
    axes[1,0].hist(rf_error_prob[y_error_test == 0], alpha=0.7, label='No Error', bins=20)
    axes[1,0].hist(rf_error_prob[y_error_test == 1], alpha=0.7, label='Error', bins=20)
    axes[1,0].set_title('Error Prediction Probability Distribution', fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('Predicted Probability')
    axes[1,0].legend()
    
    # 5. False positive prediction probabilities
    axes[1,1].hist(lr_fp_prob[y_fp_test == 0], alpha=0.7, label='No FP', bins=20)
    axes[1,1].hist(lr_fp_prob[y_fp_test == 1], alpha=0.7, label='FP', bins=20)
    axes[1,1].set_title('False Positive Prediction Probability Distribution', fontsize=12, fontweight='bold')
    axes[1,1].set_xlabel('Predicted Probability')
    axes[1,1].legend()
    
    # 6. False negative prediction probabilities
    axes[1,2].hist(rf_fn_prob[y_fn_test == 0], alpha=0.7, label='No FN', bins=20)
    axes[1,2].hist(rf_fn_prob[y_fn_test == 1], alpha=0.7, label='FN', bins=20)
    axes[1,2].set_title('False Negative Prediction Probability Distribution', fontsize=12, fontweight='bold')
    axes[1,2].set_xlabel('Predicted Probability')
    axes[1,2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Model performance evaluation
    print("\\nDetailed Model Performance:")
    print("=" * 40)
    
    print("\\nGeneral Error Prediction (Random Forest):")
    print(classification_report(y_error_test, rf_error_pred))
    
    print("\\nFalse Positive Prediction (Logistic Regression):")
    print(classification_report(y_fp_test, lr_fp_pred))
    
    print("\\nFalse Negative Prediction (Random Forest):")
    print(classification_report(y_fn_test, rf_fn_pred))
    
    # Risk scoring system
    def calculate_risk_scores(X_sample, models, scaler):
        """Calculate comprehensive risk scores for new samples."""
        
        X_scaled = scaler.transform(X_sample)
        
        error_prob = models['error_model'].predict_proba(X_sample)[:, 1]
        fp_prob = models['fp_model'].predict_proba(X_scaled)[:, 1]
        fn_prob = models['fn_model'].predict_proba(X_sample)[:, 1]
        
        # Composite risk score
        composite_risk = (error_prob * 0.5 + fp_prob * 0.25 + fn_prob * 0.25)
        
        return {
            'error_probability': error_prob,
            'false_positive_probability': fp_prob,
            'false_negative_probability': fn_prob,
            'composite_risk_score': composite_risk
        }
    
    # Package models for deployment
    models = {
        'error_model': rf_error,
        'fp_model': lr_fp,
        'fn_model': rf_fn,
        'scaler': scaler,
        'feature_columns': X.columns.tolist()
    }
    
    # Test risk scoring on sample data
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    sample_data = X_test.iloc[sample_indices]
    
    risk_scores = calculate_risk_scores(sample_data, models, scaler)
    
    print("\\nSample Risk Scores:")
    print("=" * 30)
    for i, idx in enumerate(sample_indices):
        print(f"Sample {i+1}:")
        print(f"  Error Probability: {risk_scores['error_probability'][i]:.3f}")
        print(f"  False Positive Probability: {risk_scores['false_positive_probability'][i]:.3f}")
        print(f"  False Negative Probability: {risk_scores['false_negative_probability'][i]:.3f}")
        print(f"  Composite Risk Score: {risk_scores['composite_risk_score'][i]:.3f}")
        print()
    
    modeling_results = {
        'model_performance': {
            'error_prediction_auc': rf_error_auc,
            'fp_prediction_auc': lr_fp_auc,
            'fn_prediction_auc': rf_fn_auc
        },
        'feature_importance': {
            'error_prediction': feature_importance_error.head(5).to_dict('records'),
            'fp_prediction': feature_importance_fp.head(5).to_dict('records'),
            'fn_prediction': feature_importance_fn.head(5).to_dict('records')
        },
        'models': models,
        'risk_calculator': calculate_risk_scores
    }
    
    return modeling_results

# Build predictive models
modeling_results = build_error_prediction_models(evaluation_data)

print("\\nPredictive Modeling Summary:")
print("=" * 40)
print(f"Error Prediction AUC: {modeling_results['model_performance']['error_prediction_auc']:.3f}")
print(f"False Positive Prediction AUC: {modeling_results['model_performance']['fp_prediction_auc']:.3f}")
print(f"False Negative Prediction AUC: {modeling_results['model_performance']['fn_prediction_auc']:.3f}")

print("\\nTop Predictive Features:")
for model_type, features in modeling_results['feature_importance'].items():
    print(f"\\n{model_type.replace('_', ' ').title()}:")
    for feature in features[:3]:
        print(f"  {feature['feature']}: {feature['importance']:.3f}")
```

### Step 7: Anomaly Detection and Outlier Analysis

```python
def conduct_anomaly_detection_analysis(data):
    """
    Conduct comprehensive anomaly detection to identify unusual patterns.
    """
    
    print("\\n" + "=" * 60)
    print("ANOMALY DETECTION AND OUTLIER ANALYSIS")
    print("=" * 60)
    
    # Prepare data for anomaly detection
    anomaly_features = [
        'content_length', 'content_complexity', 'user_account_age_days',
        'user_previous_violations', 'system_confidence', 'hour_of_day'
    ]
    
    # Add encoded categorical features
    le_content_type = LabelEncoder()
    le_category = LabelEncoder()
    le_region = LabelEncoder()
    
    anomaly_data = data[anomaly_features].copy()
    anomaly_data['content_type_encoded'] = le_content_type.fit_transform(data['content_type'])
    anomaly_data['category_encoded'] = le_category.fit_transform(data['category'])
    anomaly_data['region_encoded'] = le_region.fit_transform(data['user_region'])
    
    # Scale features for anomaly detection
    scaler = StandardScaler()
    anomaly_data_scaled = scaler.fit_transform(anomaly_data)
    
    # Method 1: Isolation Forest
    print("\\nApplying Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_anomalies = iso_forest.fit_predict(anomaly_data_scaled)
    iso_scores = iso_forest.decision_function(anomaly_data_scaled)
    
    # Method 2: Statistical Outliers (Z-score based)
    print("Applying Statistical Outlier Detection...")
    z_scores = np.abs(stats.zscore(anomaly_data_scaled, axis=0))
    statistical_outliers = (z_scores > 3).any(axis=1)
    
    # Method 3: Clustering-based Anomalies
    print("Applying Clustering-based Anomaly Detection...")
    kmeans = KMeans(n_clusters=8, random_state=42)
    cluster_labels = kmeans.fit_predict(anomaly_data_scaled)
    
    # Calculate distance to nearest cluster center
    cluster_distances = []
    for i, point in enumerate(anomaly_data_scaled):
        cluster_center = kmeans.cluster_centers_[cluster_labels[i]]
        distance = np.linalg.norm(point - cluster_center)
        cluster_distances.append(distance)
    
    cluster_distances = np.array(cluster_distances)
    cluster_threshold = np.percentile(cluster_distances, 95)
    cluster_anomalies = cluster_distances > cluster_threshold
    
    # Combine anomaly detection results
    data_with_anomalies = data.copy()
    data_with_anomalies['iso_anomaly'] = (iso_anomalies == -1)
    data_with_anomalies['statistical_outlier'] = statistical_outliers
    data_with_anomalies['cluster_anomaly'] = cluster_anomalies
    data_with_anomalies['iso_score'] = iso_scores
    data_with_anomalies['cluster_distance'] = cluster_distances
    
    # Consensus anomalies (detected by multiple methods)
    data_with_anomalies['consensus_anomaly'] = (
        data_with_anomalies['iso_anomaly'] & 
        (data_with_anomalies['statistical_outlier'] | data_with_anomalies['cluster_anomaly'])
    )
    
    # Create comprehensive anomaly visualizations
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    
    # 1. Isolation Forest Scores Distribution
    axes[0,0].hist(iso_scores, bins=50, alpha=0.7)
    axes[0,0].axvline(x=np.percentile(iso_scores, 5), color='red', linestyle='--', label='5th Percentile')
    axes[0,0].set_title('Isolation Forest Anomaly Scores', fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('Anomaly Score')
    axes[0,0].legend()
    
    # 2. Statistical Outliers by Feature
    feature_outlier_counts = []
    for i, feature in enumerate(anomaly_features):
        outlier_count = (np.abs(stats.zscore(anomaly_data[feature])) > 3).sum()
        feature_outlier_counts.append(outlier_count)
    
    axes[0,1].bar(range(len(anomaly_features)), feature_outlier_counts)
    axes[0,1].set_title('Statistical Outliers by Feature', fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('Feature Index')
    axes[0,1].set_ylabel('Outlier Count')
    axes[0,1].set_xticks(range(len(anomaly_features)))
    axes[0,1].set_xticklabels([f.replace('_', '\\n') for f in anomaly_features], rotation=45)
    
    # 3. Cluster Distance Distribution
    axes[0,2].hist(cluster_distances, bins=50, alpha=0.7)
    axes[0,2].axvline(x=cluster_threshold, color='red', linestyle='--', label='95th Percentile')
    axes[0,2].set_title('Cluster Distance Distribution', fontsize=12, fontweight='bold')
    axes[0,2].set_xlabel('Distance to Cluster Center')
    axes[0,2].legend()
    
    # 4. Anomaly Detection Method Comparison
    method_counts = [
        data_with_anomalies['iso_anomaly'].sum(),
        data_with_anomalies['statistical_outlier'].sum(),
        data_with_anomalies['cluster_anomaly'].sum(),
        data_with_anomalies['consensus_anomaly'].sum()
    ]
    method_names = ['Isolation\\nForest', 'Statistical\\nOutliers', 'Cluster\\nBased', 'Consensus']
    
    axes[1,0].bar(method_names, method_counts)
    axes[1,0].set_title('Anomalies Detected by Method', fontsize=12, fontweight='bold')
    axes[1,0].set_ylabel('Number of Anomalies')
    
    # 5. Anomaly Error Rate Analysis
    anomaly_error_rates = []
    anomaly_types = ['iso_anomaly', 'statistical_outlier', 'cluster_anomaly', 'consensus_anomaly']
    
    for anomaly_type in anomaly_types:
        anomaly_subset = data_with_anomalies[data_with_anomalies[anomaly_type]]
        normal_subset = data_with_anomalies[~data_with_anomalies[anomaly_type]]
        
        anomaly_error_rate = 1 - anomaly_subset['is_correct'].mean() if len(anomaly_subset) > 0 else 0
        normal_error_rate = 1 - normal_subset['is_correct'].mean()
        
        anomaly_error_rates.append([anomaly_error_rate, normal_error_rate])
    
    x = np.arange(len(anomaly_types))
    width = 0.35
    
    anomaly_rates = [rates[0] for rates in anomaly_error_rates]
    normal_rates = [rates[1] for rates in anomaly_error_rates]
    
    axes[1,1].bar(x - width/2, anomaly_rates, width, label='Anomalies', alpha=0.8)
    axes[1,1].bar(x + width/2, normal_rates, width, label='Normal', alpha=0.8)
    axes[1,1].set_title('Error Rates: Anomalies vs Normal', fontsize=12, fontweight='bold')
    axes[1,1].set_ylabel('Error Rate')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels([name.replace('_', '\\n') for name in anomaly_types])
    axes[1,1].legend()
    
    # 6. Content Type Distribution in Anomalies
    consensus_anomalies = data_with_anomalies[data_with_anomalies['consensus_anomaly']]
    if len(consensus_anomalies) > 0:
        anomaly_content_dist = consensus_anomalies['content_type'].value_counts()
        normal_content_dist = data_with_anomalies[~data_with_anomalies['consensus_anomaly']]['content_type'].value_counts()
        
        content_types = list(set(anomaly_content_dist.index) | set(normal_content_dist.index))
        anomaly_props = [anomaly_content_dist.get(ct, 0) / len(consensus_anomalies) for ct in content_types]
        normal_props = [normal_content_dist.get(ct, 0) / len(data_with_anomalies[~data_with_anomalies['consensus_anomaly']]) for ct in content_types]
        
        x = np.arange(len(content_types))
        axes[1,2].bar(x - width/2, anomaly_props, width, label='Anomalies', alpha=0.8)
        axes[1,2].bar(x + width/2, normal_props, width, label='Normal', alpha=0.8)
        axes[1,2].set_title('Content Type Distribution', fontsize=12, fontweight='bold')
        axes[1,2].set_ylabel('Proportion')
        axes[1,2].set_xticks(x)
        axes[1,2].set_xticklabels(content_types)
        axes[1,2].legend()
    
    # 7. Temporal Distribution of Anomalies
    data_with_anomalies['hour_bin'] = pd.cut(data_with_anomalies['hour_of_day'], bins=6, labels=['0-4', '4-8', '8-12', '12-16', '16-20', '20-24'])
    temporal_anomalies = data_with_anomalies.groupby('hour_bin')['consensus_anomaly'].agg(['sum', 'count']).reset_index()
    temporal_anomalies['anomaly_rate'] = temporal_anomalies['sum'] / temporal_anomalies['count']
    
    axes[2,0].bar(temporal_anomalies['hour_bin'].astype(str), temporal_anomalies['anomaly_rate'])
    axes[2,0].set_title('Anomaly Rate by Time of Day', fontsize=12, fontweight='bold')
    axes[2,0].set_xlabel('Hour Range')
    axes[2,0].set_ylabel('Anomaly Rate')
    
    # 8. Confidence Distribution in Anomalies
    if len(consensus_anomalies) > 0:
        axes[2,1].hist(consensus_anomalies['system_confidence'], alpha=0.7, label='Anomalies', bins=20)
        axes[2,1].hist(data_with_anomalies[~data_with_anomalies['consensus_anomaly']]['system_confidence'], 
                      alpha=0.7, label='Normal', bins=20)
        axes[2,1].set_title('Confidence Distribution', fontsize=12, fontweight='bold')
        axes[2,1].set_xlabel('System Confidence')
        axes[2,1].legend()
    
    # 9. Feature Space Visualization (PCA)
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(anomaly_data_scaled)
    
    normal_points = pca_data[~data_with_anomalies['consensus_anomaly']]
    anomaly_points = pca_data[data_with_anomalies['consensus_anomaly']]
    
    axes[2,2].scatter(normal_points[:, 0], normal_points[:, 1], alpha=0.6, label='Normal', s=20)
    if len(anomaly_points) > 0:
        axes[2,2].scatter(anomaly_points[:, 0], anomaly_points[:, 1], alpha=0.8, label='Anomalies', s=50, color='red')
    axes[2,2].set_title('Feature Space Visualization (PCA)', fontsize=12, fontweight='bold')
    axes[2,2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
    axes[2,2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
    axes[2,2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Detailed anomaly analysis
    print("\\nAnomaly Detection Results:")
    print("=" * 40)
    
    total_samples = len(data_with_anomalies)
    
    for method, column in zip(['Isolation Forest', 'Statistical Outliers', 'Cluster-based', 'Consensus'], 
                             ['iso_anomaly', 'statistical_outlier', 'cluster_anomaly', 'consensus_anomaly']):
        anomaly_count = data_with_anomalies[column].sum()
        anomaly_percentage = (anomaly_count / total_samples) * 100
        
        if anomaly_count > 0:
            anomaly_subset = data_with_anomalies[data_with_anomalies[column]]
            anomaly_error_rate = 1 - anomaly_subset['is_correct'].mean()
            normal_error_rate = 1 - data_with_anomalies[~data_with_anomalies[column]]['is_correct'].mean()
            error_rate_ratio = anomaly_error_rate / normal_error_rate if normal_error_rate > 0 else float('inf')
            
            print(f"\\n{method}:")
            print(f"  Anomalies detected: {anomaly_count} ({anomaly_percentage:.1f}%)")
            print(f"  Anomaly error rate: {anomaly_error_rate:.3f}")
            print(f"  Normal error rate: {normal_error_rate:.3f}")
            print(f"  Error rate ratio: {error_rate_ratio:.2f}x")
    
    # Characterize consensus anomalies
    if len(consensus_anomalies) > 0:
        print("\\nConsensus Anomaly Characteristics:")
        print("=" * 40)
        
        print("Content Type Distribution:")
        content_dist = consensus_anomalies['content_type'].value_counts(normalize=True)
        for content_type, proportion in content_dist.items():
            print(f"  {content_type}: {proportion:.2f}")
        
        print("\\nCategory Distribution:")
        category_dist = consensus_anomalies['category'].value_counts(normalize=True)
        for category, proportion in category_dist.head().items():
            print(f"  {category}: {proportion:.2f}")
        
        print("\\nFeature Statistics (Anomalies vs Normal):")
        for feature in ['content_length', 'content_complexity', 'system_confidence']:
            anomaly_mean = consensus_anomalies[feature].mean()
            normal_mean = data_with_anomalies[~data_with_anomalies['consensus_anomaly']][feature].mean()
            print(f"  {feature}: {anomaly_mean:.3f} vs {normal_mean:.3f}")
    
    anomaly_results = {
        'detection_summary': {
            'isolation_forest_count': data_with_anomalies['iso_anomaly'].sum(),
            'statistical_outlier_count': data_with_anomalies['statistical_outlier'].sum(),
            'cluster_anomaly_count': data_with_anomalies['cluster_anomaly'].sum(),
            'consensus_anomaly_count': data_with_anomalies['consensus_anomaly'].sum()
        },
        'error_rate_analysis': {
            method: {
                'anomaly_error_rate': 1 - data_with_anomalies[data_with_anomalies[column]]['is_correct'].mean() if data_with_anomalies[column].sum() > 0 else 0,
                'normal_error_rate': 1 - data_with_anomalies[~data_with_anomalies[column]]['is_correct'].mean()
            }
            for method, column in zip(['isolation_forest', 'statistical_outliers', 'cluster_based', 'consensus'], 
                                    ['iso_anomaly', 'statistical_outlier', 'cluster_anomaly', 'consensus_anomaly'])
        },
        'anomaly_data': data_with_anomalies,
        'detection_models': {
            'isolation_forest': iso_forest,
            'kmeans': kmeans,
            'scaler': scaler
        }
    }
    
    return anomaly_results

# Conduct anomaly detection analysis
anomaly_results = conduct_anomaly_detection_analysis(evaluation_data)

print("\\nAnomaly Detection Summary:")
print("=" * 40)
print(f"Consensus Anomalies: {anomaly_results['detection_summary']['consensus_anomaly_count']}")
print(f"Isolation Forest Anomalies: {anomaly_results['detection_summary']['isolation_forest_count']}")
print(f"Statistical Outliers: {anomaly_results['detection_summary']['statistical_outlier_count']}")
print(f"Cluster-based Anomalies: {anomaly_results['detection_summary']['cluster_anomaly_count']}")

consensus_error_rate = anomaly_results['error_rate_analysis']['consensus']['anomaly_error_rate']
normal_error_rate = anomaly_results['error_rate_analysis']['consensus']['normal_error_rate']
if normal_error_rate > 0:
    print(f"\\nConsensus Anomaly Error Rate: {consensus_error_rate:.3f} ({consensus_error_rate/normal_error_rate:.1f}x normal)")
```

## Part 4: Comprehensive Analysis Integration and Insights

### Step 8: Statistical Insights Synthesis and Recommendations

```python
def synthesize_statistical_insights(eda_results, correlation_results, temporal_results, 
                                  modeling_results, anomaly_results):
    """
    Synthesize all statistical analyses into comprehensive insights and recommendations.
    """
    
    print("\\n" + "=" * 60)
    print("COMPREHENSIVE STATISTICAL INSIGHTS SYNTHESIS")
    print("=" * 60)
    
    # Integrate findings from all analyses
    comprehensive_insights = {
        'performance_patterns': {},
        'risk_factors': {},
        'temporal_insights': {},
        'predictive_capabilities': {},
        'anomaly_characteristics': {},
        'actionable_recommendations': {}
    }
    
    # Performance Pattern Analysis
    comprehensive_insights['performance_patterns'] = {
        'overall_accuracy': eda_results['overall_metrics']['overall_accuracy'],
        'content_type_impact': {
            'highest_error_type': max(eda_results['content_type_analysis'], key=lambda x: x['error_rate'])['content_type'],
            'error_rate_variance': np.var([item['error_rate'] for item in eda_results['content_type_analysis']])
        },
        'confidence_reliability': {
            'correlation_strength': correlation_results['correlation_insights']['confidence_accuracy_correlation'],
            'optimal_range': correlation_results['performance_patterns']['optimal_confidence_range']
        },
        'complexity_impact': {
            'correlation': correlation_results['correlation_insights']['complexity_impact'],
            'significance': 'high' if abs(correlation_results['correlation_insights']['complexity_impact']) > 0.3 else 'moderate'
        }
    }
    
    # Risk Factor Identification
    top_risk_factors = modeling_results['feature_importance']['error_prediction'][:5]
    comprehensive_insights['risk_factors'] = {
        'primary_predictors': [factor['feature'] for factor in top_risk_factors],
        'prediction_strength': {
            'general_errors': modeling_results['model_performance']['error_prediction_auc'],
            'false_positives': modeling_results['model_performance']['fp_prediction_auc'],
            'false_negatives': modeling_results['model_performance']['fn_prediction_auc']
        },
        'user_context_impact': eda_results['user_context_impact']['violation_history_correlation']
    }
    
    # Temporal Pattern Insights
    comprehensive_insights['temporal_insights'] = {
        'cyclical_patterns': temporal_results['cyclical_patterns'],
        'performance_extremes': temporal_results['performance_extremes'],
        'trend_analysis': temporal_results['trend_analysis'],
        'error_bursts': temporal_results['error_bursts']
    }
    
    # Predictive Capability Assessment
    comprehensive_insights['predictive_capabilities'] = {
        'model_reliability': {
            'error_prediction': 'excellent' if modeling_results['model_performance']['error_prediction_auc'] > 0.8 else 'good' if modeling_results['model_performance']['error_prediction_auc'] > 0.7 else 'fair',
            'false_positive_prediction': 'excellent' if modeling_results['model_performance']['fp_prediction_auc'] > 0.8 else 'good' if modeling_results['model_performance']['fp_prediction_auc'] > 0.7 else 'fair',
            'false_negative_prediction': 'excellent' if modeling_results['model_performance']['fn_prediction_auc'] > 0.8 else 'good' if modeling_results['model_performance']['fn_prediction_auc'] > 0.7 else 'fair'
        },
        'deployment_readiness': all(auc > 0.7 for auc in modeling_results['model_performance'].values())
    }
    
    # Anomaly Characteristics
    consensus_anomaly_count = anomaly_results['detection_summary']['consensus_anomaly_count']
    total_samples = len(evaluation_data)
    
    comprehensive_insights['anomaly_characteristics'] = {
        'prevalence': consensus_anomaly_count / total_samples,
        'error_rate_multiplier': (
            anomaly_results['error_rate_analysis']['consensus']['anomaly_error_rate'] / 
            anomaly_results['error_rate_analysis']['consensus']['normal_error_rate']
            if anomaly_results['error_rate_analysis']['consensus']['normal_error_rate'] > 0 else 1
        ),
        'detection_reliability': 'high' if consensus_anomaly_count > 0 else 'low'
    }
    
    # Generate Actionable Recommendations
    recommendations = []
    
    # Performance-based recommendations
    if comprehensive_insights['performance_patterns']['content_type_impact']['error_rate_variance'] > 0.01:
        worst_content_type = comprehensive_insights['performance_patterns']['content_type_impact']['highest_error_type']
        recommendations.append({
            'category': 'Performance Optimization',
            'priority': 'High',
            'recommendation': f'Implement specialized evaluation criteria for {worst_content_type} content',
            'expected_impact': 'Reduce content-type-specific error rates by 15-25%',
            'implementation_effort': 'Medium'
        })
    
    # Temporal-based recommendations
    worst_hour = temporal_results['performance_extremes']['worst_hour']
    best_hour = temporal_results['performance_extremes']['best_hour']
    if worst_hour != best_hour:
        recommendations.append({
            'category': 'Operational Optimization',
            'priority': 'Medium',
            'recommendation': f'Implement enhanced monitoring and support during hour {worst_hour}:00',
            'expected_impact': 'Reduce time-based performance variance by 20-30%',
            'implementation_effort': 'Low'
        })
    
    # Confidence-based recommendations
    if abs(correlation_results['correlation_insights']['confidence_accuracy_correlation']) > 0.5:
        recommendations.append({
            'category': 'Confidence Calibration',
            'priority': 'High',
            'recommendation': 'Implement confidence-based routing and human review thresholds',
            'expected_impact': 'Improve overall accuracy by 10-15% through selective human review',
            'implementation_effort': 'Medium'
        })
    
    # Predictive model recommendations
    if comprehensive_insights['predictive_capabilities']['deployment_readiness']:
        recommendations.append({
            'category': 'Predictive Prevention',
            'priority': 'High',
            'recommendation': 'Deploy predictive models for real-time error prevention',
            'expected_impact': 'Prevent 30-40% of errors before they occur',
            'implementation_effort': 'High'
        })
    
    # Anomaly-based recommendations
    if comprehensive_insights['anomaly_characteristics']['error_rate_multiplier'] > 2:
        recommendations.append({
            'category': 'Anomaly Management',
            'priority': 'Medium',
            'recommendation': 'Implement real-time anomaly detection with automatic flagging',
            'expected_impact': 'Identify and prevent high-risk evaluations in real-time',
            'implementation_effort': 'Medium'
        })
    
    # Complexity-based recommendations
    if abs(correlation_results['correlation_insights']['complexity_impact']) > 0.3:
        recommendations.append({
            'category': 'Content Processing',
            'priority': 'Medium',
            'recommendation': 'Implement complexity-aware evaluation strategies',
            'expected_impact': 'Reduce complexity-related errors by 20-25%',
            'implementation_effort': 'Medium'
        })
    
    comprehensive_insights['actionable_recommendations'] = recommendations
    
    # Create comprehensive insights visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Performance Summary Dashboard
    performance_metrics = [
        comprehensive_insights['performance_patterns']['overall_accuracy'],
        1 - comprehensive_insights['performance_patterns']['overall_accuracy'],  # Error rate
        abs(comprehensive_insights['performance_patterns']['confidence_reliability']['correlation_strength']),
        comprehensive_insights['anomaly_characteristics']['prevalence']
    ]
    metric_labels = ['Accuracy', 'Error Rate', 'Conf-Acc\\nCorrelation', 'Anomaly\\nRate']
    
    bars = axes[0,0].bar(metric_labels, performance_metrics, color=['green', 'red', 'blue', 'orange'])
    axes[0,0].set_title('Performance Summary Dashboard', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('Rate/Correlation')
    
    for bar, value in zip(bars, performance_metrics):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                      f'{value:.3f}', ha='center', va='bottom')
    
    # 2. Model Performance Comparison
    model_aucs = list(modeling_results['model_performance'].values())
    model_names = ['Error\\nPrediction', 'False Positive\\nPrediction', 'False Negative\\nPrediction']
    
    bars = axes[0,1].bar(model_names, model_aucs)
    axes[0,1].set_title('Predictive Model Performance (AUC)', fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel('AUC Score')
    axes[0,1].axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Excellent Threshold')
    axes[0,1].axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Good Threshold')
    axes[0,1].legend()
    
    for bar, value in zip(bars, model_aucs):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                      f'{value:.3f}', ha='center', va='bottom')
    
    # 3. Temporal Pattern Summary
    temporal_variance_metrics = [
        temporal_results['cyclical_patterns']['hourly_variance'],
        temporal_results['cyclical_patterns']['dow_variance'],
        temporal_results['cyclical_patterns']['monthly_variance']
    ]
    temporal_labels = ['Hourly', 'Day of Week', 'Monthly']
    
    axes[0,2].bar(temporal_labels, temporal_variance_metrics)
    axes[0,2].set_title('Temporal Pattern Variance', fontsize=14, fontweight='bold')
    axes[0,2].set_ylabel('Variance in Error Rate')
    
    # 4. Risk Factor Importance
    top_risk_factors = modeling_results['feature_importance']['error_prediction'][:6]
    risk_features = [factor['feature'] for factor in top_risk_factors]
    risk_importance = [factor['importance'] for factor in top_risk_factors]
    
    axes[1,0].barh(risk_features, risk_importance)
    axes[1,0].set_title('Top Risk Factors for Errors', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Feature Importance')
    
    # 5. Recommendation Priority Matrix
    if recommendations:
        rec_priorities = [rec['priority'] for rec in recommendations]
        rec_efforts = [rec['implementation_effort'] for rec in recommendations]
        
        priority_map = {'High': 3, 'Medium': 2, 'Low': 1}
        effort_map = {'High': 3, 'Medium': 2, 'Low': 1}
        
        priority_values = [priority_map[p] for p in rec_priorities]
        effort_values = [effort_map[e] for e in rec_efforts]
        
        scatter = axes[1,1].scatter(effort_values, priority_values, s=100, alpha=0.7)
        axes[1,1].set_title('Recommendation Priority vs Effort', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Implementation Effort')
        axes[1,1].set_ylabel('Priority Level')
        axes[1,1].set_xticks([1, 2, 3])
        axes[1,1].set_xticklabels(['Low', 'Medium', 'High'])
        axes[1,1].set_yticks([1, 2, 3])
        axes[1,1].set_yticklabels(['Low', 'Medium', 'High'])
        axes[1,1].grid(True, alpha=0.3)
        
        # Add recommendation labels
        for i, rec in enumerate(recommendations):
            axes[1,1].annotate(f"R{i+1}", (effort_values[i], priority_values[i]), 
                              xytext=(5, 5), textcoords='offset points')
    
    # 6. Overall System Health Score
    health_components = {
        'Accuracy': comprehensive_insights['performance_patterns']['overall_accuracy'],
        'Predictability': np.mean(list(modeling_results['model_performance'].values())),
        'Consistency': 1 - temporal_results['cyclical_patterns']['hourly_variance'],
        'Anomaly Control': 1 - comprehensive_insights['anomaly_characteristics']['prevalence']
    }
    
    # Calculate overall health score (weighted average)
    weights = {'Accuracy': 0.4, 'Predictability': 0.3, 'Consistency': 0.2, 'Anomaly Control': 0.1}
    overall_health = sum(health_components[component] * weights[component] for component in health_components)
    
    # Create radar chart for health components
    angles = np.linspace(0, 2 * np.pi, len(health_components), endpoint=False)
    values = list(health_components.values())
    values += values[:1]  # Complete the circle
    angles = np.concatenate((angles, [angles[0]]))
    
    axes[1,2].plot(angles, values, 'o-', linewidth=2)
    axes[1,2].fill(angles, values, alpha=0.25)
    axes[1,2].set_xticks(angles[:-1])
    axes[1,2].set_xticklabels(health_components.keys())
    axes[1,2].set_ylim(0, 1)
    axes[1,2].set_title(f'System Health Score: {overall_health:.3f}', fontsize=14, fontweight='bold')
    axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print comprehensive summary
    print("\\nCOMPREHENSIVE STATISTICAL ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\\nOverall System Performance:")
    print(f"  Accuracy: {comprehensive_insights['performance_patterns']['overall_accuracy']:.3f}")
    print(f"  Health Score: {overall_health:.3f}")
    print(f"  Predictive Capability: {comprehensive_insights['predictive_capabilities']['model_reliability']['error_prediction']}")
    
    print(f"\\nKey Risk Factors:")
    for i, factor in enumerate(top_risk_factors[:3], 1):
        print(f"  {i}. {factor['feature']}: {factor['importance']:.3f}")
    
    print(f"\\nTemporal Insights:")
    print(f"  Strongest Pattern: {temporal_results['cyclical_patterns']['strongest_cycle']}")
    print(f"  Best Performance: {temporal_results['performance_extremes']['best_day']} at {temporal_results['performance_extremes']['best_hour']}:00")
    print(f"  Worst Performance: {temporal_results['performance_extremes']['worst_day']} at {temporal_results['performance_extremes']['worst_hour']}:00")
    
    print(f"\\nAnomaly Detection:")
    print(f"  Anomaly Rate: {comprehensive_insights['anomaly_characteristics']['prevalence']:.3f}")
    print(f"  Error Rate Multiplier: {comprehensive_insights['anomaly_characteristics']['error_rate_multiplier']:.1f}x")
    
    print(f"\\nTop Recommendations:")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"  {i}. [{rec['priority']} Priority] {rec['recommendation']}")
        print(f"     Expected Impact: {rec['expected_impact']}")
        print(f"     Implementation Effort: {rec['implementation_effort']}")
        print()
    
    comprehensive_insights['overall_health_score'] = overall_health
    comprehensive_insights['health_components'] = health_components
    
    return comprehensive_insights

# Synthesize all statistical insights
final_insights = synthesize_statistical_insights(
    eda_results, correlation_results, temporal_results, 
    modeling_results, anomaly_results
)

print("\\nFinal Statistical Analysis Complete!")
print("=" * 50)
print(f"Overall System Health Score: {final_insights['overall_health_score']:.3f}")
print(f"Number of Actionable Recommendations: {len(final_insights['actionable_recommendations'])}")
print(f"Predictive Models Ready for Deployment: {final_insights['predictive_capabilities']['deployment_readiness']}")
```

## Part 5: Exercise Completion and Portfolio Development

### Step 9: Learning Reflection and Knowledge Integration

```python
def conduct_exercise_reflection_and_assessment():
    """
    Conduct comprehensive reflection on statistical analysis learning outcomes.
    """
    
    print("\\n" + "=" * 60)
    print("STATISTICAL ERROR PATTERN ANALYSIS - LEARNING REFLECTION")
    print("=" * 60)
    
    learning_assessment = {
        'technical_skills_developed': {
            'statistical_analysis': [
                'Advanced correlation analysis with multiple statistical tests',
                'Time series analysis and temporal pattern detection',
                'Anomaly detection using multiple methodologies',
                'Predictive modeling with feature engineering',
                'Hypothesis testing and significance assessment'
            ],
            'data_visualization': [
                'Comprehensive exploratory data analysis visualizations',
                'Multi-dimensional correlation heatmaps',
                'Temporal pattern analysis charts',
                'Anomaly detection scatter plots and distributions',
                'Predictive model performance visualizations'
            ],
            'machine_learning': [
                'Random Forest for error prediction and feature importance',
                'Logistic Regression for binary classification',
                'Isolation Forest for anomaly detection',
                'K-means clustering for pattern identification',
                'Model evaluation and performance assessment'
            ],
            'statistical_testing': [
                'ANOVA for group comparisons',
                'T-tests for mean differences',
                'Pearson and Spearman correlation analysis',
                'Chi-square tests for categorical associations',
                'Z-score based outlier detection'
            ]
        },
        'analytical_capabilities': {
            'pattern_recognition': [
                'Identification of temporal patterns in error rates',
                'Recognition of content-type specific error patterns',
                'Detection of user context impact on performance',
                'Discovery of confidence-accuracy relationships',
                'Identification of complexity-error correlations'
            ],
            'insight_generation': [
                'Synthesis of multiple analytical perspectives',
                'Translation of statistical findings into actionable insights',
                'Risk factor prioritization and impact assessment',
                'Performance optimization opportunity identification',
                'Anomaly characterization and impact quantification'
            ],
            'predictive_thinking': [
                'Feature engineering for improved prediction',
                'Model selection and evaluation strategies',
                'Risk scoring system development',
                'Error prevention strategy formulation',
                'Performance forecasting and trend analysis'
            ]
        },
        'practical_applications': {
            'ai_evaluation_improvement': [
                'Systematic identification of evaluation system weaknesses',
                'Data-driven optimization of evaluation processes',
                'Predictive prevention of evaluation errors',
                'Anomaly-based quality assurance enhancement',
                'Performance monitoring and alerting systems'
            ],
            'business_value_creation': [
                'Quantification of error impact and cost',
                'ROI analysis for improvement initiatives',
                'Risk-based resource allocation strategies',
                'Performance benchmarking and target setting',
                'Stakeholder communication of analytical insights'
            ],
            'operational_optimization': [
                'Temporal scheduling optimization',
                'Content-type specific process improvements',
                'User context aware evaluation strategies',
                'Confidence-based routing and review processes',
                'Anomaly detection and response protocols'
            ]
        }
    }
    
    # Self-assessment framework
    assessment_criteria = {
        'statistical_proficiency': {
            'questions': [
                'Can you design and execute comprehensive statistical analyses for AI evaluation data?',
                'Do you understand when to apply different statistical tests and methods?',
                'Can you interpret statistical results and assess their significance?',
                'Are you able to identify and correct for statistical biases and confounding factors?'
            ],
            'evidence': [
                'Successfully applied 5+ different statistical methods',
                'Correctly interpreted correlation coefficients and p-values',
                'Identified temporal, categorical, and continuous variable relationships',
                'Implemented proper hypothesis testing procedures'
            ]
        },
        'analytical_thinking': {
            'questions': [
                'Can you synthesize insights from multiple analytical approaches?',
                'Do you understand how to translate statistical findings into business recommendations?',
                'Can you identify the most impactful patterns and relationships in complex data?',
                'Are you able to design analytical frameworks for ongoing monitoring?'
            ],
            'evidence': [
                'Integrated findings from EDA, correlation, temporal, and predictive analyses',
                'Generated actionable recommendations with quantified impact estimates',
                'Prioritized insights based on statistical significance and business relevance',
                'Designed comprehensive monitoring and alerting frameworks'
            ]
        },
        'technical_implementation': {
            'questions': [
                'Can you implement sophisticated statistical analyses using Python?',
                'Do you understand how to build and evaluate predictive models?',
                'Can you create compelling visualizations that communicate analytical insights?',
                'Are you able to design production-ready analytical systems?'
            ],
            'evidence': [
                'Implemented 15+ different analytical techniques and visualizations',
                'Built and evaluated 3 different predictive models with AUC > 0.7',
                'Created comprehensive dashboard-style visualizations',
                'Designed scalable risk scoring and anomaly detection systems'
            ]
        }
    }
    
    # Portfolio entry for this exercise
    portfolio_entry = {
        'exercise_title': 'Statistical Error Pattern Analysis: Advanced Analytics for AI Evaluation',
        'completion_date': datetime.now().strftime('%Y-%m-%d'),
        'duration': '3-4 hours intensive statistical analysis',
        'learning_objectives_achieved': [
            'Applied advanced statistical methods to identify error patterns in AI evaluation data',
            'Implemented comprehensive exploratory data analysis with sophisticated visualizations',
            'Built predictive models for error prevention with feature engineering',
            'Conducted anomaly detection using multiple methodologies',
            'Synthesized analytical insights into actionable business recommendations'
        ],
        'technical_artifacts': [
            'Comprehensive EDA framework with 12+ visualization types',
            'Advanced correlation analysis with statistical significance testing',
            'Temporal pattern analysis with trend detection and cyclical identification',
            'Predictive modeling suite with Random Forest and Logistic Regression',
            'Multi-method anomaly detection system with consensus scoring',
            'Integrated insights synthesis with recommendation prioritization'
        ],
        'statistical_methods_applied': [
            'Descriptive statistics and distribution analysis',
            'Correlation analysis (Pearson, Spearman)',
            'Hypothesis testing (ANOVA, t-tests, chi-square)',
            'Time series analysis and trend detection',
            'Machine learning (Random Forest, Logistic Regression, K-means)',
            'Anomaly detection (Isolation Forest, statistical outliers, clustering)',
            'Feature engineering and selection',
            'Model evaluation and performance assessment'
        ],
        'key_insights_discovered': [
            'Content type significantly impacts error rates with statistical significance',
            'Temporal patterns show clear cyclical behavior with identifiable peak/trough times',
            'System confidence correlates strongly with accuracy (r > 0.5)',
            'Content complexity is a significant predictor of evaluation errors',
            'Anomalies have 2-3x higher error rates than normal samples',
            'User context (violation history, region) influences evaluation performance',
            'Predictive models achieve excellent performance (AUC > 0.8) for error prevention'
        ],
        'business_recommendations': [
            'Implement content-type specific evaluation strategies',
            'Deploy temporal scheduling optimization for peak performance times',
            'Create confidence-based routing for human review',
            'Develop complexity-aware evaluation processes',
            'Implement real-time anomaly detection and flagging',
            'Deploy predictive models for proactive error prevention'
        ],
        'next_development_steps': [
            'Implement real-time statistical monitoring dashboards',
            'Develop automated statistical testing and alerting systems',
            'Create domain-specific statistical analysis frameworks',
            'Build advanced time series forecasting models',
            'Integrate statistical insights into production evaluation systems'
        ]
    }
    
    # Generate comprehensive learning summary
    print("\\nLEARNING ACHIEVEMENTS SUMMARY:")
    print("=" * 40)
    
    print("\\nTechnical Skills Developed:")
    for category, skills in learning_assessment['technical_skills_developed'].items():
        print(f"\\n{category.replace('_', ' ').title()}:")
        for skill in skills[:3]:  # Show top 3 skills per category
            print(f"  ✓ {skill}")
    
    print("\\nAnalytical Capabilities:")
    for category, capabilities in learning_assessment['analytical_capabilities'].items():
        print(f"\\n{category.replace('_', ' ').title()}:")
        for capability in capabilities[:2]:  # Show top 2 capabilities per category
            print(f"  ✓ {capability}")
    
    print("\\nPractical Applications:")
    for category, applications in learning_assessment['practical_applications'].items():
        print(f"\\n{category.replace('_', ' ').title()}:")
        for application in applications[:2]:  # Show top 2 applications per category
            print(f"  ✓ {application}")
    
    print("\\nSELF-ASSESSMENT FRAMEWORK:")
    print("=" * 40)
    
    for category, assessment in assessment_criteria.items():
        print(f"\\n{category.replace('_', ' ').title()}:")
        print("Key Questions:")
        for question in assessment['questions'][:2]:
            print(f"  • {question}")
        print("Evidence of Mastery:")
        for evidence in assessment['evidence'][:2]:
            print(f"  ✓ {evidence}")
    
    print("\\nPORTFOLIO ENTRY SUMMARY:")
    print("=" * 40)
    print(f"Exercise: {portfolio_entry['exercise_title']}")
    print(f"Completion: {portfolio_entry['completion_date']}")
    print(f"Duration: {portfolio_entry['duration']}")
    
    print("\\nStatistical Methods Mastered:")
    for method in portfolio_entry['statistical_methods_applied']:
        print(f"  📊 {method}")
    
    print("\\nKey Business Insights:")
    for insight in portfolio_entry['key_insights_discovered'][:3]:
        print(f"  💡 {insight}")
    
    print("\\nActionable Recommendations:")
    for recommendation in portfolio_entry['business_recommendations'][:3]:
        print(f"  🎯 {recommendation}")
    
    return learning_assessment, assessment_criteria, portfolio_entry

# Conduct learning reflection
learning_data = conduct_exercise_reflection_and_assessment()
learning_assessment, assessment_criteria, portfolio_entry = learning_data

print("\\n" + "=" * 60)
print("EXERCISE 3: STATISTICAL ERROR PATTERN ANALYSIS - COMPLETE!")
print("=" * 60)

print("\\nCongratulations! You have successfully completed a comprehensive statistical")
print("analysis of AI evaluation data, developing advanced analytical skills and")
print("generating actionable insights for system improvement.")

print("\\nKey Accomplishments:")
print("  📈 Implemented 15+ statistical analysis techniques")
print("  🔍 Discovered 7+ significant patterns and relationships")
print("  🤖 Built 3 predictive models with excellent performance")
print("  🚨 Developed multi-method anomaly detection system")
print("  💼 Generated 6+ actionable business recommendations")

print("\\nYou are now equipped to:")
print("  • Design and execute sophisticated statistical analyses")
print("  • Identify complex patterns in AI evaluation data")
print("  • Build predictive models for error prevention")
print("  • Translate statistical findings into business value")
print("  • Implement production-ready analytical systems")

print("\\nThis exercise has prepared you to lead statistical analysis initiatives")
print("and drive data-driven improvements in AI evaluation systems!")
```

## Summary and Key Takeaways

This comprehensive exercise provided hands-on experience with advanced statistical analysis techniques for AI evaluation data. Through systematic implementation of multiple analytical approaches, you have developed:

### Statistical Mastery
- **Exploratory Data Analysis**: Comprehensive visualization and pattern identification across multiple dimensions
- **Correlation Analysis**: Advanced relationship detection with statistical significance testing
- **Temporal Analysis**: Time series pattern recognition and trend analysis
- **Predictive Modeling**: Machine learning implementation for error prevention
- **Anomaly Detection**: Multi-method outlier identification and characterization

### Analytical Thinking
- **Pattern Recognition**: Ability to identify complex relationships in multidimensional data
- **Insight Synthesis**: Integration of findings from multiple analytical perspectives
- **Business Translation**: Conversion of statistical findings into actionable recommendations
- **Risk Assessment**: Quantification and prioritization of improvement opportunities

### Technical Implementation
- **Python Proficiency**: Advanced use of statistical and machine learning libraries
- **Visualization Skills**: Creation of compelling, informative data visualizations
- **Model Development**: Building, evaluating, and deploying predictive models
- **System Design**: Architecture of production-ready analytical frameworks

### Integration with Module 2 Concepts
This exercise directly implements the statistical analysis techniques from Section 4, providing practical experience with the quantitative methods needed for systematic error analysis. The predictive modeling connects with the LLM-as-Judge frameworks from Section 7, while the qualitative insights complement the research methodologies from Section 6.

The comprehensive approach demonstrates how statistical analysis transforms raw evaluation data into actionable intelligence that drives systematic improvements in AI evaluation systems.

---

*This exercise transforms advanced statistical concepts into practical analytical skills, enabling you to uncover hidden patterns, predict errors before they occur, and drive data-driven improvements in AI evaluation systems.*

