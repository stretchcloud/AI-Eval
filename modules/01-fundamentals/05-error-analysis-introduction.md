# Error Analysis Introduction: Understanding AI System Failures

## Introduction: Beyond Simple Bug Fixing

Error analysis in AI systems represents a fundamental shift from traditional software debugging. While conventional software errors typically have clear causes and deterministic solutions, AI system failures often emerge from complex interactions between data, models, and context that require sophisticated analysis techniques to understand and address.

Traditional debugging focuses on identifying and fixing specific code defects that cause predictable failures. AI error analysis, by contrast, must grapple with probabilistic behaviors, emergent patterns, and context-dependent failures that may not have simple root causes or straightforward solutions.

This complexity makes error analysis both more challenging and more critical for AI systems. A single misunderstood failure mode can lead to systematic issues that affect thousands of users, while effective error analysis can reveal improvement opportunities that enhance the entire system's performance.

## Types of AI System Errors

### Accuracy and Factual Errors

Accuracy errors represent one of the most visible and concerning types of AI system failures. These errors occur when the AI system provides information that is objectively incorrect, potentially misleading users and undermining trust in the system.

Factual errors can range from simple mistakes, such as incorrect dates or numbers, to more complex inaccuracies involving relationships between concepts or causal reasoning. The challenge in addressing factual errors is that they often stem from training data limitations or model reasoning failures that are difficult to predict or prevent systematically.

The impact of factual errors varies significantly based on the application domain and user context. In a casual conversation, a minor factual error might be inconsequential, while the same error in a medical or financial application could have serious consequences.

Detecting factual errors requires specialized evaluation approaches that can verify claims against authoritative sources or expert knowledge. This verification process is often computationally expensive and may require human oversight, making it challenging to implement at scale.

### Appropriateness and Context Errors

Appropriateness errors occur when AI systems provide responses that are technically accurate but inappropriate for the specific context, user, or situation. These errors highlight the challenge of context understanding in AI systems and the difficulty of encoding social and cultural norms into automated systems.

Context errors can manifest in various ways: using overly technical language for a general audience, providing inappropriate humor in serious situations, or failing to recognize cultural sensitivities. These errors are particularly challenging because they require understanding not just the literal content of the interaction but also the broader social and situational context.

The subjective nature of appropriateness makes these errors difficult to detect and address systematically. What one user considers appropriate, another might find offensive or unhelpful. This variability requires evaluation approaches that can account for different perspectives and contexts.

Addressing appropriateness errors often requires fine-tuning AI systems for specific use cases, audiences, or cultural contexts. This customization can be resource-intensive and may require ongoing adjustment as social norms and expectations evolve.

### Consistency and Reliability Errors

Consistency errors occur when AI systems provide different responses to similar inputs or when their behavior varies unpredictably across similar contexts. These errors undermine user trust and make it difficult for users to develop reliable mental models of system behavior.

Reliability errors encompass broader patterns of inconsistent performance, including degradation under load, sensitivity to minor input variations, and unpredictable failure modes. These errors are particularly problematic because they make AI systems feel unreliable and unprofessional.

The stochastic nature of AI systems makes some degree of output variation inevitable and even desirable. The challenge is distinguishing between beneficial diversity and problematic inconsistency. This distinction requires careful analysis of output patterns and user expectations.

Addressing consistency errors often involves adjusting model parameters, improving training data quality, or implementing post-processing techniques that reduce unwanted variation while preserving beneficial diversity.

### Safety and Bias Errors

Safety errors represent the most serious category of AI system failures, encompassing outputs that could cause harm to users or society. These errors include generation of harmful content, amplification of dangerous misinformation, and facilitation of illegal or unethical activities.

Bias errors involve systematic unfairness or discrimination in AI system behavior, often reflecting biases present in training data or model architecture. These errors can perpetuate or amplify existing social inequalities and may violate legal or ethical standards for fair treatment.

The detection of safety and bias errors requires specialized evaluation approaches that go beyond traditional performance metrics. These approaches must consider not just what the system produces but also the potential impacts on different user populations and society as a whole.

Addressing safety and bias errors often requires fundamental changes to training data, model architecture, or system design. These changes can be complex and resource-intensive, but they are essential for building trustworthy AI systems.

## Systematic Error Analysis Methodology

### Error Classification and Categorization

Effective error analysis begins with systematic classification and categorization of observed failures. This classification provides the foundation for understanding error patterns, prioritizing improvement efforts, and tracking progress over time.

Error classification schemes should be tailored to the specific AI system and application domain while maintaining consistency and clarity. Common classification dimensions include error type (accuracy, appropriateness, safety), severity (critical, major, minor), frequency (common, occasional, rare), and impact (user experience, business metrics, safety).

The classification process should be systematic and reproducible, with clear criteria for assigning errors to different categories. This consistency enables meaningful analysis of error patterns and trends over time.

Automated classification tools can help scale the error analysis process, but human oversight remains essential for nuanced categorization and for identifying new error types that automated systems might miss.

### Root Cause Analysis

Root cause analysis for AI systems requires understanding the complex interactions between data, models, and context that contribute to system failures. This analysis goes beyond identifying immediate causes to understand the underlying factors that make errors possible or likely.

Common root causes in AI systems include training data limitations, model architecture constraints, insufficient context information, and misalignment between training objectives and real-world requirements. Identifying these root causes requires deep understanding of both the technical system and the application domain.

The probabilistic nature of AI systems means that root cause analysis must consider statistical patterns rather than deterministic relationships. This analysis often involves examining error distributions, correlations with input characteristics, and patterns across different user populations or contexts.

Effective root cause analysis combines quantitative analysis of error patterns with qualitative investigation of specific failure cases. This combination provides both statistical insight into systematic issues and detailed understanding of individual failure modes.

### Impact Assessment

Understanding the impact of different error types is crucial for prioritizing improvement efforts and making informed decisions about system deployment and updates. Impact assessment considers both immediate effects on users and broader implications for business objectives and system reliability.

User impact assessment examines how different error types affect user experience, task completion, and satisfaction. This assessment should consider not just the immediate frustration of encountering an error but also the longer-term effects on user trust and engagement.

Business impact assessment evaluates how errors affect key business metrics such as user retention, conversion rates, and support costs. This assessment helps prioritize error types that have the greatest business consequences and justify investment in improvement efforts.

System impact assessment considers how errors affect the overall reliability and performance of the AI system. Some errors may have cascading effects that impact other system components or create additional failure modes.

## Practical Error Analysis Techniques

### Statistical Error Analysis

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum

class ErrorType(Enum):
    FACTUAL = "factual"
    APPROPRIATENESS = "appropriateness"
    CONSISTENCY = "consistency"
    SAFETY = "safety"
    BIAS = "bias"

class ErrorSeverity(Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"

@dataclass
class ErrorRecord:
    """Structured representation of an AI system error"""
    error_id: str
    timestamp: float
    error_type: ErrorType
    severity: ErrorSeverity
    input_text: str
    output_text: str
    expected_output: str
    user_id: str
    session_id: str
    context: Dict[str, Any]
    human_assessment: Dict[str, Any]

class ErrorAnalyzer:
    """Statistical analysis of AI system errors"""
    
    def __init__(self):
        self.errors: List[ErrorRecord] = []
    
    def add_error(self, error: ErrorRecord):
        """Add an error record for analysis"""
        self.errors.append(error)
    
    def load_errors_from_csv(self, filepath: str):
        """Load error records from CSV file"""
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            error = ErrorRecord(
                error_id=row['error_id'],
                timestamp=row['timestamp'],
                error_type=ErrorType(row['error_type']),
                severity=ErrorSeverity(row['severity']),
                input_text=row['input_text'],
                output_text=row['output_text'],
                expected_output=row['expected_output'],
                user_id=row['user_id'],
                session_id=row['session_id'],
                context=eval(row['context']) if row['context'] else {},
                human_assessment=eval(row['human_assessment']) if row['human_assessment'] else {}
            )
            self.errors.append(error)
    
    def analyze_error_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of errors by type and severity"""
        if not self.errors:
            return {}
        
        df = pd.DataFrame([
            {
                'error_type': error.error_type.value,
                'severity': error.severity.value,
                'timestamp': error.timestamp
            }
            for error in self.errors
        ])
        
        # Error type distribution
        type_distribution = df['error_type'].value_counts().to_dict()
        
        # Severity distribution
        severity_distribution = df['severity'].value_counts().to_dict()
        
        # Error type by severity
        type_severity_crosstab = pd.crosstab(df['error_type'], df['severity'])
        
        return {
            'total_errors': len(self.errors),
            'error_type_distribution': type_distribution,
            'severity_distribution': severity_distribution,
            'type_severity_matrix': type_severity_crosstab.to_dict()
        }
    
    def analyze_temporal_patterns(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Analyze temporal patterns in error occurrence"""
        if not self.errors:
            return {}
        
        df = pd.DataFrame([
            {
                'timestamp': error.timestamp,
                'error_type': error.error_type.value,
                'severity': error.severity.value
            }
            for error in self.errors
        ])
        
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        
        # Hourly distribution
        hourly_distribution = df['hour'].value_counts().sort_index().to_dict()
        
        # Daily distribution
        daily_distribution = df['day_of_week'].value_counts().sort_index().to_dict()
        
        # Error rate over time
        df_resampled = df.set_index('datetime').resample(f'{time_window_hours}H').size()
        
        return {
            'hourly_distribution': hourly_distribution,
            'daily_distribution': daily_distribution,
            'error_rate_over_time': df_resampled.to_dict()
        }
    
    def identify_error_clusters(self) -> List[Dict[str, Any]]:
        """Identify clusters of similar errors"""
        if not self.errors:
            return []
        
        # Group errors by similar characteristics
        clusters = {}
        
        for error in self.errors:
            # Create a cluster key based on error characteristics
            cluster_key = (
                error.error_type.value,
                error.severity.value,
                len(error.input_text.split()),  # Input length category
                len(error.output_text.split())   # Output length category
            )
            
            if cluster_key not in clusters:
                clusters[cluster_key] = []
            clusters[cluster_key].append(error)
        
        # Convert to list of cluster summaries
        cluster_summaries = []
        for cluster_key, cluster_errors in clusters.items():
            if len(cluster_errors) >= 2:  # Only include clusters with multiple errors
                cluster_summaries.append({
                    'cluster_id': str(hash(cluster_key)),
                    'error_type': cluster_key[0],
                    'severity': cluster_key[1],
                    'input_length_category': cluster_key[2],
                    'output_length_category': cluster_key[3],
                    'error_count': len(cluster_errors),
                    'sample_errors': [
                        {
                            'error_id': error.error_id,
                            'input_text': error.input_text[:100] + '...' if len(error.input_text) > 100 else error.input_text,
                            'output_text': error.output_text[:100] + '...' if len(error.output_text) > 100 else error.output_text
                        }
                        for error in cluster_errors[:3]  # Show up to 3 sample errors
                    ]
                })
        
        return sorted(cluster_summaries, key=lambda x: x['error_count'], reverse=True)
    
    def generate_error_report(self) -> str:
        """Generate comprehensive error analysis report"""
        distribution = self.analyze_error_distribution()
        temporal = self.analyze_temporal_patterns()
        clusters = self.identify_error_clusters()
        
        report = f"""
# AI System Error Analysis Report

## Summary
- Total Errors: {distribution.get('total_errors', 0)}
- Analysis Period: {len(self.errors)} error records

## Error Distribution

### By Type
{self._format_distribution(distribution.get('error_type_distribution', {}))}

### By Severity
{self._format_distribution(distribution.get('severity_distribution', {}))}

## Temporal Patterns

### Peak Error Hours
{self._format_temporal_patterns(temporal.get('hourly_distribution', {}))}

### Peak Error Days
{self._format_temporal_patterns(temporal.get('daily_distribution', {}), day_names=True)}

## Error Clusters

{self._format_clusters(clusters)}

## Recommendations

{self._generate_recommendations(distribution, temporal, clusters)}
        """
        
        return report.strip()
    
    def _format_distribution(self, distribution: Dict[str, int]) -> str:
        """Format distribution data for report"""
        if not distribution:
            return "No data available"
        
        total = sum(distribution.values())
        formatted = []
        for key, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
            formatted.append(f"- {key}: {count} ({percentage:.1f}%)")
        
        return "\n".join(formatted)
    
    def _format_temporal_patterns(self, patterns: Dict[int, int], day_names: bool = False) -> str:
        """Format temporal patterns for report"""
        if not patterns:
            return "No data available"
        
        day_name_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                       4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        formatted = []
        
        for key, count in sorted_patterns:
            if day_names:
                key_str = day_name_map.get(key, str(key))
            else:
                key_str = f"{key}:00"
            formatted.append(f"- {key_str}: {count} errors")
        
        return "\n".join(formatted)
    
    def _format_clusters(self, clusters: List[Dict[str, Any]]) -> str:
        """Format cluster data for report"""
        if not clusters:
            return "No significant error clusters identified"
        
        formatted = []
        for i, cluster in enumerate(clusters[:5], 1):  # Show top 5 clusters
            formatted.append(f"""
### Cluster {i}: {cluster['error_type'].title()} Errors
- Count: {cluster['error_count']} errors
- Severity: {cluster['severity'].title()}
- Sample Error: "{cluster['sample_errors'][0]['input_text']}" → "{cluster['sample_errors'][0]['output_text']}"
            """.strip())
        
        return "\n".join(formatted)
    
    def _generate_recommendations(self, distribution: Dict[str, Any], 
                                temporal: Dict[str, Any], 
                                clusters: List[Dict[str, Any]]) -> str:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Recommendations based on error distribution
        if distribution.get('error_type_distribution'):
            top_error_type = max(distribution['error_type_distribution'].items(), key=lambda x: x[1])
            recommendations.append(f"1. Focus on {top_error_type[0]} errors, which represent the largest category ({top_error_type[1]} errors)")
        
        # Recommendations based on severity
        if distribution.get('severity_distribution'):
            critical_count = distribution['severity_distribution'].get('critical', 0)
            if critical_count > 0:
                recommendations.append(f"2. Address {critical_count} critical errors as highest priority")
        
        # Recommendations based on clusters
        if clusters:
            top_cluster = clusters[0]
            recommendations.append(f"3. Investigate the largest error cluster ({top_cluster['error_count']} {top_cluster['error_type']} errors)")
        
        # Recommendations based on temporal patterns
        if temporal.get('hourly_distribution'):
            peak_hour = max(temporal['hourly_distribution'].items(), key=lambda x: x[1])
            recommendations.append(f"4. Monitor system performance during peak error time ({peak_hour[0]}:00)")
        
        if not recommendations:
            recommendations.append("1. Continue monitoring for error patterns")
            recommendations.append("2. Implement more comprehensive error tracking")
        
        return "\n".join(recommendations)

# Usage example
analyzer = ErrorAnalyzer()

# Add sample errors
sample_errors = [
    ErrorRecord(
        error_id="err_001",
        timestamp=1703097600,  # 2023-12-20 12:00:00
        error_type=ErrorType.FACTUAL,
        severity=ErrorSeverity.MAJOR,
        input_text="What is the capital of France?",
        output_text="The capital of France is Berlin.",
        expected_output="The capital of France is Paris.",
        user_id="user_123",
        session_id="session_456",
        context={"language": "en", "topic": "geography"},
        human_assessment={"confidence": 0.9, "impact": "high"}
    ),
    ErrorRecord(
        error_id="err_002",
        timestamp=1703101200,  # 2023-12-20 13:00:00
        error_type=ErrorType.APPROPRIATENESS,
        severity=ErrorSeverity.MINOR,
        input_text="I'm feeling sad today",
        output_text="That's great! Keep up the positive attitude!",
        expected_output="I'm sorry to hear you're feeling sad. Would you like to talk about what's bothering you?",
        user_id="user_124",
        session_id="session_457",
        context={"language": "en", "topic": "emotional_support"},
        human_assessment={"confidence": 0.8, "impact": "medium"}
    )
]

for error in sample_errors:
    analyzer.add_error(error)

# Generate analysis report
report = analyzer.generate_error_report()
print(report)
```

### Qualitative Error Analysis

```python
from typing import List, Dict, Any, Optional
import re
from collections import defaultdict

class QualitativeErrorAnalyzer:
    """Qualitative analysis of AI system errors"""
    
    def __init__(self):
        self.error_patterns = defaultdict(list)
        self.linguistic_patterns = defaultdict(list)
        self.context_patterns = defaultdict(list)
    
    def analyze_linguistic_patterns(self, errors: List[ErrorRecord]) -> Dict[str, Any]:
        """Analyze linguistic patterns in errors"""
        patterns = {
            'input_patterns': defaultdict(int),
            'output_patterns': defaultdict(int),
            'length_correlations': [],
            'complexity_indicators': []
        }
        
        for error in errors:
            # Analyze input patterns
            input_words = error.input_text.lower().split()
            input_length = len(input_words)
            
            # Look for question patterns
            if error.input_text.strip().endswith('?'):
                patterns['input_patterns']['questions'] += 1
            
            # Look for command patterns
            if any(word in input_words for word in ['please', 'can you', 'help me']):
                patterns['input_patterns']['requests'] += 1
            
            # Analyze output patterns
            output_words = error.output_text.lower().split()
            output_length = len(output_words)
            
            # Look for hedge words indicating uncertainty
            hedge_words = ['maybe', 'perhaps', 'might', 'could', 'possibly']
            if any(word in output_words for word in hedge_words):
                patterns['output_patterns']['uncertain'] += 1
            
            # Look for overly confident language
            confident_words = ['definitely', 'certainly', 'absolutely', 'always', 'never']
            if any(word in output_words for word in confident_words):
                patterns['output_patterns']['overconfident'] += 1
            
            # Track length correlations
            patterns['length_correlations'].append({
                'input_length': input_length,
                'output_length': output_length,
                'error_type': error.error_type.value,
                'severity': error.severity.value
            })
            
            # Analyze complexity indicators
            complex_indicators = {
                'technical_terms': len(re.findall(r'\b[A-Z]{2,}\b', error.input_text)),
                'numbers': len(re.findall(r'\d+', error.input_text)),
                'punctuation_density': len(re.findall(r'[^\w\s]', error.input_text)) / len(error.input_text)
            }
            
            patterns['complexity_indicators'].append({
                'error_id': error.error_id,
                'complexity_score': sum(complex_indicators.values()),
                'indicators': complex_indicators,
                'error_type': error.error_type.value
            })
        
        return patterns
    
    def analyze_context_patterns(self, errors: List[ErrorRecord]) -> Dict[str, Any]:
        """Analyze contextual patterns in errors"""
        context_analysis = {
            'topic_distribution': defaultdict(int),
            'user_patterns': defaultdict(int),
            'session_patterns': defaultdict(list),
            'environmental_factors': defaultdict(int)
        }
        
        for error in errors:
            # Analyze topic context
            if 'topic' in error.context:
                context_analysis['topic_distribution'][error.context['topic']] += 1
            
            # Analyze user patterns
            context_analysis['user_patterns'][error.user_id] += 1
            
            # Analyze session patterns
            context_analysis['session_patterns'][error.session_id].append({
                'error_type': error.error_type.value,
                'timestamp': error.timestamp
            })
            
            # Analyze environmental factors
            for key, value in error.context.items():
                if key in ['language', 'platform', 'device_type']:
                    context_analysis['environmental_factors'][f"{key}_{value}"] += 1
        
        # Identify problematic sessions (multiple errors)
        problematic_sessions = {
            session_id: errors_list 
            for session_id, errors_list in context_analysis['session_patterns'].items()
            if len(errors_list) > 1
        }
        
        context_analysis['problematic_sessions'] = problematic_sessions
        
        return context_analysis
    
    def identify_error_themes(self, errors: List[ErrorRecord]) -> List[Dict[str, Any]]:
        """Identify common themes across errors"""
        themes = []
        
        # Group errors by type for thematic analysis
        errors_by_type = defaultdict(list)
        for error in errors:
            errors_by_type[error.error_type].append(error)
        
        for error_type, type_errors in errors_by_type.items():
            if len(type_errors) < 2:
                continue
            
            # Analyze common words in inputs
            all_input_words = []
            for error in type_errors:
                all_input_words.extend(error.input_text.lower().split())
            
            word_freq = defaultdict(int)
            for word in all_input_words:
                if len(word) > 3:  # Skip short words
                    word_freq[word] += 1
            
            # Find common themes
            common_words = [word for word, freq in word_freq.items() if freq >= len(type_errors) * 0.3]
            
            if common_words:
                themes.append({
                    'error_type': error_type.value,
                    'theme': f"Common words: {', '.join(common_words[:5])}",
                    'frequency': len(type_errors),
                    'sample_errors': [error.error_id for error in type_errors[:3]]
                })
        
        return themes
    
    def generate_qualitative_insights(self, errors: List[ErrorRecord]) -> str:
        """Generate qualitative insights report"""
        linguistic = self.analyze_linguistic_patterns(errors)
        contextual = self.analyze_context_patterns(errors)
        themes = self.identify_error_themes(errors)
        
        report = f"""
# Qualitative Error Analysis

## Linguistic Patterns

### Input Characteristics
- Questions: {linguistic['input_patterns']['questions']} errors
- Requests: {linguistic['input_patterns']['requests']} errors

### Output Characteristics
- Uncertain language: {linguistic['output_patterns']['uncertain']} errors
- Overconfident language: {linguistic['output_patterns']['overconfident']} errors

## Contextual Patterns

### Topic Distribution
{self._format_distribution(dict(contextual['topic_distribution']))}

### Problematic Sessions
{len(contextual['problematic_sessions'])} sessions with multiple errors

### Environmental Factors
{self._format_distribution(dict(contextual['environmental_factors']))}

## Common Themes
{self._format_themes(themes)}

## Key Insights

{self._generate_qualitative_insights(linguistic, contextual, themes)}
        """
        
        return report.strip()
    
    def _format_distribution(self, distribution: Dict[str, int]) -> str:
        """Format distribution for report"""
        if not distribution:
            return "No patterns identified"
        
        sorted_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        return "\n".join([f"- {key}: {value}" for key, value in sorted_items[:5]])
    
    def _format_themes(self, themes: List[Dict[str, Any]]) -> str:
        """Format themes for report"""
        if not themes:
            return "No common themes identified"
        
        formatted = []
        for theme in themes:
            formatted.append(f"- {theme['error_type'].title()}: {theme['theme']} ({theme['frequency']} errors)")
        
        return "\n".join(formatted)
    
    def _generate_qualitative_insights(self, linguistic: Dict[str, Any], 
                                     contextual: Dict[str, Any], 
                                     themes: List[Dict[str, Any]]) -> str:
        """Generate key insights from qualitative analysis"""
        insights = []
        
        # Linguistic insights
        if linguistic['output_patterns']['uncertain'] > 0:
            insights.append("System shows uncertainty in responses, which may indicate training data gaps")
        
        if linguistic['output_patterns']['overconfident'] > 0:
            insights.append("System exhibits overconfident language, which may mislead users")
        
        # Contextual insights
        if contextual['problematic_sessions']:
            insights.append(f"{len(contextual['problematic_sessions'])} sessions had multiple errors, suggesting systematic issues")
        
        # Theme insights
        if themes:
            insights.append(f"Identified {len(themes)} common error themes that could guide targeted improvements")
        
        if not insights:
            insights.append("No significant qualitative patterns identified in current dataset")
        
        return "\n".join([f"{i+1}. {insight}" for i, insight in enumerate(insights)])

# Usage example
qualitative_analyzer = QualitativeErrorAnalyzer()
insights_report = qualitative_analyzer.generate_qualitative_insights(sample_errors)
print(insights_report)
```

## Building Error Analysis Workflows

### Automated Error Detection

Implementing automated error detection systems enables teams to identify and categorize errors at scale, providing the foundation for systematic error analysis.

```python
import asyncio
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import re

class DetectionMethod(Enum):
    RULE_BASED = "rule_based"
    ML_CLASSIFIER = "ml_classifier"
    SIMILARITY_CHECK = "similarity_check"
    EXTERNAL_API = "external_api"

@dataclass
class ErrorDetectionResult:
    """Result of automated error detection"""
    is_error: bool
    error_type: Optional[ErrorType]
    confidence: float
    explanation: str
    detection_method: DetectionMethod
    metadata: Dict[str, Any]

class AutomatedErrorDetector:
    """Automated detection of AI system errors"""
    
    def __init__(self):
        self.detectors: List[Callable] = []
        self.detection_history: List[ErrorDetectionResult] = []
    
    def register_detector(self, detector: Callable):
        """Register an error detection function"""
        self.detectors.append(detector)
    
    async def detect_errors(self, input_text: str, output_text: str, 
                          context: Dict[str, Any]) -> List[ErrorDetectionResult]:
        """Run all registered detectors on input/output pair"""
        results = []
        
        for detector in self.detectors:
            try:
                result = await detector(input_text, output_text, context)
                if isinstance(result, ErrorDetectionResult):
                    results.append(result)
                    self.detection_history.append(result)
            except Exception as e:
                # Log detection errors but don't fail the entire process
                print(f"Error in detector {detector.__name__}: {e}")
        
        return results
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get statistics on detection performance"""
        if not self.detection_history:
            return {}
        
        total_detections = len(self.detection_history)
        error_detections = sum(1 for r in self.detection_history if r.is_error)
        
        method_stats = {}
        for method in DetectionMethod:
            method_results = [r for r in self.detection_history if r.detection_method == method]
            if method_results:
                method_stats[method.value] = {
                    'total': len(method_results),
                    'errors_detected': sum(1 for r in method_results if r.is_error),
                    'avg_confidence': sum(r.confidence for r in method_results) / len(method_results)
                }
        
        return {
            'total_detections': total_detections,
            'errors_detected': error_detections,
            'error_rate': error_detections / total_detections if total_detections > 0 else 0,
            'method_stats': method_stats
        }

# Example detectors
async def factual_accuracy_detector(input_text: str, output_text: str, 
                                  context: Dict[str, Any]) -> ErrorDetectionResult:
    """Detect potential factual accuracy errors"""
    
    # Simple rule-based checks
    confidence = 0.6
    is_error = False
    explanation = "No obvious factual errors detected"
    
    # Check for contradictory statements
    if "yes" in output_text.lower() and "no" in output_text.lower():
        is_error = True
        confidence = 0.8
        explanation = "Output contains contradictory statements"
    
    # Check for impossible dates
    date_pattern = r'\b(19|20)\d{2}\b'
    dates = re.findall(date_pattern, output_text)
    if dates:
        for date in dates:
            if int(date) > 2024:  # Future dates might be errors
                is_error = True
                confidence = 0.7
                explanation = f"Output contains future date: {date}"
    
    return ErrorDetectionResult(
        is_error=is_error,
        error_type=ErrorType.FACTUAL if is_error else None,
        confidence=confidence,
        explanation=explanation,
        detection_method=DetectionMethod.RULE_BASED,
        metadata={"dates_found": dates}
    )

async def safety_detector(input_text: str, output_text: str, 
                         context: Dict[str, Any]) -> ErrorDetectionResult:
    """Detect potential safety issues"""
    
    # List of concerning keywords
    safety_keywords = [
        "violence", "harm", "hurt", "kill", "weapon", "bomb", "poison",
        "suicide", "self-harm", "illegal", "drugs", "hack", "steal"
    ]
    
    output_lower = output_text.lower()
    found_keywords = [kw for kw in safety_keywords if kw in output_lower]
    
    if found_keywords:
        confidence = min(0.9, 0.5 + len(found_keywords) * 0.1)
        return ErrorDetectionResult(
            is_error=True,
            error_type=ErrorType.SAFETY,
            confidence=confidence,
            explanation=f"Detected safety-related keywords: {', '.join(found_keywords)}",
            detection_method=DetectionMethod.RULE_BASED,
            metadata={"keywords_found": found_keywords}
        )
    
    return ErrorDetectionResult(
        is_error=False,
        error_type=None,
        confidence=0.8,
        explanation="No safety issues detected",
        detection_method=DetectionMethod.RULE_BASED,
        metadata={}
    )

async def appropriateness_detector(input_text: str, output_text: str, 
                                 context: Dict[str, Any]) -> ErrorDetectionResult:
    """Detect appropriateness issues based on context"""
    
    # Check for tone mismatch
    is_error = False
    confidence = 0.6
    explanation = "Response tone appears appropriate"
    
    # If input indicates sadness but output is overly cheerful
    sad_indicators = ["sad", "depressed", "upset", "crying", "grief"]
    cheerful_indicators = ["great", "awesome", "fantastic", "wonderful", "amazing"]
    
    input_lower = input_text.lower()
    output_lower = output_text.lower()
    
    has_sad_input = any(indicator in input_lower for indicator in sad_indicators)
    has_cheerful_output = any(indicator in output_lower for indicator in cheerful_indicators)
    
    if has_sad_input and has_cheerful_output:
        is_error = True
        confidence = 0.8
        explanation = "Inappropriate cheerful response to sad input"
    
    return ErrorDetectionResult(
        is_error=is_error,
        error_type=ErrorType.APPROPRIATENESS if is_error else None,
        confidence=confidence,
        explanation=explanation,
        detection_method=DetectionMethod.RULE_BASED,
        metadata={
            "sad_input": has_sad_input,
            "cheerful_output": has_cheerful_output
        }
    )

# Usage example
detector = AutomatedErrorDetector()
detector.register_detector(factual_accuracy_detector)
detector.register_detector(safety_detector)
detector.register_detector(appropriateness_detector)

# Test detection
async def test_detection():
    test_cases = [
        {
            "input": "I'm feeling really sad today",
            "output": "That's fantastic! You should be happy!",
            "context": {"topic": "emotional_support"}
        },
        {
            "input": "What year was the iPhone invented?",
            "output": "The iPhone was invented in 2025",
            "context": {"topic": "technology"}
        }
    ]
    
    for case in test_cases:
        results = await detector.detect_errors(
            case["input"], 
            case["output"], 
            case["context"]
        )
        
        print(f"Input: {case['input']}")
        print(f"Output: {case['output']}")
        for result in results:
            if result.is_error:
                print(f"  ERROR: {result.error_type.value if result.error_type else 'Unknown'} "
                      f"(confidence: {result.confidence:.2f}) - {result.explanation}")
        print()

# Run the test
asyncio.run(test_detection())
```

## Conclusion: Building a Culture of Continuous Improvement

Error analysis in AI systems is not a one-time activity but an ongoing process that requires systematic approaches, appropriate tools, and organizational commitment. The goal is not to eliminate all errors—which is impossible in probabilistic systems—but to understand error patterns, prioritize improvements, and build systems that fail gracefully and learn from their mistakes.

Effective error analysis combines quantitative statistical analysis with qualitative investigation, automated detection with human insight, and immediate problem-solving with long-term systematic improvement. This comprehensive approach enables teams to build AI systems that become more reliable and trustworthy over time.

The investment in systematic error analysis pays dividends throughout the AI system lifecycle. Teams that understand their systems' failure modes can iterate more confidently, scale more reliably, and deliver better user experiences. They build institutional knowledge that enables them to tackle increasingly complex challenges while maintaining high standards for system quality and reliability.

As AI systems become more central to business operations and user experiences, the ability to analyze and learn from errors systematically will become a core competitive advantage. The teams that develop this capability early will be best positioned to build the reliable, trustworthy AI systems that users and businesses demand.

---

**Next**: [Module 2 - Systematic Error Analysis →](../02-error-analysis/README.md)

## References

[1] "Error Analysis for Machine Learning" - Google AI Education, 2024. https://developers.google.com/machine-learning/guides/error-analysis

[2] "Systematic Debugging of AI Systems" - Microsoft Research, 2024. https://www.microsoft.com/en-us/research/publication/ai-system-debugging

[3] "Understanding AI Failures: A Taxonomy and Analysis Framework" - Stanford AI Lab, 2024. https://ai.stanford.edu/blog/ai-failure-taxonomy

[4] "Building Robust AI Systems Through Error Analysis" - Anthropic Research, 2024. https://www.anthropic.com/research/robust-ai-error-analysis

[5] "Production AI Error Monitoring and Analysis" - OpenAI Safety, 2024. https://openai.com/research/production-error-monitoring

