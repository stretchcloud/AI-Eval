# Exercise 2: AMI Lifecycle Implementation

## 🎯 Learning Objectives

By completing this exercise, you will:
- Implement a complete Analyze-Measure-Improve cycle for a real system
- Apply systematic qualitative analysis techniques
- Design and implement quantitative evaluation metrics
- Develop and validate targeted improvements
- Experience the iterative nature of the AMI lifecycle

## 📋 Scenario: Document Summarization System

You're responsible for improving an LLM-powered document summarization system used by a legal firm. The system takes legal documents and produces executive summaries for lawyers. Initial deployment showed promise, but lawyers report inconsistent quality and occasional critical omissions.

**System Overview:**
- Input: Legal documents (contracts, briefs, case law)
- Output: Executive summaries (2-3 paragraphs)
- Current performance: 70% lawyer satisfaction
- Goal: Achieve 90% lawyer satisfaction

## 📊 Part 1: Analyze Phase Implementation (40 minutes)

### Task 1.1: Data Collection and Sampling

Implement a systematic data collection strategy:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import random
from collections import defaultdict, Counter

# TODO: Implement the data collection framework
class DocumentAnalysisCollector:
    def __init__(self):
        self.examples = []
        self.sampling_strategies = []
        
    def collect_representative_sample(self, documents: List[Dict], target_size: int = 100):
        """
        Collect a representative sample using multiple sampling strategies.
        
        Args:
            documents: List of document examples with summaries
            target_size: Target number of examples to collect
        """
        # TODO: Implement stratified sampling by document type
        # TODO: Implement error-focused sampling (lawyer complaints)
        # TODO: Implement edge case sampling (very long/short documents)
        # TODO: Implement temporal sampling (recent vs. older documents)
        
        pass
    
    def analyze_document_characteristics(self, document: Dict) -> Dict:
        """
        Analyze characteristics of a document that might affect summarization.
        
        Args:
            document: Document with content and metadata
            
        Returns:
            Dictionary of document characteristics
        """
        # TODO: Implement analysis of:
        # - Document length and complexity
        # - Legal domain (contract, litigation, regulatory)
        # - Technical terminology density
        # - Structure and formatting
        # - Date and jurisdiction
        
        pass

# Sample data structure for testing
sample_documents = [
    {
        'id': 'doc_001',
        'content': 'This Employment Agreement is entered into between Company X and Employee Y...',
        'summary': 'Employment agreement establishing terms for Employee Y at Company X.',
        'lawyer_feedback': 'Missing key compensation details',
        'satisfaction_score': 3,  # 1-5 scale
        'document_type': 'employment_contract',
        'length': 2500,
        'complexity': 'medium'
    },
    # TODO: Add more sample documents for testing
]

# TODO: Implement and test your data collection
collector = DocumentAnalysisCollector()
# Test your implementation here
```

### Task 1.2: Qualitative Failure Mode Analysis

Implement systematic failure mode identification:

```python
# TODO: Implement failure mode analysis
class FailureModeAnalyzer:
    def __init__(self):
        self.failure_taxonomy = defaultdict(list)
        self.success_patterns = defaultdict(list)
        
    def analyze_example(self, document: Dict, summary: str, feedback: str, score: int):
        """
        Analyze a single example for failure modes and success patterns.
        
        Args:
            document: Original document
            summary: Generated summary
            feedback: Lawyer feedback
            score: Satisfaction score (1-5)
        """
        # TODO: Implement analysis for:
        # - Content omissions (missing key information)
        # - Content hallucinations (information not in original)
        # - Length issues (too long/short)
        # - Clarity problems (unclear language)
        # - Legal accuracy issues (misinterpretation of legal concepts)
        
        pass
    
    def identify_patterns(self) -> Dict:
        """
        Identify patterns across all analyzed examples.
        
        Returns:
            Dictionary with failure patterns and success patterns
        """
        # TODO: Implement pattern identification:
        # - Most common failure modes
        # - Conditions that lead to success
        # - Document characteristics associated with failures
        # - Correlation between failure types and satisfaction scores
        
        pass
    
    def generate_hypotheses(self) -> List[str]:
        """
        Generate improvement hypotheses based on identified patterns.
        
        Returns:
            List of improvement hypotheses
        """
        # TODO: Generate hypotheses like:
        # - "Adding explicit instruction to include compensation details will reduce omission failures"
        # - "Providing examples of good summaries will improve consistency"
        # - "Adding length constraints will address verbosity issues"
        
        pass

# TODO: Implement and test your failure mode analysis
analyzer = FailureModeAnalyzer()
# Test with sample data
```

### Task 1.3: Root Cause Investigation

Implement systematic root cause analysis:

```python
# TODO: Implement root cause analysis
class RootCauseInvestigator:
    def __init__(self):
        self.root_causes = defaultdict(list)
        
    def investigate_failure_mode(self, failure_mode: str, examples: List[Dict]) -> Dict:
        """
        Investigate root causes for a specific failure mode.
        
        Args:
            failure_mode: Type of failure to investigate
            examples: Examples exhibiting this failure mode
            
        Returns:
            Analysis of potential root causes
        """
        # TODO: Implement investigation for:
        # - Prompt design issues
        # - Training data gaps
        # - Model capability limitations
        # - Context window constraints
        # - Domain knowledge requirements
        
        pass
    
    def create_fishbone_analysis(self, failure_mode: str) -> Dict:
        """
        Create a fishbone (cause-and-effect) analysis for a failure mode.
        
        Args:
            failure_mode: Failure mode to analyze
            
        Returns:
            Structured cause-and-effect analysis
        """
        # TODO: Implement fishbone analysis with categories:
        # - People (skills, training, domain knowledge)
        # - Process (prompt design, review process)
        # - Technology (model limitations, infrastructure)
        # - Environment (data quality, context)
        
        pass

# TODO: Implement and test root cause investigation
investigator = RootCauseInvestigator()
```

## 📏 Part 2: Measure Phase Implementation (30 minutes)

### Task 2.1: Metric Design and Implementation

Design metrics based on your analysis findings:

```python
# TODO: Implement evaluation metrics
class SummarizationMetrics:
    def __init__(self):
        self.metrics = {}
        
    def content_completeness_metric(self, document: str, summary: str, key_points: List[str]) -> float:
        """
        Measure how well the summary covers key points from the document.
        
        Args:
            document: Original document text
            summary: Generated summary
            key_points: List of key points that should be covered
            
        Returns:
            Completeness score (0-1)
        """
        # TODO: Implement completeness measurement:
        # - Check if key legal concepts are mentioned
        # - Verify important dates, amounts, parties are included
        # - Assess coverage of main legal obligations
        
        pass
    
    def accuracy_metric(self, document: str, summary: str) -> float:
        """
        Measure accuracy of information in the summary.
        
        Args:
            document: Original document text
            summary: Generated summary
            
        Returns:
            Accuracy score (0-1)
        """
        # TODO: Implement accuracy measurement:
        # - Check for hallucinated information
        # - Verify factual consistency
        # - Validate legal interpretations
        
        pass
    
    def clarity_metric(self, summary: str) -> float:
        """
        Measure clarity and readability of the summary.
        
        Args:
            summary: Generated summary
            
        Returns:
            Clarity score (0-1)
        """
        # TODO: Implement clarity measurement:
        # - Assess sentence structure and flow
        # - Check for legal jargon vs. plain language balance
        # - Evaluate logical organization
        
        pass
    
    def length_appropriateness_metric(self, document: str, summary: str, target_length: int) -> float:
        """
        Measure whether summary length is appropriate.
        
        Args:
            document: Original document text
            summary: Generated summary
            target_length: Target summary length in words
            
        Returns:
            Length appropriateness score (0-1)
        """
        # TODO: Implement length assessment:
        # - Compare to target length
        # - Consider document complexity
        # - Account for information density requirements
        
        pass

# TODO: Implement automated evaluation system
class EvaluationSystem:
    def __init__(self):
        self.metrics = SummarizationMetrics()
        self.results_history = []
        
    def evaluate_batch(self, examples: List[Dict]) -> Dict:
        """
        Evaluate a batch of document-summary pairs.
        
        Args:
            examples: List of examples with documents, summaries, and metadata
            
        Returns:
            Comprehensive evaluation results
        """
        # TODO: Implement batch evaluation:
        # - Apply all metrics to each example
        # - Calculate aggregate statistics
        # - Identify patterns in metric performance
        # - Generate evaluation report
        
        pass
    
    def track_performance_over_time(self) -> Dict:
        """
        Track how performance changes over time.
        
        Returns:
            Performance trends and insights
        """
        # TODO: Implement performance tracking:
        # - Compare current vs. historical performance
        # - Identify improvement or degradation trends
        # - Correlate changes with system modifications
        
        pass

# TODO: Test your evaluation system
eval_system = EvaluationSystem()
```

### Task 2.2: Baseline Establishment

Establish performance baselines:

```python
# TODO: Implement baseline establishment
def establish_baselines(evaluation_system: EvaluationSystem, baseline_data: List[Dict]) -> Dict:
    """
    Establish performance baselines using current system performance.
    
    Args:
        evaluation_system: Configured evaluation system
        baseline_data: Representative dataset for baseline measurement
        
    Returns:
        Baseline performance metrics and thresholds
    """
    # TODO: Implement baseline establishment:
    # - Measure current performance across all metrics
    # - Calculate confidence intervals
    # - Set improvement targets
    # - Define success criteria for improvements
    
    pass

# TODO: Test baseline establishment
```

## 🔧 Part 3: Improve Phase Implementation (30 minutes)

### Task 3.1: Improvement Strategy Development

Develop targeted improvements based on your analysis:

```python
# TODO: Implement improvement strategies
class ImprovementStrategy:
    def __init__(self):
        self.strategies = {}
        
    def prompt_enhancement_strategy(self, failure_modes: List[str]) -> Dict:
        """
        Develop prompt enhancement strategy based on identified failure modes.
        
        Args:
            failure_modes: List of failure modes to address
            
        Returns:
            Prompt enhancement strategy
        """
        # TODO: Implement prompt improvements:
        # - Add specific instructions for key information extraction
        # - Include examples of good summaries
        # - Add constraints for length and format
        # - Specify legal accuracy requirements
        
        pass
    
    def validation_enhancement_strategy(self) -> Dict:
        """
        Develop validation enhancement strategy.
        
        Returns:
            Validation enhancement strategy
        """
        # TODO: Implement validation improvements:
        # - Add completeness checks
        # - Implement accuracy validation
        # - Create length validation
        # - Add legal concept verification
        
        pass
    
    def feedback_integration_strategy(self) -> Dict:
        """
        Develop strategy for integrating lawyer feedback.
        
        Returns:
            Feedback integration strategy
        """
        # TODO: Implement feedback integration:
        # - Create feedback collection system
        # - Implement feedback analysis
        # - Design feedback-driven improvements
        # - Establish feedback loop
        
        pass

# TODO: Implement improvement implementation and validation
class ImprovementImplementor:
    def __init__(self):
        self.implementations = []
        
    def implement_improvement(self, strategy: Dict, current_system: Dict) -> Dict:
        """
        Implement a specific improvement strategy.
        
        Args:
            strategy: Improvement strategy to implement
            current_system: Current system configuration
            
        Returns:
            Updated system configuration
        """
        # TODO: Implement improvement application:
        # - Apply prompt changes
        # - Add validation steps
        # - Integrate feedback mechanisms
        # - Update system configuration
        
        pass
    
    def validate_improvement(self, improved_system: Dict, validation_data: List[Dict]) -> Dict:
        """
        Validate the effectiveness of an improvement.
        
        Args:
            improved_system: System with improvements applied
            validation_data: Data for validation testing
            
        Returns:
            Validation results
        """
        # TODO: Implement improvement validation:
        # - Compare improved vs. baseline performance
        # - Test on held-out validation set
        # - Measure improvement across all metrics
        # - Assess statistical significance
        
        pass

# TODO: Test improvement implementation
implementor = ImprovementImplementor()
```

### Task 3.2: A/B Testing Framework

Implement A/B testing for improvement validation:

```python
# TODO: Implement A/B testing framework
class ABTestingFramework:
    def __init__(self):
        self.experiments = {}
        
    def design_experiment(self, baseline_system: Dict, improved_system: Dict, 
                         test_data: List[Dict]) -> Dict:
        """
        Design an A/B test to compare baseline vs. improved system.
        
        Args:
            baseline_system: Current system configuration
            improved_system: Improved system configuration
            test_data: Data for testing
            
        Returns:
            Experiment design
        """
        # TODO: Implement experiment design:
        # - Define success metrics
        # - Calculate required sample size
        # - Design randomization strategy
        # - Set up statistical testing plan
        
        pass
    
    def run_experiment(self, experiment_design: Dict) -> Dict:
        """
        Run the A/B test experiment.
        
        Args:
            experiment_design: Experiment configuration
            
        Returns:
            Experiment results
        """
        # TODO: Implement experiment execution:
        # - Randomly assign examples to conditions
        # - Collect performance data
        # - Monitor for statistical significance
        # - Ensure proper blinding and controls
        
        pass
    
    def analyze_results(self, experiment_results: Dict) -> Dict:
        """
        Analyze A/B test results and make recommendations.
        
        Args:
            experiment_results: Raw experiment data
            
        Returns:
            Analysis and recommendations
        """
        # TODO: Implement results analysis:
        # - Calculate statistical significance
        # - Measure effect sizes
        # - Assess practical significance
        # - Generate deployment recommendations
        
        pass

# TODO: Test A/B testing framework
ab_tester = ABTestingFramework()
```

## 🔄 Part 4: Lifecycle Integration (20 minutes)

### Task 4.1: Iterative Improvement Loop

Implement the complete AMI cycle:

```python
# TODO: Implement complete AMI lifecycle
class AMILifecycle:
    def __init__(self):
        self.cycle_history = []
        self.current_system = {}
        
    def run_complete_cycle(self, data: List[Dict]) -> Dict:
        """
        Run a complete Analyze-Measure-Improve cycle.
        
        Args:
            data: Data for analysis and evaluation
            
        Returns:
            Cycle results and updated system
        """
        # TODO: Implement complete cycle:
        # 1. Analyze: Collect data, identify failure modes, investigate root causes
        # 2. Measure: Design metrics, establish baselines, evaluate performance
        # 3. Improve: Develop strategies, implement changes, validate improvements
        # 4. Document: Record learnings, update system, plan next cycle
        
        pass
    
    def plan_next_cycle(self, current_results: Dict) -> Dict:
        """
        Plan the next AMI cycle based on current results.
        
        Args:
            current_results: Results from current cycle
            
        Returns:
            Plan for next cycle
        """
        # TODO: Implement next cycle planning:
        # - Identify remaining issues
        # - Prioritize next improvements
        # - Update analysis focus areas
        # - Refine measurement approaches
        
        pass

# TODO: Test complete lifecycle
ami_lifecycle = AMILifecycle()
```

## 🤔 Reflection Questions

After completing the exercise, reflect on these questions:

1. **Analysis Insights**: What were the most surprising insights from your qualitative analysis? How did they differ from your initial assumptions?

2. **Measurement Challenges**: What challenges did you encounter in designing quantitative metrics? How did you balance comprehensiveness with practicality?

3. **Improvement Trade-offs**: What trade-offs did you identify between different improvement strategies? How did you prioritize them?

4. **Lifecycle Integration**: How did insights from each phase inform the others? What would you do differently in the next cycle?

5. **Real-world Application**: How would you adapt this approach for a different domain or application?

## 📝 Deliverables

Submit the following:

1. **Complete code implementation** for all phases of the AMI lifecycle
2. **Analysis report** (1000-1500 words) documenting your findings from each phase
3. **Improvement recommendations** with prioritization and implementation plan
4. **Validation results** from your A/B testing framework
5. **Next cycle plan** based on your results

## 🔍 Self-Assessment Checklist

- [ ] Implemented systematic data collection and sampling
- [ ] Conducted thorough qualitative failure mode analysis
- [ ] Designed and implemented quantitative evaluation metrics
- [ ] Developed targeted improvement strategies
- [ ] Implemented A/B testing for validation
- [ ] Integrated all phases into complete lifecycle
- [ ] Documented insights and learnings
- [ ] Planned next iteration cycle

## 🎯 Extension Activities

For additional challenge:

1. **Multi-metric Optimization**: Implement strategies for optimizing multiple conflicting metrics simultaneously.

2. **Continuous Learning**: Design a system that continuously learns from lawyer feedback and adapts over time.

3. **Domain Adaptation**: Extend your approach to handle multiple legal domains (corporate, litigation, regulatory).

4. **Stakeholder Integration**: Design processes for integrating feedback from multiple stakeholder types (lawyers, paralegals, clients).

---
