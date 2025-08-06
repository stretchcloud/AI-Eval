# Exercise 1: Qualitative Error Analysis

## Objective
Apply open coding and axial coding methodologies to analyze AI system errors, developing deep insights into error patterns, root causes, and improvement opportunities through systematic qualitative research methods.

## Duration
3-4 hours

## Skills Developed
- Qualitative research methods (open coding, axial coding)
- Pattern recognition and systematic analysis
- Error categorization and taxonomy development
- Insight generation from unstructured data
- Mixed-methods integration

## Prerequisites
- Understanding of open coding and axial coding from Section 6
- Basic familiarity with qualitative research principles
- Access to Python environment with required packages

## Learning Outcomes
By completing this exercise, you will be able to:
- Apply systematic open coding to break down error data into meaningful concepts
- Use axial coding to identify relationships and patterns between error categories
- Develop comprehensive error taxonomies based on qualitative analysis
- Generate actionable insights for AI system improvement
- Integrate qualitative findings with quantitative metrics

## Exercise Overview

This exercise guides you through a comprehensive qualitative analysis of AI system errors using real customer support chatbot data. You'll apply the systematic methodologies from Section 6 to uncover deep insights that purely quantitative analysis might miss.

### Scenario
You're analyzing errors from an AI-powered customer support chatbot that handles technical support requests. The system has been experiencing various types of failures, and stakeholders need deep insights into the nature and causes of these errors to guide improvement efforts.

### Dataset
The exercise uses a curated dataset of 200 error cases from a production customer support system, including:
- User queries and system responses
- Error classifications and severity levels
- Context information and metadata
- Human evaluator feedback and corrections

## Part 1: Data Preparation and Initial Exploration

### Step 1: Environment Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import json
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up visualization style
plt.style.use('default')
sns.set_palette("husl")

# Create data structures for qualitative analysis
class QualitativeAnalyzer:
    """
    Framework for systematic qualitative analysis of AI system errors.
    Implements open coding and axial coding methodologies.
    """
    
    def __init__(self):
        self.raw_data = []
        self.open_codes = {}
        self.categories = {}
        self.axial_relationships = {}
        self.themes = {}
        self.analysis_log = []
        
    def load_error_data(self, data_source):
        """Load and prepare error data for qualitative analysis."""
        # In a real scenario, this would load from your data source
        # For this exercise, we'll create representative sample data
        sample_errors = [
            {
                'error_id': 'E001',
                'user_query': 'How do I reset my password?',
                'system_response': 'I can help you troubleshoot network connectivity issues.',
                'error_type': 'intent_misclassification',
                'severity': 'high',
                'context': 'user_authentication',
                'timestamp': '2024-01-15 10:30:00',
                'human_feedback': 'System completely misunderstood the user intent',
                'correction': 'Should have provided password reset instructions'
            },
            {
                'error_id': 'E002',
                'user_query': 'My account is locked, what should I do?',
                'system_response': 'Account lockouts typically resolve within 24 hours.',
                'error_type': 'incomplete_response',
                'severity': 'medium',
                'context': 'account_management',
                'timestamp': '2024-01-15 11:15:00',
                'human_feedback': 'Response is partially correct but lacks actionable steps',
                'correction': 'Should include specific unlock procedures and contact information'
            },
            {
                'error_id': 'E003',
                'user_query': 'Can you help me install the mobile app?',
                'system_response': 'I apologize, but I cannot assist with software installation.',
                'error_type': 'capability_limitation',
                'severity': 'low',
                'context': 'technical_support',
                'timestamp': '2024-01-15 12:00:00',
                'human_feedback': 'System correctly identified limitation but could be more helpful',
                'correction': 'Should provide links to installation guides or alternative resources'
            },
            {
                'error_id': 'E004',
                'user_query': 'Why is my billing amount different this month?',
                'system_response': 'Billing discrepancies can occur due to various factors including usage changes, plan modifications, or promotional credits expiring.',
                'error_type': 'vague_response',
                'severity': 'medium',
                'context': 'billing_inquiry',
                'timestamp': '2024-01-15 13:45:00',
                'human_feedback': 'Response is too generic and doesn\'t help user understand their specific situation',
                'correction': 'Should ask for account details to provide specific billing analysis'
            },
            {
                'error_id': 'E005',
                'user_query': 'I need to cancel my subscription immediately',
                'system_response': 'I understand you want to make changes to your account. Let me help you explore your options.',
                'error_type': 'avoidance_behavior',
                'severity': 'high',
                'context': 'subscription_management',
                'timestamp': '2024-01-15 14:20:00',
                'human_feedback': 'System is avoiding direct response to cancellation request',
                'correction': 'Should directly address cancellation process while offering retention options'
            }
        ]
        
        # Extend with more diverse examples for comprehensive analysis
        additional_errors = self._generate_additional_samples()
        self.raw_data = sample_errors + additional_errors
        
        print(f"Loaded {len(self.raw_data)} error cases for analysis")
        return self.raw_data
    
    def _generate_additional_samples(self):
        """Generate additional sample data for comprehensive analysis."""
        # This would typically load from a real dataset
        # For the exercise, we create diverse representative samples
        return [
            {
                'error_id': f'E{str(i).zfill(3)}',
                'user_query': query,
                'system_response': response,
                'error_type': error_type,
                'severity': severity,
                'context': context,
                'timestamp': f'2024-01-{15 + (i % 10):02d} {10 + (i % 12):02d}:{(i * 7) % 60:02d}:00',
                'human_feedback': feedback,
                'correction': correction
            }
            for i, (query, response, error_type, severity, context, feedback, correction) in enumerate([
                ('How do I update my profile?', 'Profile updates are important for account security.', 'non_actionable_response', 'medium', 'profile_management', 'Response acknowledges importance but provides no instructions', 'Should provide step-by-step profile update instructions'),
                ('What are your business hours?', 'We strive to provide excellent customer service.', 'question_avoidance', 'high', 'general_inquiry', 'Completely avoided answering the direct question', 'Should provide specific business hours and contact information'),
                ('I\'m having trouble logging in', 'Login issues can be frustrating. Have you tried turning it off and on again?', 'generic_troubleshooting', 'medium', 'technical_support', 'Provided generic advice without understanding specific issue', 'Should ask diagnostic questions about specific login failure symptoms'),
                ('Can I get a refund for last month?', 'Refunds are processed according to our terms of service.', 'policy_deflection', 'high', 'billing_inquiry', 'Deflected to policy without addressing specific request', 'Should explain refund eligibility and process for user\'s specific situation'),
                ('The app keeps crashing on my phone', 'App performance can vary across different devices and operating systems.', 'acknowledgment_without_solution', 'medium', 'technical_support', 'Acknowledged issue but provided no troubleshooting steps', 'Should gather device/OS details and provide specific troubleshooting steps')
            ], 6)
        ]

# Initialize analyzer and load data
analyzer = QualitativeAnalyzer()
error_data = analyzer.load_error_data("sample_dataset")

# Display sample data for initial exploration
print("\\nSample Error Cases:")
for i, error in enumerate(error_data[:3]):
    print(f"\\nError {i+1}:")
    print(f"  Query: {error['user_query']}")
    print(f"  Response: {error['system_response']}")
    print(f"  Type: {error['error_type']}")
    print(f"  Feedback: {error['human_feedback']}")
```

### Step 2: Initial Data Exploration

```python
# Create overview of error dataset
def explore_error_dataset(data):
    """Provide comprehensive overview of error dataset."""
    
    df = pd.DataFrame(data)
    
    print("=== DATASET OVERVIEW ===")
    print(f"Total error cases: {len(data)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    print("\\n=== ERROR TYPE DISTRIBUTION ===")
    error_type_counts = df['error_type'].value_counts()
    print(error_type_counts)
    
    print("\\n=== SEVERITY DISTRIBUTION ===")
    severity_counts = df['severity'].value_counts()
    print(severity_counts)
    
    print("\\n=== CONTEXT DISTRIBUTION ===")
    context_counts = df['context'].value_counts()
    print(context_counts)
    
    # Visualize distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Error type distribution
    error_type_counts.plot(kind='bar', ax=axes[0,0], color='skyblue')
    axes[0,0].set_title('Error Type Distribution')
    axes[0,0].set_xlabel('Error Type')
    axes[0,0].set_ylabel('Count')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Severity distribution
    severity_counts.plot(kind='pie', ax=axes[0,1], autopct='%1.1f%%')
    axes[0,1].set_title('Severity Distribution')
    
    # Context distribution
    context_counts.plot(kind='bar', ax=axes[1,0], color='lightcoral')
    axes[1,0].set_title('Context Distribution')
    axes[1,0].set_xlabel('Context')
    axes[1,0].set_ylabel('Count')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Error type vs severity heatmap
    crosstab = pd.crosstab(df['error_type'], df['severity'])
    sns.heatmap(crosstab, annot=True, fmt='d', ax=axes[1,1], cmap='YlOrRd')
    axes[1,1].set_title('Error Type vs Severity')
    
    plt.tight_layout()
    plt.show()
    
    return df

# Explore the dataset
df = explore_error_dataset(error_data)
```

## Part 2: Open Coding Implementation

### Step 3: Systematic Open Coding

```python
class OpenCodingFramework:
    """
    Implementation of systematic open coding for error analysis.
    Breaks down error data into concepts, properties, and dimensions.
    """
    
    def __init__(self):
        self.codes = {}
        self.concepts = {}
        self.properties = {}
        self.dimensions = {}
        self.coding_log = []
        
    def perform_line_by_line_coding(self, text_data, data_source):
        """
        Perform systematic line-by-line open coding.
        
        Args:
            text_data: Text to be coded
            data_source: Source identifier for tracking
        """
        lines = text_data.split('.')
        codes_found = []
        
        for line_num, line in enumerate(lines):
            if line.strip():
                line_codes = self._code_single_line(line.strip(), f"{data_source}_L{line_num}")
                codes_found.extend(line_codes)
        
        return codes_found
    
    def _code_single_line(self, line, line_id):
        """Code a single line of text."""
        codes = []
        
        # Define coding patterns for different types of content
        coding_patterns = {
            'user_intent_indicators': [
                r'\\b(how|what|why|when|where|can|could|should|would)\\b',
                r'\\b(help|assist|support|guide)\\b',
                r'\\b(problem|issue|trouble|error)\\b'
            ],
            'system_response_patterns': [
                r'\\b(I can|I will|I understand|I apologize)\\b',
                r'\\b(typically|usually|generally|often)\\b',
                r'\\b(various|different|multiple|several)\\b'
            ],
            'emotional_indicators': [
                r'\\b(frustrated|confused|urgent|immediately)\\b',
                r'\\b(important|critical|necessary|required)\\b'
            ],
            'action_indicators': [
                r'\\b(reset|cancel|update|install|configure)\\b',
                r'\\b(login|access|connect|download)\\b'
            ],
            'vagueness_indicators': [
                r'\\b(various|different|typically|generally)\\b',
                r'\\b(can occur|may happen|might be|could be)\\b'
            ]
        }
        
        for category, patterns in coding_patterns.items():
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    code = {
                        'code': f"{category}_{len(self.codes)}",
                        'category': category,
                        'text_segment': line,
                        'line_id': line_id,
                        'pattern_matched': pattern
                    }
                    codes.append(code)
                    self.codes[code['code']] = code
        
        return codes
    
    def develop_concepts_from_codes(self):
        """Develop higher-level concepts from individual codes."""
        
        # Group codes by category
        code_categories = defaultdict(list)
        for code in self.codes.values():
            code_categories[code['category']].append(code)
        
        # Develop concepts from code groupings
        for category, codes in code_categories.items():
            concept = {
                'concept_name': category.replace('_', ' ').title(),
                'definition': self._generate_concept_definition(category, codes),
                'properties': self._identify_concept_properties(codes),
                'dimensions': self._identify_concept_dimensions(codes),
                'code_count': len(codes),
                'representative_examples': codes[:3]  # First 3 as examples
            }
            self.concepts[category] = concept
        
        return self.concepts
    
    def _generate_concept_definition(self, category, codes):
        """Generate definition for a concept based on its codes."""
        definitions = {
            'user_intent_indicators': 'Linguistic patterns that reveal user intentions and goals',
            'system_response_patterns': 'Common response structures and phrasings used by the AI system',
            'emotional_indicators': 'Words and phrases that convey user emotional state or urgency',
            'action_indicators': 'Specific actions or tasks that users want to perform',
            'vagueness_indicators': 'Language patterns that indicate non-specific or evasive responses'
        }
        return definitions.get(category, f'Concept derived from {category} patterns')
    
    def _identify_concept_properties(self, codes):
        """Identify properties of a concept."""
        properties = {
            'frequency': len(codes),
            'text_diversity': len(set(code['text_segment'] for code in codes)),
            'pattern_variety': len(set(code['pattern_matched'] for code in codes))
        }
        return properties
    
    def _identify_concept_dimensions(self, codes):
        """Identify dimensions along which a concept varies."""
        dimensions = {
            'intensity': 'Low to High based on language strength',
            'specificity': 'Vague to Specific based on detail level',
            'complexity': 'Simple to Complex based on linguistic structure'
        }
        return dimensions

# Apply open coding to error data
open_coder = OpenCodingFramework()

print("=== PERFORMING OPEN CODING ===")

# Code each error case
all_codes = []
for error in error_data:
    # Code user query
    query_codes = open_coder.perform_line_by_line_coding(
        error['user_query'], f"query_{error['error_id']}"
    )
    
    # Code system response
    response_codes = open_coder.perform_line_by_line_coding(
        error['system_response'], f"response_{error['error_id']}"
    )
    
    # Code human feedback
    feedback_codes = open_coder.perform_line_by_line_coding(
        error['human_feedback'], f"feedback_{error['error_id']}"
    )
    
    all_codes.extend(query_codes + response_codes + feedback_codes)

print(f"Generated {len(open_coder.codes)} individual codes")

# Develop concepts from codes
concepts = open_coder.develop_concepts_from_codes()

print(f"\\nDeveloped {len(concepts)} concepts:")
for concept_name, concept_data in concepts.items():
    print(f"\\n{concept_data['concept_name']}:")
    print(f"  Definition: {concept_data['definition']}")
    print(f"  Code count: {concept_data['code_count']}")
    print(f"  Properties: {concept_data['properties']}")
```

### Step 4: Concept Development and Refinement

```python
def refine_concepts_through_constant_comparison(open_coder, error_data):
    """
    Refine concepts through constant comparison method.
    Compare new data against existing concepts to refine and validate.
    """
    
    print("=== CONCEPT REFINEMENT THROUGH CONSTANT COMPARISON ===")
    
    refined_concepts = {}
    
    for concept_name, concept in open_coder.concepts.items():
        print(f"\\nRefining concept: {concept['concept_name']}")
        
        # Analyze concept across different error contexts
        concept_analysis = {
            'cross_context_validation': {},
            'property_refinement': {},
            'dimensional_analysis': {},
            'saturation_assessment': {}
        }
        
        # Cross-context validation
        contexts = set(error['context'] for error in error_data)
        for context in contexts:
            context_errors = [e for e in error_data if e['context'] == context]
            context_codes = []
            
            for error in context_errors:
                error_codes = open_coder.perform_line_by_line_coding(
                    f"{error['user_query']} {error['system_response']}", 
                    f"{context}_{error['error_id']}"
                )
                context_codes.extend([c for c in error_codes if c['category'] == concept_name])
            
            concept_analysis['cross_context_validation'][context] = {
                'code_count': len(context_codes),
                'unique_patterns': len(set(c['pattern_matched'] for c in context_codes)),
                'consistency_score': len(context_codes) / max(1, len(context_errors))
            }
        
        # Property refinement based on cross-context analysis
        concept_analysis['property_refinement'] = {
            'context_consistency': np.mean([
                data['consistency_score'] 
                for data in concept_analysis['cross_context_validation'].values()
            ]),
            'pattern_diversity': np.mean([
                data['unique_patterns'] 
                for data in concept_analysis['cross_context_validation'].values()
            ]),
            'contextual_relevance': len([
                ctx for ctx, data in concept_analysis['cross_context_validation'].items()
                if data['code_count'] > 0
            ]) / len(contexts)
        }
        
        # Dimensional analysis
        concept_analysis['dimensional_analysis'] = analyze_concept_dimensions(
            concept_name, concept_analysis['cross_context_validation']
        )
        
        # Saturation assessment
        concept_analysis['saturation_assessment'] = assess_concept_saturation(
            concept_name, open_coder.codes
        )
        
        refined_concepts[concept_name] = {
            'original_concept': concept,
            'refinement_analysis': concept_analysis,
            'refined_definition': refine_concept_definition(concept, concept_analysis),
            'validated_properties': validate_concept_properties(concept, concept_analysis),
            'confirmed_dimensions': confirm_concept_dimensions(concept, concept_analysis)
        }
        
        print(f"  Refined definition: {refined_concepts[concept_name]['refined_definition']}")
        print(f"  Context consistency: {concept_analysis['property_refinement']['context_consistency']:.2f}")
        print(f"  Contextual relevance: {concept_analysis['property_refinement']['contextual_relevance']:.2f}")
    
    return refined_concepts

def analyze_concept_dimensions(concept_name, cross_context_data):
    """Analyze how concept varies across dimensions."""
    
    dimensions = {
        'frequency_dimension': {
            'low': [ctx for ctx, data in cross_context_data.items() if data['code_count'] <= 1],
            'medium': [ctx for ctx, data in cross_context_data.items() if 1 < data['code_count'] <= 3],
            'high': [ctx for ctx, data in cross_context_data.items() if data['code_count'] > 3]
        },
        'diversity_dimension': {
            'low': [ctx for ctx, data in cross_context_data.items() if data['unique_patterns'] <= 1],
            'medium': [ctx for ctx, data in cross_context_data.items() if 1 < data['unique_patterns'] <= 2],
            'high': [ctx for ctx, data in cross_context_data.items() if data['unique_patterns'] > 2]
        },
        'consistency_dimension': {
            'low': [ctx for ctx, data in cross_context_data.items() if data['consistency_score'] <= 0.3],
            'medium': [ctx for ctx, data in cross_context_data.items() if 0.3 < data['consistency_score'] <= 0.7],
            'high': [ctx for ctx, data in cross_context_data.items() if data['consistency_score'] > 0.7]
        }
    }
    
    return dimensions

def assess_concept_saturation(concept_name, all_codes):
    """Assess whether concept has reached theoretical saturation."""
    
    concept_codes = [code for code in all_codes.values() if code['category'] == concept_name]
    
    # Analyze pattern emergence over time
    pattern_emergence = {}
    unique_patterns = set()
    
    for i, code in enumerate(concept_codes):
        pattern = code['pattern_matched']
        if pattern not in unique_patterns:
            pattern_emergence[i] = pattern
            unique_patterns.add(pattern)
    
    # Calculate saturation indicators
    saturation_score = 1 - (len(pattern_emergence) / max(1, len(concept_codes)))
    
    return {
        'saturation_score': saturation_score,
        'unique_patterns_found': len(unique_patterns),
        'total_codes': len(concept_codes),
        'pattern_emergence_rate': len(pattern_emergence) / max(1, len(concept_codes)),
        'is_saturated': saturation_score > 0.8
    }

def refine_concept_definition(original_concept, analysis):
    """Refine concept definition based on analysis."""
    
    base_definition = original_concept['definition']
    consistency = analysis['property_refinement']['context_consistency']
    relevance = analysis['property_refinement']['contextual_relevance']
    
    if consistency > 0.7 and relevance > 0.7:
        return f"{base_definition} (Highly consistent across contexts)"
    elif consistency > 0.5:
        return f"{base_definition} (Moderately consistent with contextual variation)"
    else:
        return f"{base_definition} (Context-dependent with significant variation)"

def validate_concept_properties(original_concept, analysis):
    """Validate and refine concept properties."""
    
    validated_properties = original_concept['properties'].copy()
    validated_properties.update({
        'cross_context_consistency': analysis['property_refinement']['context_consistency'],
        'contextual_relevance': analysis['property_refinement']['contextual_relevance'],
        'pattern_diversity': analysis['property_refinement']['pattern_diversity'],
        'saturation_level': analysis['saturation_assessment']['saturation_score']
    })
    
    return validated_properties

def confirm_concept_dimensions(original_concept, analysis):
    """Confirm and refine concept dimensions."""
    
    confirmed_dimensions = original_concept['dimensions'].copy()
    
    # Add empirically derived dimensions
    dimensional_analysis = analysis['dimensional_analysis']
    
    confirmed_dimensions.update({
        'contextual_frequency': f"Varies from {len(dimensional_analysis['frequency_dimension']['low'])} low-frequency contexts to {len(dimensional_analysis['frequency_dimension']['high'])} high-frequency contexts",
        'pattern_diversity': f"Shows {len(dimensional_analysis['diversity_dimension']['high'])} high-diversity contexts vs {len(dimensional_analysis['diversity_dimension']['low'])} low-diversity contexts",
        'contextual_consistency': f"Demonstrates {len(dimensional_analysis['consistency_dimension']['high'])} highly consistent contexts"
    })
    
    return confirmed_dimensions

# Apply concept refinement
refined_concepts = refine_concepts_through_constant_comparison(open_coder, error_data)

# Display refined concepts
print("\\n=== REFINED CONCEPTS SUMMARY ===")
for concept_name, refined_data in refined_concepts.items():
    print(f"\\n{concept_name.upper()}:")
    print(f"  Refined Definition: {refined_data['refined_definition']}")
    print(f"  Saturation Level: {refined_data['refinement_analysis']['saturation_assessment']['saturation_score']:.2f}")
    print(f"  Cross-Context Relevance: {refined_data['refinement_analysis']['property_refinement']['contextual_relevance']:.2f}")
```

## Part 3: Axial Coding Implementation

### Step 5: Relationship Identification and Mapping

```python
class AxialCodingFramework:
    """
    Implementation of axial coding for identifying relationships between concepts.
    Builds paradigm models and causal relationships.
    """
    
    def __init__(self, refined_concepts):
        self.concepts = refined_concepts
        self.relationships = {}
        self.paradigm_models = {}
        self.causal_networks = {}
        
    def identify_concept_relationships(self):
        """Identify relationships between concepts using systematic analysis."""
        
        print("=== IDENTIFYING CONCEPT RELATIONSHIPS ===")
        
        concept_names = list(self.concepts.keys())
        
        for i, concept_a in enumerate(concept_names):
            for j, concept_b in enumerate(concept_names[i+1:], i+1):
                relationship = self._analyze_concept_relationship(concept_a, concept_b)
                if relationship['strength'] > 0.3:  # Threshold for meaningful relationships
                    relationship_key = f"{concept_a}_{concept_b}"
                    self.relationships[relationship_key] = relationship
                    
                    print(f"\\nRelationship found: {concept_a} <-> {concept_b}")
                    print(f"  Type: {relationship['type']}")
                    print(f"  Strength: {relationship['strength']:.2f}")
                    print(f"  Description: {relationship['description']}")
        
        return self.relationships
    
    def _analyze_concept_relationship(self, concept_a, concept_b):
        """Analyze relationship between two concepts."""
        
        # Get concept data
        data_a = self.concepts[concept_a]
        data_b = self.concepts[concept_b]
        
        # Analyze different types of relationships
        relationship_analysis = {
            'causal_relationship': self._analyze_causal_relationship(data_a, data_b),
            'conditional_relationship': self._analyze_conditional_relationship(data_a, data_b),
            'associative_relationship': self._analyze_associative_relationship(data_a, data_b),
            'temporal_relationship': self._analyze_temporal_relationship(data_a, data_b)
        }
        
        # Determine strongest relationship type
        strongest_type = max(relationship_analysis.keys(), 
                           key=lambda k: relationship_analysis[k]['strength'])
        
        strongest_relationship = relationship_analysis[strongest_type]
        
        return {
            'concept_a': concept_a,
            'concept_b': concept_b,
            'type': strongest_type,
            'strength': strongest_relationship['strength'],
            'description': strongest_relationship['description'],
            'evidence': strongest_relationship['evidence'],
            'all_analyses': relationship_analysis
        }
    
    def _analyze_causal_relationship(self, data_a, data_b):
        """Analyze potential causal relationship between concepts."""
        
        # Look for causal indicators in cross-context validation
        context_a = data_a['refinement_analysis']['cross_context_validation']
        context_b = data_b['refinement_analysis']['cross_context_validation']
        
        # Calculate causal strength based on co-occurrence patterns
        common_contexts = set(context_a.keys()) & set(context_b.keys())
        
        if not common_contexts:
            return {'strength': 0.0, 'description': 'No common contexts', 'evidence': []}
        
        causal_evidence = []
        causal_strength = 0.0
        
        for context in common_contexts:
            freq_a = context_a[context]['code_count']
            freq_b = context_b[context]['code_count']
            
            # Simple causal heuristic: if A is frequent and B follows, suggest causation
            if freq_a > 0 and freq_b > 0:
                causal_strength += min(freq_a, freq_b) / max(freq_a, freq_b)
                causal_evidence.append(f"Co-occurrence in {context}: A={freq_a}, B={freq_b}")
        
        causal_strength = causal_strength / len(common_contexts) if common_contexts else 0.0
        
        return {
            'strength': causal_strength,
            'description': f"Potential causal relationship with {causal_strength:.2f} strength",
            'evidence': causal_evidence
        }
    
    def _analyze_conditional_relationship(self, data_a, data_b):
        """Analyze conditional relationship (if-then patterns)."""
        
        # Analyze consistency patterns that might indicate conditional relationships
        consistency_a = data_a['refinement_analysis']['property_refinement']['context_consistency']
        consistency_b = data_b['refinement_analysis']['property_refinement']['context_consistency']
        
        # If one concept is highly consistent and another varies, might be conditional
        consistency_difference = abs(consistency_a - consistency_b)
        
        conditional_strength = consistency_difference * min(consistency_a, consistency_b)
        
        return {
            'strength': conditional_strength,
            'description': f"Conditional relationship based on consistency patterns",
            'evidence': [f"Consistency A: {consistency_a:.2f}, B: {consistency_b:.2f}"]
        }
    
    def _analyze_associative_relationship(self, data_a, data_b):
        """Analyze associative relationship (co-occurrence)."""
        
        # Calculate association based on contextual co-occurrence
        context_a = set(data_a['refinement_analysis']['cross_context_validation'].keys())
        context_b = set(data_b['refinement_analysis']['cross_context_validation'].keys())
        
        intersection = context_a & context_b
        union = context_a | context_b
        
        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        
        return {
            'strength': jaccard_similarity,
            'description': f"Associative relationship with {jaccard_similarity:.2f} context overlap",
            'evidence': [f"Common contexts: {list(intersection)}"]
        }
    
    def _analyze_temporal_relationship(self, data_a, data_b):
        """Analyze temporal relationship patterns."""
        
        # For this exercise, we'll use a simplified temporal analysis
        # In practice, this would analyze timestamp patterns
        
        # Use saturation levels as proxy for temporal emergence
        saturation_a = data_a['refinement_analysis']['saturation_assessment']['saturation_score']
        saturation_b = data_b['refinement_analysis']['saturation_assessment']['saturation_score']
        
        temporal_strength = abs(saturation_a - saturation_b) * 0.5  # Simplified metric
        
        return {
            'strength': temporal_strength,
            'description': f"Temporal relationship based on emergence patterns",
            'evidence': [f"Saturation A: {saturation_a:.2f}, B: {saturation_b:.2f}"]
        }
    
    def build_paradigm_models(self):
        """Build paradigm models showing causal conditions, phenomena, and consequences."""
        
        print("\\n=== BUILDING PARADIGM MODELS ===")
        
        # Identify central phenomena (concepts with highest relationship connectivity)
        concept_connectivity = defaultdict(int)
        for relationship in self.relationships.values():
            concept_connectivity[relationship['concept_a']] += 1
            concept_connectivity[relationship['concept_b']] += 1
        
        # Build paradigm model for each central phenomenon
        central_phenomena = sorted(concept_connectivity.items(), 
                                 key=lambda x: x[1], reverse=True)[:3]
        
        for phenomenon, connectivity in central_phenomena:
            paradigm_model = self._build_single_paradigm_model(phenomenon)
            self.paradigm_models[phenomenon] = paradigm_model
            
            print(f"\\nParadigm Model for: {phenomenon}")
            print(f"  Causal Conditions: {paradigm_model['causal_conditions']}")
            print(f"  Context: {paradigm_model['context']}")
            print(f"  Intervening Conditions: {paradigm_model['intervening_conditions']}")
            print(f"  Action/Interaction Strategies: {paradigm_model['action_strategies']}")
            print(f"  Consequences: {paradigm_model['consequences']}")
        
        return self.paradigm_models
    
    def _build_single_paradigm_model(self, central_phenomenon):
        """Build paradigm model for a single central phenomenon."""
        
        # Find all relationships involving this phenomenon
        related_relationships = [
            rel for rel in self.relationships.values()
            if rel['concept_a'] == central_phenomenon or rel['concept_b'] == central_phenomenon
        ]
        
        # Categorize related concepts based on relationship types
        paradigm_model = {
            'central_phenomenon': central_phenomenon,
            'causal_conditions': [],
            'context': [],
            'intervening_conditions': [],
            'action_strategies': [],
            'consequences': []
        }
        
        for relationship in related_relationships:
            other_concept = (relationship['concept_b'] if relationship['concept_a'] == central_phenomenon 
                           else relationship['concept_a'])
            
            # Categorize based on relationship type and strength
            if relationship['type'] == 'causal_relationship' and relationship['strength'] > 0.5:
                paradigm_model['causal_conditions'].append(other_concept)
            elif relationship['type'] == 'conditional_relationship':
                paradigm_model['intervening_conditions'].append(other_concept)
            elif relationship['type'] == 'associative_relationship':
                paradigm_model['context'].append(other_concept)
            elif relationship['type'] == 'temporal_relationship':
                paradigm_model['consequences'].append(other_concept)
        
        # Add action strategies based on concept analysis
        paradigm_model['action_strategies'] = self._identify_action_strategies(
            central_phenomenon, paradigm_model
        )
        
        return paradigm_model
    
    def _identify_action_strategies(self, phenomenon, model):
        """Identify action/interaction strategies for paradigm model."""
        
        # Based on the phenomenon type, suggest relevant action strategies
        strategy_mapping = {
            'user_intent_indicators': ['Intent clarification', 'Query reformulation', 'Context gathering'],
            'system_response_patterns': ['Response optimization', 'Template refinement', 'Personalization'],
            'emotional_indicators': ['Empathy enhancement', 'Urgency handling', 'Escalation protocols'],
            'action_indicators': ['Task decomposition', 'Step-by-step guidance', 'Resource provision'],
            'vagueness_indicators': ['Specificity improvement', 'Context enrichment', 'Clarification requests']
        }
        
        return strategy_mapping.get(phenomenon, ['General improvement strategies'])

# Apply axial coding
axial_coder = AxialCodingFramework(refined_concepts)
relationships = axial_coder.identify_concept_relationships()
paradigm_models = axial_coder.build_paradigm_models()

# Visualize relationship network
def visualize_concept_relationships(relationships, concepts):
    """Create network visualization of concept relationships."""
    
    import networkx as nx
    
    # Create network graph
    G = nx.Graph()
    
    # Add nodes (concepts)
    for concept in concepts.keys():
        G.add_node(concept)
    
    # Add edges (relationships)
    for relationship in relationships.values():
        G.add_edge(
            relationship['concept_a'], 
            relationship['concept_b'],
            weight=relationship['strength'],
            type=relationship['type']
        )
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=1000, alpha=0.7)
    
    # Draw edges with different styles for different relationship types
    edge_colors = {
        'causal_relationship': 'red',
        'conditional_relationship': 'blue',
        'associative_relationship': 'green',
        'temporal_relationship': 'orange'
    }
    
    for relationship in relationships.values():
        nx.draw_networkx_edges(G, pos, 
                              [(relationship['concept_a'], relationship['concept_b'])],
                              edge_color=edge_colors.get(relationship['type'], 'gray'),
                              width=relationship['strength'] * 3,
                              alpha=0.6)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title('Concept Relationship Network')
    plt.axis('off')
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], color=color, lw=2, label=rel_type.replace('_', ' ').title())
                      for rel_type, color in edge_colors.items()]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()

# Visualize the relationships
visualize_concept_relationships(relationships, refined_concepts)
```

## Part 4: Thematic Analysis and Insight Generation

### Step 6: Theme Development and Validation

```python
class ThematicAnalysisFramework:
    """
    Framework for developing themes from axial coding results.
    Generates actionable insights for AI system improvement.
    """
    
    def __init__(self, paradigm_models, relationships, concepts):
        self.paradigm_models = paradigm_models
        self.relationships = relationships
        self.concepts = concepts
        self.themes = {}
        self.insights = {}
        
    def develop_themes_from_paradigms(self):
        """Develop overarching themes from paradigm models."""
        
        print("=== DEVELOPING THEMES FROM PARADIGM ANALYSIS ===")
        
        # Analyze patterns across paradigm models
        cross_paradigm_patterns = self._identify_cross_paradigm_patterns()
        
        # Develop themes based on patterns
        themes = {
            'communication_breakdown_theme': self._develop_communication_theme(cross_paradigm_patterns),
            'system_capability_theme': self._develop_capability_theme(cross_paradigm_patterns),
            'user_experience_theme': self._develop_experience_theme(cross_paradigm_patterns),
            'response_quality_theme': self._develop_quality_theme(cross_paradigm_patterns)
        }
        
        # Validate themes through cross-checking
        validated_themes = self._validate_themes(themes)
        
        self.themes = validated_themes
        
        for theme_name, theme_data in validated_themes.items():
            print(f"\\n{theme_name.upper()}:")
            print(f"  Description: {theme_data['description']}")
            print(f"  Supporting Evidence: {len(theme_data['evidence'])} data points")
            print(f"  Validation Score: {theme_data['validation_score']:.2f}")
        
        return validated_themes
    
    def _identify_cross_paradigm_patterns(self):
        """Identify patterns that appear across multiple paradigm models."""
        
        patterns = {
            'common_causal_conditions': defaultdict(int),
            'common_consequences': defaultdict(int),
            'common_action_strategies': defaultdict(int),
            'recurring_contexts': defaultdict(int)
        }
        
        for paradigm in self.paradigm_models.values():
            for condition in paradigm['causal_conditions']:
                patterns['common_causal_conditions'][condition] += 1
            
            for consequence in paradigm['consequences']:
                patterns['common_consequences'][consequence] += 1
            
            for strategy in paradigm['action_strategies']:
                patterns['common_action_strategies'][strategy] += 1
            
            for context in paradigm['context']:
                patterns['recurring_contexts'][context] += 1
        
        return patterns
    
    def _develop_communication_theme(self, patterns):
        """Develop theme related to communication breakdown."""
        
        communication_indicators = [
            'user_intent_indicators', 'vagueness_indicators', 'system_response_patterns'
        ]
        
        relevant_patterns = {
            key: count for key, count in patterns['common_causal_conditions'].items()
            if any(indicator in key for indicator in communication_indicators)
        }
        
        theme = {
            'name': 'Communication Breakdown and Misalignment',
            'description': 'Systematic patterns of miscommunication between users and AI system, characterized by intent misinterpretation and vague responses',
            'core_components': [
                'Intent recognition failures',
                'Response vagueness and evasion',
                'Context misunderstanding',
                'Feedback loop disruption'
            ],
            'evidence': relevant_patterns,
            'manifestations': [
                'Users express clear intent but receive irrelevant responses',
                'System provides generic responses instead of specific solutions',
                'Lack of clarification when user intent is ambiguous',
                'Failure to acknowledge limitations appropriately'
            ],
            'implications': [
                'User frustration and task abandonment',
                'Reduced trust in AI system capabilities',
                'Increased support ticket volume',
                'Negative impact on user experience metrics'
            ]
        }
        
        return theme
    
    def _develop_capability_theme(self, patterns):
        """Develop theme related to system capability limitations."""
        
        capability_indicators = ['action_indicators', 'system_response_patterns']
        
        relevant_patterns = {
            key: count for key, count in patterns['common_consequences'].items()
            if any(indicator in key for indicator in capability_indicators)
        }
        
        theme = {
            'name': 'System Capability Boundaries and Limitations',
            'description': 'Patterns revealing the boundaries of AI system capabilities and how these limitations are communicated to users',
            'core_components': [
                'Task complexity thresholds',
                'Knowledge domain boundaries',
                'Interaction capability limits',
                'Escalation trigger points'
            ],
            'evidence': relevant_patterns,
            'manifestations': [
                'System avoids direct responses to complex requests',
                'Generic responses when specific expertise is needed',
                'Failure to escalate appropriately',
                'Inconsistent capability demonstration across contexts'
            ],
            'implications': [
                'Unclear user expectations about system capabilities',
                'Missed opportunities for appropriate task completion',
                'Inefficient resource utilization',
                'Need for better capability communication strategies'
            ]
        }
        
        return theme
    
    def _develop_experience_theme(self, patterns):
        """Develop theme related to user experience quality."""
        
        experience_indicators = ['emotional_indicators', 'user_intent_indicators']
        
        relevant_patterns = {
            key: count for key, count in patterns['recurring_contexts'].items()
            if any(indicator in key for indicator in experience_indicators)
        }
        
        theme = {
            'name': 'User Experience Quality and Satisfaction',
            'description': 'Patterns affecting overall user experience quality, including emotional responses and satisfaction indicators',
            'core_components': [
                'Emotional state recognition',
                'Urgency handling',
                'Personalization effectiveness',
                'Task completion satisfaction'
            ],
            'evidence': relevant_patterns,
            'manifestations': [
                'System fails to recognize user urgency or frustration',
                'Responses lack empathy or appropriate tone',
                'Generic interactions without personalization',
                'Incomplete task resolution leading to user dissatisfaction'
            ],
            'implications': [
                'Reduced user engagement and retention',
                'Negative brand perception',
                'Increased customer service costs',
                'Competitive disadvantage in user experience'
            ]
        }
        
        return theme
    
    def _develop_quality_theme(self, patterns):
        """Develop theme related to response quality."""
        
        quality_indicators = ['vagueness_indicators', 'system_response_patterns']
        
        relevant_patterns = {
            key: count for key, count in patterns['common_action_strategies'].items()
            if any(indicator in key for indicator in quality_indicators)
        }
        
        theme = {
            'name': 'Response Quality and Specificity',
            'description': 'Patterns related to the quality, specificity, and usefulness of AI system responses',
            'core_components': [
                'Response specificity levels',
                'Actionability of provided information',
                'Accuracy and relevance',
                'Completeness of responses'
            ],
            'evidence': relevant_patterns,
            'manifestations': [
                'Vague responses that don\'t provide actionable guidance',
                'Generic information instead of specific solutions',
                'Incomplete responses that require follow-up',
                'Responses that acknowledge problems without offering solutions'
            ],
            'implications': [
                'Users need multiple interactions to complete tasks',
                'Reduced efficiency and increased interaction costs',
                'User frustration with unhelpful responses',
                'Opportunity for response quality improvement'
            ]
        }
        
        return theme
    
    def _validate_themes(self, themes):
        """Validate themes through cross-checking and evidence assessment."""
        
        validated_themes = {}
        
        for theme_name, theme_data in themes.items():
            validation_score = self._calculate_theme_validation_score(theme_data)
            
            validated_theme = theme_data.copy()
            validated_theme['validation_score'] = validation_score
            validated_theme['validation_criteria'] = {
                'evidence_strength': len(theme_data['evidence']) / 10.0,  # Normalize to 0-1
                'cross_paradigm_support': self._check_cross_paradigm_support(theme_data),
                'conceptual_coherence': self._assess_conceptual_coherence(theme_data),
                'practical_relevance': self._assess_practical_relevance(theme_data)
            }
            
            validated_themes[theme_name] = validated_theme
        
        return validated_themes
    
    def _calculate_theme_validation_score(self, theme_data):
        """Calculate validation score for a theme."""
        
        evidence_score = min(1.0, len(theme_data['evidence']) / 5.0)
        manifestation_score = min(1.0, len(theme_data['manifestations']) / 4.0)
        implication_score = min(1.0, len(theme_data['implications']) / 3.0)
        
        return (evidence_score + manifestation_score + implication_score) / 3.0
    
    def _check_cross_paradigm_support(self, theme_data):
        """Check if theme is supported across multiple paradigms."""
        
        # Count how many paradigm models support this theme
        supporting_paradigms = 0
        for paradigm in self.paradigm_models.values():
            paradigm_elements = (paradigm['causal_conditions'] + 
                               paradigm['consequences'] + 
                               paradigm['action_strategies'] + 
                               paradigm['context'])
            
            theme_elements = list(theme_data['evidence'].keys())
            
            if any(element in paradigm_elements for element in theme_elements):
                supporting_paradigms += 1
        
        return supporting_paradigms / len(self.paradigm_models) if self.paradigm_models else 0
    
    def _assess_conceptual_coherence(self, theme_data):
        """Assess conceptual coherence of theme."""
        
        # Simple coherence assessment based on component relationships
        components = theme_data['core_components']
        manifestations = theme_data['manifestations']
        
        # Check if manifestations align with components
        alignment_score = 0
        for component in components:
            component_words = set(component.lower().split())
            for manifestation in manifestations:
                manifestation_words = set(manifestation.lower().split())
                if component_words & manifestation_words:
                    alignment_score += 1
                    break
        
        return alignment_score / len(components) if components else 0
    
    def _assess_practical_relevance(self, theme_data):
        """Assess practical relevance of theme for improvement."""
        
        # Themes with clear implications and actionable components are more relevant
        implications = theme_data['implications']
        actionable_implications = [
            imp for imp in implications 
            if any(word in imp.lower() for word in ['improve', 'reduce', 'increase', 'enhance', 'optimize'])
        ]
        
        return len(actionable_implications) / len(implications) if implications else 0
    
    def generate_actionable_insights(self):
        """Generate actionable insights from validated themes."""
        
        print("\\n=== GENERATING ACTIONABLE INSIGHTS ===")
        
        insights = {}
        
        for theme_name, theme_data in self.themes.items():
            theme_insights = self._generate_theme_insights(theme_name, theme_data)
            insights[theme_name] = theme_insights
            
            print(f"\\n{theme_name.upper()} INSIGHTS:")
            for insight_type, insight_list in theme_insights.items():
                print(f"  {insight_type.replace('_', ' ').title()}:")
                for insight in insight_list:
                    print(f"    - {insight}")
        
        self.insights = insights
        return insights
    
    def _generate_theme_insights(self, theme_name, theme_data):
        """Generate insights for a specific theme."""
        
        insights = {
            'immediate_actions': [],
            'system_improvements': [],
            'process_changes': [],
            'measurement_strategies': []
        }
        
        # Generate insights based on theme implications
        for implication in theme_data['implications']:
            if 'frustration' in implication.lower():
                insights['immediate_actions'].append('Implement user frustration detection and response protocols')
                insights['measurement_strategies'].append('Track user frustration indicators in conversations')
            
            elif 'trust' in implication.lower():
                insights['system_improvements'].append('Enhance transparency in system capabilities and limitations')
                insights['measurement_strategies'].append('Monitor trust-related feedback and sentiment')
            
            elif 'cost' in implication.lower():
                insights['process_changes'].append('Optimize interaction efficiency to reduce operational costs')
                insights['measurement_strategies'].append('Track interaction completion rates and efficiency metrics')
            
            elif 'experience' in implication.lower():
                insights['system_improvements'].append('Improve personalization and context awareness')
                insights['measurement_strategies'].append('Implement comprehensive user experience tracking')
        
        # Generate insights based on theme manifestations
        for manifestation in theme_data['manifestations']:
            if 'vague' in manifestation.lower() or 'generic' in manifestation.lower():
                insights['system_improvements'].append('Develop more specific and contextual response templates')
                insights['immediate_actions'].append('Review and refine response generation prompts')
            
            elif 'intent' in manifestation.lower():
                insights['system_improvements'].append('Enhance intent recognition and classification capabilities')
                insights['process_changes'].append('Implement intent clarification workflows')
            
            elif 'escalation' in manifestation.lower():
                insights['process_changes'].append('Define clear escalation criteria and workflows')
                insights['immediate_actions'].append('Train system on appropriate escalation triggers')
        
        return insights

# Apply thematic analysis
thematic_analyzer = ThematicAnalysisFramework(paradigm_models, relationships, refined_concepts)
themes = thematic_analyzer.develop_themes_from_paradigms()
insights = thematic_analyzer.generate_actionable_insights()
```

## Part 5: Integration and Reporting

### Step 7: Comprehensive Analysis Report

```python
def generate_comprehensive_analysis_report(open_coder, refined_concepts, relationships, 
                                         paradigm_models, themes, insights):
    """Generate comprehensive qualitative analysis report."""
    
    report = {
        'executive_summary': {},
        'methodology_overview': {},
        'findings_summary': {},
        'detailed_analysis': {},
        'recommendations': {},
        'implementation_roadmap': {}
    }
    
    # Executive Summary
    report['executive_summary'] = {
        'analysis_scope': f"Qualitative analysis of {len(open_coder.raw_data)} AI system error cases",
        'key_findings': [
            f"Identified {len(open_coder.concepts)} core concepts through systematic open coding",
            f"Discovered {len(relationships)} significant relationships between concepts",
            f"Developed {len(paradigm_models)} paradigm models explaining error phenomena",
            f"Generated {len(themes)} overarching themes with actionable insights"
        ],
        'primary_themes': list(themes.keys()),
        'validation_scores': {theme: data['validation_score'] for theme, data in themes.items()},
        'business_impact': "Analysis reveals systematic patterns affecting user experience, operational efficiency, and system reliability"
    }
    
    # Methodology Overview
    report['methodology_overview'] = {
        'approach': 'Systematic qualitative analysis using grounded theory methodology',
        'phases': [
            'Open coding: Breaking down data into concepts and categories',
            'Axial coding: Identifying relationships and building paradigm models',
            'Thematic analysis: Developing overarching themes and insights'
        ],
        'validation_methods': [
            'Constant comparison across contexts',
            'Cross-paradigm pattern validation',
            'Evidence-based theme development',
            'Multi-criteria validation scoring'
        ],
        'quality_assurance': 'Systematic validation at each phase with quantitative validation metrics'
    }
    
    # Findings Summary
    report['findings_summary'] = {
        'concept_development': {
            'total_concepts': len(refined_concepts),
            'saturated_concepts': len([c for c in refined_concepts.values() 
                                     if c['refinement_analysis']['saturation_assessment']['is_saturated']]),
            'cross_context_validated': len([c for c in refined_concepts.values() 
                                          if c['refinement_analysis']['property_refinement']['contextual_relevance'] > 0.7])
        },
        'relationship_analysis': {
            'total_relationships': len(relationships),
            'strong_relationships': len([r for r in relationships.values() if r['strength'] > 0.7]),
            'relationship_types': Counter([r['type'] for r in relationships.values()])
        },
        'paradigm_models': {
            'models_developed': len(paradigm_models),
            'central_phenomena': list(paradigm_models.keys()),
            'action_strategies_identified': sum(len(p['action_strategies']) for p in paradigm_models.values())
        },
        'thematic_analysis': {
            'themes_developed': len(themes),
            'average_validation_score': np.mean([t['validation_score'] for t in themes.values()]),
            'high_confidence_themes': len([t for t in themes.values() if t['validation_score'] > 0.7])
        }
    }
    
    # Detailed Analysis
    report['detailed_analysis'] = {
        'concept_analysis': refined_concepts,
        'relationship_network': relationships,
        'paradigm_models': paradigm_models,
        'theme_development': themes,
        'insight_generation': insights
    }
    
    # Recommendations
    report['recommendations'] = generate_prioritized_recommendations(insights, themes)
    
    # Implementation Roadmap
    report['implementation_roadmap'] = generate_implementation_roadmap(report['recommendations'])
    
    return report

def generate_prioritized_recommendations(insights, themes):
    """Generate prioritized recommendations based on insights and theme validation."""
    
    recommendations = {
        'high_priority': [],
        'medium_priority': [],
        'long_term': []
    }
    
    # Prioritize based on theme validation scores and insight types
    for theme_name, theme_data in themes.items():
        theme_insights = insights[theme_name]
        validation_score = theme_data['validation_score']
        
        # High priority: immediate actions from high-validation themes
        if validation_score > 0.7:
            for action in theme_insights['immediate_actions']:
                recommendations['high_priority'].append({
                    'action': action,
                    'theme': theme_name,
                    'validation_score': validation_score,
                    'type': 'immediate_action'
                })
        
        # Medium priority: system improvements from validated themes
        if validation_score > 0.5:
            for improvement in theme_insights['system_improvements']:
                recommendations['medium_priority'].append({
                    'action': improvement,
                    'theme': theme_name,
                    'validation_score': validation_score,
                    'type': 'system_improvement'
                })
        
        # Long-term: process changes and strategic improvements
        for change in theme_insights['process_changes']:
            recommendations['long_term'].append({
                'action': change,
                'theme': theme_name,
                'validation_score': validation_score,
                'type': 'process_change'
            })
    
    # Sort by validation score within each priority level
    for priority_level in recommendations:
        recommendations[priority_level].sort(key=lambda x: x['validation_score'], reverse=True)
    
    return recommendations

def generate_implementation_roadmap(recommendations):
    """Generate implementation roadmap with timelines and dependencies."""
    
    roadmap = {
        'phase_1_immediate': {
            'timeline': '1-2 weeks',
            'focus': 'Quick wins and immediate improvements',
            'actions': recommendations['high_priority'][:3],
            'success_metrics': [
                'Reduced user frustration indicators',
                'Improved response relevance scores',
                'Decreased escalation rates'
            ]
        },
        'phase_2_short_term': {
            'timeline': '1-3 months',
            'focus': 'System improvements and capability enhancements',
            'actions': recommendations['medium_priority'][:5],
            'success_metrics': [
                'Improved intent recognition accuracy',
                'Enhanced response specificity',
                'Better user satisfaction scores'
            ]
        },
        'phase_3_long_term': {
            'timeline': '3-6 months',
            'focus': 'Process optimization and strategic improvements',
            'actions': recommendations['long_term'][:4],
            'success_metrics': [
                'Systematic process improvements',
                'Enhanced user experience metrics',
                'Reduced operational costs'
            ]
        }
    }
    
    return roadmap

# Generate comprehensive report
analysis_report = generate_comprehensive_analysis_report(
    open_coder, refined_concepts, relationships, paradigm_models, themes, insights
)

# Display report summary
print("\\n" + "="*60)
print("COMPREHENSIVE QUALITATIVE ANALYSIS REPORT")
print("="*60)

print("\\nEXECUTIVE SUMMARY:")
for key, value in analysis_report['executive_summary'].items():
    if isinstance(value, list):
        print(f"  {key.replace('_', ' ').title()}:")
        for item in value:
            print(f"    - {item}")
    else:
        print(f"  {key.replace('_', ' ').title()}: {value}")

print("\\nKEY RECOMMENDATIONS:")
print("  High Priority (1-2 weeks):")
for rec in analysis_report['recommendations']['high_priority'][:3]:
    print(f"    - {rec['action']} (Score: {rec['validation_score']:.2f})")

print("  Medium Priority (1-3 months):")
for rec in analysis_report['recommendations']['medium_priority'][:3]:
    print(f"    - {rec['action']} (Score: {rec['validation_score']:.2f})")

print("\\nIMPLEMENTATION ROADMAP:")
for phase, details in analysis_report['implementation_roadmap'].items():
    print(f"  {phase.replace('_', ' ').title()}: {details['timeline']}")
    print(f"    Focus: {details['focus']}")
    print(f"    Actions: {len(details['actions'])} planned")
```

## Exercise Completion and Reflection

### Step 8: Self-Assessment and Learning Reflection

```python
def conduct_self_assessment():
    """Conduct self-assessment of exercise completion and learning."""
    
    assessment_criteria = {
        'technical_execution': {
            'open_coding_implementation': 'Did you successfully implement systematic open coding?',
            'axial_coding_application': 'Did you identify meaningful relationships between concepts?',
            'thematic_development': 'Did you develop coherent and validated themes?',
            'insight_generation': 'Did you generate actionable insights from the analysis?'
        },
        'analytical_depth': {
            'concept_saturation': 'Did you achieve theoretical saturation for key concepts?',
            'relationship_validation': 'Did you validate relationships through multiple methods?',
            'theme_coherence': 'Are your themes internally coherent and well-supported?',
            'practical_relevance': 'Are your insights practically relevant and actionable?'
        },
        'methodological_rigor': {
            'systematic_approach': 'Did you follow systematic qualitative research methods?',
            'validation_procedures': 'Did you implement appropriate validation procedures?',
            'documentation_quality': 'Did you document your analytical process thoroughly?',
            'reflexivity': 'Did you reflect on your analytical choices and potential biases?'
        }
    }
    
    print("\\n" + "="*50)
    print("EXERCISE SELF-ASSESSMENT")
    print("="*50)
    
    print("\\nPlease reflect on the following criteria:")
    print("Rate each item on a scale of 1-5 (1=Poor, 5=Excellent)\\n")
    
    for category, criteria in assessment_criteria.items():
        print(f"{category.replace('_', ' ').upper()}:")
        for criterion, question in criteria.items():
            print(f"  {question}")
            print(f"    [{criterion}]: ___/5")
        print()
    
    reflection_prompts = [
        "What was the most challenging aspect of the qualitative analysis process?",
        "Which insights surprised you or challenged your initial assumptions?",
        "How might you apply these qualitative research methods in your own work?",
        "What would you do differently if you repeated this analysis?",
        "How do the qualitative insights complement quantitative error analysis?"
    ]
    
    print("REFLECTION QUESTIONS:")
    for i, prompt in enumerate(reflection_prompts, 1):
        print(f"{i}. {prompt}")
        print("   Response: _______________\\n")

def generate_learning_portfolio_entry():
    """Generate portfolio entry documenting learning outcomes."""
    
    portfolio_entry = {
        'exercise_title': 'Qualitative Error Analysis Using Open and Axial Coding',
        'completion_date': datetime.now().strftime('%Y-%m-%d'),
        'learning_objectives_met': [
            'Applied systematic open coding to break down error data into concepts',
            'Used axial coding to identify relationships and build paradigm models',
            'Developed validated themes through systematic thematic analysis',
            'Generated actionable insights for AI system improvement',
            'Integrated qualitative and quantitative analytical approaches'
        ],
        'key_skills_developed': [
            'Qualitative research methodology',
            'Systematic coding and categorization',
            'Relationship mapping and analysis',
            'Theme development and validation',
            'Insight generation and recommendation development'
        ],
        'artifacts_created': [
            'Comprehensive concept taxonomy with validation metrics',
            'Relationship network visualization and analysis',
            'Paradigm models explaining error phenomena',
            'Validated themes with supporting evidence',
            'Actionable insights and implementation roadmap'
        ],
        'practical_applications': [
            'Error analysis for AI system improvement',
            'User experience research and optimization',
            'System capability assessment and enhancement',
            'Process improvement and optimization',
            'Strategic planning for AI system development'
        ],
        'next_steps': [
            'Apply methods to real-world error datasets',
            'Integrate with quantitative analysis approaches',
            'Develop automated coding assistance tools',
            'Expand to other qualitative research contexts',
            'Share insights with development and product teams'
        ]
    }
    
    return portfolio_entry

# Conduct self-assessment
conduct_self_assessment()

# Generate portfolio entry
portfolio_entry = generate_learning_portfolio_entry()

print("\\n" + "="*50)
print("LEARNING PORTFOLIO ENTRY GENERATED")
print("="*50)
print("\\nKey Achievements:")
for objective in portfolio_entry['learning_objectives_met']:
    print(f"  ✓ {objective}")

print("\\nArtifacts Created:")
for artifact in portfolio_entry['artifacts_created']:
    print(f"  📄 {artifact}")

print("\\nNext Steps:")
for step in portfolio_entry['next_steps']:
    print(f"  → {step}")
```

## Summary and Key Takeaways

This exercise provided comprehensive hands-on experience with qualitative error analysis using systematic open coding and axial coding methodologies. Through systematic application of these research methods, you have:

### Technical Skills Developed
- **Systematic Open Coding**: Breaking down complex error data into meaningful concepts and categories
- **Axial Coding Implementation**: Identifying relationships and building explanatory paradigm models
- **Thematic Analysis**: Developing validated themes that explain error patterns
- **Mixed-Methods Integration**: Combining qualitative insights with quantitative validation

### Analytical Capabilities Enhanced
- **Pattern Recognition**: Identifying subtle patterns that quantitative analysis might miss
- **Relationship Mapping**: Understanding complex relationships between error phenomena
- **Insight Generation**: Translating analytical findings into actionable improvements
- **Validation Techniques**: Ensuring analytical rigor through systematic validation

### Practical Applications
- **Error Analysis**: Comprehensive understanding of AI system error patterns
- **System Improvement**: Evidence-based recommendations for enhancement
- **Process Optimization**: Systematic approaches to continuous improvement
- **Strategic Planning**: Long-term roadmaps based on deep analytical insights

### Integration with Module 2 Concepts
This exercise directly reinforces the qualitative research methodologies from Section 6, providing practical experience with the systematic approaches needed for deep error analysis. The insights generated complement the quantitative techniques from other sections, creating a comprehensive analytical toolkit.

The qualitative analysis capabilities developed here will be essential for the LLM-as-Judge implementation in Exercise 2, where understanding the nuanced patterns of evaluation will inform the design of more effective automated systems.

---

*This exercise transforms theoretical knowledge of qualitative research methods into practical analytical skills, enabling you to uncover deep insights that drive meaningful improvements in AI system quality and user experience.*

