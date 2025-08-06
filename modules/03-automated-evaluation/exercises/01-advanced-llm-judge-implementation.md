# Exercise 1: Advanced LLM-as-Judge Implementation

## Learning Objectives

By completing this exercise, you will:
- Implement a production-grade LLM-as-Judge system with chain-of-thought reasoning
- Build multi-dimensional evaluation frameworks with weighted scoring
- Develop calibration protocols for human-AI alignment
- Create bias detection and correction mechanisms
- Deploy a scalable evaluation system with performance optimization

## Background Context

This exercise builds upon the advanced LLM-as-Judge implementation concepts from Section 6, providing hands-on experience with the most sophisticated automated evaluation techniques. You'll implement a complete system that combines chain-of-thought reasoning, multi-dimensional assessment, and calibration protocols.

The system you'll build represents the state-of-the-art in automated evaluation, incorporating lessons learned from production deployments and addressing the key challenges identified in real-world implementations.

## Prerequisites

- Completion of Module 3 Sections 1 and 6
- Strong Python programming skills
- Familiarity with OpenAI API and async programming
- Understanding of statistical analysis concepts
- Basic knowledge of machine learning evaluation metrics

## Setup Requirements

### Environment Setup
```bash
# Create virtual environment
python -m venv llm_judge_env
source llm_judge_env/bin/activate  # On Windows: llm_judge_env\Scripts\activate

# Install required packages
pip install openai anthropic asyncio aiohttp numpy scipy scikit-learn pandas matplotlib seaborn pytest

# Set environment variables
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### Data Setup
```python
# Download exercise data
import requests
import json

# Sample evaluation data for testing
sample_data = {
    "content_samples": [
        {
            "id": "sample_1",
            "content": "The quarterly financial report shows a 15% increase in revenue compared to the previous quarter. This growth is primarily attributed to strong performance in our digital services division, which saw a 28% increase in client acquisitions. Operating expenses remained stable at 12% of total revenue, indicating efficient cost management. The company's debt-to-equity ratio improved from 0.45 to 0.38, reflecting stronger financial health.",
            "type": "financial_report",
            "human_evaluation": {
                "accuracy": 85,
                "clarity": 90,
                "completeness": 80,
                "relevance": 95,
                "overall_score": 87
            }
        },
        {
            "id": "sample_2", 
            "content": "To implement the new user authentication system, we need to integrate OAuth 2.0 with our existing database. The process involves creating new API endpoints for login, logout, and token refresh. We should also implement rate limiting to prevent abuse and ensure proper error handling for invalid credentials.",
            "type": "technical_documentation",
            "human_evaluation": {
                "accuracy": 90,
                "clarity": 85,
                "completeness": 75,
                "relevance": 90,
                "overall_score": 85
            }
        }
    ]
}

# Save sample data
with open('exercise_data.json', 'w') as f:
    json.dump(sample_data, f, indent=2)
```

## Implementation Tasks

### Task 1: Chain-of-Thought Evaluator Implementation (90 minutes)

Implement the `ChainOfThoughtEvaluator` class with structured reasoning capabilities.

```python
import openai
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

class ChainOfThoughtEvaluator:
    """
    Advanced LLM-as-Judge implementation with chain-of-thought reasoning
    for transparent and reliable evaluation processes.
    """
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        self.evaluation_history = []
        
    def create_cot_evaluation_prompt(self, 
                                   evaluation_criteria: Dict[str, Any],
                                   content_to_evaluate: str,
                                   reference_content: Optional[str] = None,
                                   domain_context: Optional[str] = None) -> str:
        """
        Create chain-of-thought evaluation prompt with structured reasoning.
        
        TODO: Implement this method following the pattern from Section 6.
        Your implementation should:
        1. Create a structured reasoning framework
        2. Include step-by-step analysis instructions
        3. Provide clear scoring guidelines
        4. Format the response as structured JSON
        
        Hint: Use the template from Section 6 as a starting point.
        """
        # YOUR CODE HERE
        pass
    
    async def evaluate_with_cot(self, 
                               content: str,
                               criteria: Dict[str, Any],
                               reference: Optional[str] = None,
                               domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Conduct evaluation using chain-of-thought reasoning.
        
        TODO: Implement this method to:
        1. Generate the evaluation prompt
        2. Call the LLM API
        3. Parse the structured response
        4. Handle errors gracefully
        5. Store evaluation history
        """
        # YOUR CODE HERE
        pass
    
    def _extract_evaluation_result(self, response_text: str) -> Dict[str, Any]:
        """
        Extract structured evaluation result from LLM response.
        
        TODO: Implement JSON extraction and validation logic.
        """
        # YOUR CODE HERE
        pass

# Test your implementation
async def test_cot_evaluator():
    evaluator = ChainOfThoughtEvaluator()
    
    criteria = {
        "accuracy": {
            "definition": "Factual correctness and precision of information",
            "scale": "0-100",
            "indicators": "Correct facts, precise numbers, accurate statements"
        },
        "clarity": {
            "definition": "Clear communication and understandability",
            "scale": "0-100", 
            "indicators": "Clear language, logical structure, easy to understand"
        }
    }
    
    # Load test data
    with open('exercise_data.json', 'r') as f:
        data = json.load(f)
    
    sample = data['content_samples'][0]
    
    result = await evaluator.evaluate_with_cot(
        content=sample['content'],
        criteria=criteria,
        domain="financial_reporting"
    )
    
    print("Evaluation Result:")
    print(json.dumps(result, indent=2))
    
    return result

# Run the test
# asyncio.run(test_cot_evaluator())
```

**Expected Output Structure:**
```json
{
    "scores": {
        "accuracy": 85,
        "clarity": 90
    },
    "overall_score": 87,
    "confidence": 85,
    "reasoning_summary": "The content demonstrates high accuracy with specific financial metrics...",
    "improvement_recommendations": [
        "Add more context about market conditions",
        "Include comparison with industry benchmarks"
    ]
}
```

### Task 2: Multi-Dimensional Evaluation Framework (120 minutes)

Implement the `MultiDimensionalEvaluator` class for comprehensive quality assessment.

```python
class MultiDimensionalEvaluator:
    """
    Advanced multi-dimensional evaluation framework for comprehensive
    quality assessment across multiple criteria.
    """
    
    def __init__(self, evaluation_dimensions: Dict[str, Dict[str, Any]]):
        self.dimensions = evaluation_dimensions
        self.dimension_weights = self._calculate_dimension_weights()
        self.evaluation_cache = {}
        
    def _calculate_dimension_weights(self) -> Dict[str, float]:
        """
        Calculate normalized weights for evaluation dimensions.
        
        TODO: Implement weight calculation and normalization.
        """
        # YOUR CODE HERE
        pass
    
    def create_multidimensional_prompt(self, 
                                     content: str,
                                     context: Optional[str] = None) -> str:
        """
        Create comprehensive multi-dimensional evaluation prompt.
        
        TODO: Build a prompt that:
        1. Describes all evaluation dimensions
        2. Provides clear assessment instructions
        3. Requests structured JSON output
        4. Includes cross-dimensional analysis
        """
        # YOUR CODE HERE
        pass
    
    async def evaluate_multidimensional(self, 
                                      content: str,
                                      context: Optional[str] = None,
                                      model: str = "gpt-4") -> Dict[str, Any]:
        """
        Conduct comprehensive multi-dimensional evaluation.
        
        TODO: Implement the complete evaluation workflow.
        """
        # YOUR CODE HERE
        pass

# Test implementation
async def test_multidimensional_evaluator():
    dimensions = {
        "accuracy": {
            "definition": "Factual correctness and precision",
            "weight": 0.3,
            "scale": "0-100",
            "indicators": "Correct facts, precise data"
        },
        "clarity": {
            "definition": "Clear communication and readability",
            "weight": 0.25,
            "scale": "0-100",
            "indicators": "Clear language, logical flow"
        },
        "completeness": {
            "definition": "Comprehensive coverage of topic",
            "weight": 0.25,
            "scale": "0-100",
            "indicators": "All key points covered"
        },
        "relevance": {
            "definition": "Relevance to intended purpose",
            "weight": 0.2,
            "scale": "0-100",
            "indicators": "On-topic, purposeful content"
        }
    }
    
    evaluator = MultiDimensionalEvaluator(dimensions)
    
    # Test with sample data
    with open('exercise_data.json', 'r') as f:
        data = json.load(f)
    
    sample = data['content_samples'][1]
    
    result = await evaluator.evaluate_multidimensional(
        content=sample['content'],
        context="Technical documentation for software development team"
    )
    
    print("Multi-dimensional Evaluation:")
    print(json.dumps(result, indent=2))
    
    return result

# asyncio.run(test_multidimensional_evaluator())
```

### Task 3: Calibration System Implementation (90 minutes)

Implement the `LLMJudgeCalibrator` class for human-AI alignment analysis.

```python
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error
import numpy as np

class LLMJudgeCalibrator:
    """
    Comprehensive calibration system for LLM-as-Judge implementations.
    Analyzes human-AI agreement and implements bias correction mechanisms.
    """
    
    def __init__(self, human_evaluations: List[Dict[str, Any]], 
                 llm_evaluations: List[Dict[str, Any]]):
        self.human_evaluations = human_evaluations
        self.llm_evaluations = llm_evaluations
        self.calibration_results = {}
        self.bias_corrections = {}
        
    def conduct_comprehensive_calibration(self) -> Dict[str, Any]:
        """
        Conduct comprehensive calibration analysis.
        
        TODO: Implement calibration analysis including:
        1. Agreement metrics calculation
        2. Bias detection and analysis
        3. Correlation analysis
        4. Recommendation generation
        """
        # YOUR CODE HERE
        pass
    
    def _analyze_agreement(self) -> Dict[str, Any]:
        """
        Analyze agreement between human and LLM evaluations.
        
        TODO: Calculate correlation and agreement metrics.
        """
        # YOUR CODE HERE
        pass
    
    def _analyze_systematic_biases(self) -> Dict[str, Any]:
        """
        Analyze systematic biases in LLM evaluations.
        
        TODO: Detect score bias, length bias, and other systematic issues.
        """
        # YOUR CODE HERE
        pass
    
    def develop_bias_correction_mechanisms(self) -> Dict[str, Any]:
        """
        Develop bias correction mechanisms based on calibration analysis.
        
        TODO: Create correction algorithms for identified biases.
        """
        # YOUR CODE HERE
        pass

# Test calibration system
def test_calibration_system():
    # Create test data
    human_evals = [
        {"content_id": "sample_1", "overall_score": 87, "content": "test content 1"},
        {"content_id": "sample_2", "overall_score": 85, "content": "test content 2"}
    ]
    
    llm_evals = [
        {"content_id": "sample_1", "overall_score": 90, "content": "test content 1"},
        {"content_id": "sample_2", "overall_score": 82, "content": "test content 2"}
    ]
    
    calibrator = LLMJudgeCalibrator(human_evals, llm_evals)
    results = calibrator.conduct_comprehensive_calibration()
    
    print("Calibration Results:")
    print(json.dumps(results, indent=2))
    
    return results

# test_calibration_system()
```

### Task 4: Integration and Testing (60 minutes)

Create a comprehensive integration test that combines all components.

```python
class AdvancedLLMJudgeSystem:
    """
    Complete advanced LLM-as-Judge system integrating all components.
    """
    
    def __init__(self, evaluation_config: Dict[str, Any]):
        self.config = evaluation_config
        self.cot_evaluator = ChainOfThoughtEvaluator(
            model_name=config.get('model', 'gpt-4'),
            temperature=config.get('temperature', 0.1)
        )
        self.multidim_evaluator = MultiDimensionalEvaluator(
            config['dimensions']
        )
        self.calibrator = None  # Will be set when calibration data is available
        
    async def comprehensive_evaluation(self, 
                                     content: str,
                                     evaluation_type: str = "standard",
                                     context: Optional[str] = None) -> Dict[str, Any]:
        """
        Conduct comprehensive evaluation using all available methods.
        
        TODO: Implement integrated evaluation that:
        1. Uses chain-of-thought reasoning
        2. Performs multi-dimensional assessment
        3. Applies bias corrections if available
        4. Provides comprehensive results
        """
        # YOUR CODE HERE
        pass

# Integration test
async def integration_test():
    config = {
        "model": "gpt-4",
        "temperature": 0.1,
        "dimensions": {
            "accuracy": {"definition": "Factual correctness", "weight": 0.3},
            "clarity": {"definition": "Clear communication", "weight": 0.25},
            "completeness": {"definition": "Comprehensive coverage", "weight": 0.25},
            "relevance": {"definition": "Relevance to purpose", "weight": 0.2}
        }
    }
    
    system = AdvancedLLMJudgeSystem(config)
    
    # Load test data
    with open('exercise_data.json', 'r') as f:
        data = json.load(f)
    
    for sample in data['content_samples']:
        print(f"\nEvaluating Sample {sample['id']}:")
        print(f"Content: {sample['content'][:100]}...")
        
        result = await system.comprehensive_evaluation(
            content=sample['content'],
            context=f"Evaluation of {sample['type']}"
        )
        
        print("Results:")
        print(json.dumps(result, indent=2))
        
        # Compare with human evaluation
        human_score = sample['human_evaluation']['overall_score']
        llm_score = result.get('overall_score', 0)
        difference = abs(human_score - llm_score)
        
        print(f"Human Score: {human_score}")
        print(f"LLM Score: {llm_score}")
        print(f"Difference: {difference}")

# asyncio.run(integration_test())
```

## Testing and Validation

### Unit Tests
Create comprehensive unit tests for each component:

```python
import pytest

class TestChainOfThoughtEvaluator:
    def test_prompt_creation(self):
        """Test that prompts are created correctly."""
        # YOUR TEST CODE HERE
        pass
    
    def test_result_extraction(self):
        """Test JSON extraction from LLM responses."""
        # YOUR TEST CODE HERE
        pass

class TestMultiDimensionalEvaluator:
    def test_weight_calculation(self):
        """Test dimension weight calculation."""
        # YOUR TEST CODE HERE
        pass
    
    def test_prompt_generation(self):
        """Test multi-dimensional prompt generation."""
        # YOUR TEST CODE HERE
        pass

class TestCalibrationSystem:
    def test_agreement_analysis(self):
        """Test human-AI agreement calculation."""
        # YOUR TEST CODE HERE
        pass
    
    def test_bias_detection(self):
        """Test systematic bias detection."""
        # YOUR TEST CODE HERE
        pass

# Run tests
# pytest test_advanced_llm_judge.py -v
```

### Performance Tests
```python
import time
import asyncio

async def performance_test():
    """Test system performance with multiple evaluations."""
    
    system = AdvancedLLMJudgeSystem(config)
    
    # Generate test content
    test_contents = [
        f"Test content sample {i} with varying length and complexity..."
        for i in range(10)
    ]
    
    start_time = time.time()
    
    tasks = [
        system.comprehensive_evaluation(content)
        for content in test_contents
    ]
    
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Evaluated {len(test_contents)} samples in {total_time:.2f} seconds")
    print(f"Average time per evaluation: {total_time/len(test_contents):.2f} seconds")
    
    return results

# asyncio.run(performance_test())
```

## Extension Challenges

### Challenge 1: Ensemble Integration
Extend your implementation to support ensemble evaluation with multiple judges.

### Challenge 2: Real-time Calibration
Implement continuous calibration that updates bias corrections based on new human feedback.

### Challenge 3: Domain Adaptation
Create domain-specific evaluation configurations for different content types.

### Challenge 4: Performance Optimization
Implement caching, batching, and other optimizations for production deployment.

## Reflection Questions

1. **Evaluation Quality**: How does chain-of-thought reasoning improve evaluation quality compared to simple prompting?

2. **Bias Mitigation**: What types of biases did you observe in your implementation, and how effective were the correction mechanisms?

3. **Multi-dimensional Assessment**: How do weighted multi-dimensional evaluations provide more nuanced quality assessment?

4. **Production Readiness**: What additional features would be needed to deploy this system in a production environment?

5. **Scalability**: How would you modify the system to handle thousands of evaluations per hour?

## Solution Guidelines

### Key Implementation Points
- Use structured prompting with clear reasoning steps
- Implement robust error handling and fallback mechanisms
- Cache results to improve performance
- Validate all JSON responses before processing
- Monitor API usage and implement rate limiting

### Common Pitfalls
- Insufficient error handling for API failures
- Poor JSON parsing leading to system crashes
- Inadequate prompt engineering resulting in inconsistent responses
- Missing validation of evaluation criteria
- Lack of performance optimization for scale

### Best Practices
- Always validate input parameters
- Implement comprehensive logging for debugging
- Use async/await for better performance
- Cache expensive operations
- Provide clear error messages and recovery options

---

**Estimated Completion Time**: 5-6 hours  
**Difficulty Level**: Advanced  
**Skills Developed**: Production-grade LLM-as-Judge implementation, calibration protocols, multi-dimensional evaluation

This exercise provides hands-on experience with the most advanced automated evaluation techniques, preparing you for real-world deployment of sophisticated evaluation systems.

