# Exercise 2: Tool Calling Evaluation System

## Learning Objectives

By completing this exercise, you will:
- Implement specialized evaluation frameworks for tool calling and function execution
- Build parameter validation systems for function calls
- Create execution outcome assessment mechanisms
- Develop multi-step reasoning evaluation protocols
- Deploy a comprehensive tool calling evaluation system

## Background Context

Tool calling evaluation represents one of the most complex challenges in AI evaluation. Unlike simple text generation, tool calling involves multiple dimensions: function selection appropriateness, parameter accuracy, execution logic, and outcome effectiveness. This exercise provides hands-on experience with specialized evaluation techniques for these scenarios.

You'll build a complete evaluation system that can assess AI agents' ability to select appropriate tools, provide correct parameters, execute logical sequences, and achieve desired outcomes.

## Prerequisites

- Completion of Module 3 Sections 1 and 6
- Understanding of function calling and tool usage in AI systems
- Strong Python programming skills
- Familiarity with API design and validation
- Basic knowledge of multi-step reasoning evaluation

## Setup Requirements

### Environment Setup
```bash
# Install additional packages for tool calling evaluation
pip install jsonschema pydantic typing-extensions functools

# Tool calling simulation packages
pip install requests beautifulsoup4 pandas numpy
```

### Sample Tool Definitions
```python
import json
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

# Define sample tools for evaluation
SAMPLE_TOOLS = {
    "web_search": {
        "name": "web_search",
        "description": "Search the web for information on a given topic",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20
                }
            },
            "required": ["query"]
        }
    },
    "calculate": {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    },
    "send_email": {
        "name": "send_email",
        "description": "Send an email to specified recipients",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Email recipients"
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject"
                },
                "body": {
                    "type": "string",
                    "description": "Email body content"
                }
            },
            "required": ["to", "subject", "body"]
        }
    }
}

# Save tool definitions
with open('sample_tools.json', 'w') as f:
    json.dump(SAMPLE_TOOLS, f, indent=2)
```

### Sample Tool Calling Scenarios
```python
# Sample scenarios for evaluation
EVALUATION_SCENARIOS = [
    {
        "id": "scenario_1",
        "task": "Find information about the latest developments in quantum computing and calculate the market size growth rate",
        "expected_tools": ["web_search", "calculate"],
        "sample_execution": [
            {
                "tool": "web_search",
                "parameters": {"query": "latest quantum computing developments 2024", "num_results": 10},
                "rationale": "Need current information about quantum computing developments"
            },
            {
                "tool": "calculate", 
                "parameters": {"expression": "(50.2 - 42.1) / 42.1 * 100"},
                "rationale": "Calculate growth rate from market size data found in search"
            }
        ],
        "success_criteria": {
            "tool_selection": "Both web_search and calculate tools should be used",
            "parameter_accuracy": "Search query should be relevant and specific",
            "execution_logic": "Search should precede calculation",
            "outcome_effectiveness": "Should provide both information and calculated growth rate"
        }
    },
    {
        "id": "scenario_2",
        "task": "Send a summary email to the team about today's meeting outcomes",
        "expected_tools": ["send_email"],
        "sample_execution": [
            {
                "tool": "send_email",
                "parameters": {
                    "to": ["team@company.com"],
                    "subject": "Meeting Summary - Project Alpha Discussion",
                    "body": "Hi team,\n\nHere's a summary of today's meeting:\n- Discussed project timeline\n- Reviewed budget allocation\n- Assigned action items\n\nBest regards"
                },
                "rationale": "Sending structured summary to keep team informed"
            }
        ],
        "success_criteria": {
            "tool_selection": "send_email tool should be selected",
            "parameter_accuracy": "All required parameters should be provided correctly",
            "execution_logic": "Single tool execution is appropriate",
            "outcome_effectiveness": "Email should contain relevant meeting information"
        }
    }
]

# Save scenarios
with open('evaluation_scenarios.json', 'w') as f:
    json.dump(EVALUATION_SCENARIOS, f, indent=2)
```

## Implementation Tasks

### Task 1: Function Selection Evaluator (90 minutes)

Implement the `FunctionSelectionEvaluator` class to assess tool selection appropriateness.

```python
import jsonschema
from typing import Dict, List, Any, Optional

class FunctionSelectionEvaluator:
    """
    Evaluates the appropriateness of function/tool selection for given tasks.
    """
    
    def __init__(self, available_tools: Dict[str, Any]):
        self.available_tools = available_tools
        self.selection_history = []
        
    def evaluate_tool_selection(self, 
                               task_description: str,
                               selected_tools: List[str],
                               expected_tools: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate whether selected tools are appropriate for the given task.
        
        TODO: Implement evaluation logic that:
        1. Analyzes task requirements
        2. Assesses tool appropriateness
        3. Identifies missing or unnecessary tools
        4. Provides detailed feedback
        """
        # YOUR CODE HERE
        pass
    
    def _analyze_task_requirements(self, task_description: str) -> Dict[str, Any]:
        """
        Analyze task to identify required capabilities.
        
        TODO: Implement task analysis using LLM to identify:
        - Required capabilities
        - Information needs
        - Action requirements
        - Output expectations
        """
        # YOUR CODE HERE
        pass
    
    def _assess_tool_coverage(self, 
                             task_requirements: Dict[str, Any],
                             selected_tools: List[str]) -> Dict[str, Any]:
        """
        Assess how well selected tools cover task requirements.
        
        TODO: Implement coverage analysis.
        """
        # YOUR CODE HERE
        pass
    
    async def create_selection_evaluation_prompt(self, 
                                               task: str,
                                               tools: List[str],
                                               available_tools: Dict[str, Any]) -> str:
        """
        Create prompt for LLM-based tool selection evaluation.
        
        TODO: Build comprehensive evaluation prompt.
        """
        # YOUR CODE HERE
        pass

# Test function selection evaluator
async def test_function_selection():
    with open('sample_tools.json', 'r') as f:
        tools = json.load(f)
    
    with open('evaluation_scenarios.json', 'r') as f:
        scenarios = json.load(f)
    
    evaluator = FunctionSelectionEvaluator(tools)
    
    for scenario in scenarios:
        print(f"\nEvaluating Tool Selection for: {scenario['task']}")
        
        selected_tools = [exec['tool'] for exec in scenario['sample_execution']]
        expected_tools = scenario['expected_tools']
        
        result = evaluator.evaluate_tool_selection(
            task_description=scenario['task'],
            selected_tools=selected_tools,
            expected_tools=expected_tools
        )
        
        print("Selection Evaluation Result:")
        print(json.dumps(result, indent=2))

# asyncio.run(test_function_selection())
```

### Task 2: Parameter Validation System (120 minutes)

Implement the `ParameterValidator` class for comprehensive parameter accuracy assessment.

```python
class ParameterValidator:
    """
    Comprehensive parameter validation system for tool calling evaluation.
    """
    
    def __init__(self, tool_schemas: Dict[str, Any]):
        self.tool_schemas = tool_schemas
        self.validation_history = []
        
    def validate_parameters(self, 
                          tool_name: str,
                          provided_parameters: Dict[str, Any],
                          context: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate parameters for a specific tool call.
        
        TODO: Implement comprehensive parameter validation:
        1. Schema validation
        2. Semantic appropriateness
        3. Context relevance
        4. Completeness assessment
        """
        # YOUR CODE HERE
        pass
    
    def _validate_schema_compliance(self, 
                                  tool_name: str,
                                  parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameters against tool schema.
        
        TODO: Use jsonschema to validate parameter structure and types.
        """
        # YOUR CODE HERE
        pass
    
    def _assess_semantic_appropriateness(self, 
                                       tool_name: str,
                                       parameters: Dict[str, Any],
                                       context: Optional[str] = None) -> Dict[str, Any]:
        """
        Assess semantic appropriateness of parameter values.
        
        TODO: Use LLM to evaluate parameter quality and relevance.
        """
        # YOUR CODE HERE
        pass
    
    async def create_parameter_evaluation_prompt(self, 
                                               tool_name: str,
                                               parameters: Dict[str, Any],
                                               context: str) -> str:
        """
        Create prompt for LLM-based parameter evaluation.
        
        TODO: Build evaluation prompt for parameter assessment.
        """
        # YOUR CODE HERE
        pass

# Test parameter validation
async def test_parameter_validation():
    with open('sample_tools.json', 'r') as f:
        tools = json.load(f)
    
    with open('evaluation_scenarios.json', 'r') as f:
        scenarios = json.load(f)
    
    validator = ParameterValidator(tools)
    
    for scenario in scenarios:
        print(f"\nValidating Parameters for: {scenario['task']}")
        
        for execution in scenario['sample_execution']:
            tool_name = execution['tool']
            parameters = execution['parameters']
            
            result = validator.validate_parameters(
                tool_name=tool_name,
                provided_parameters=parameters,
                context=scenario['task']
            )
            
            print(f"\nTool: {tool_name}")
            print(f"Parameters: {parameters}")
            print("Validation Result:")
            print(json.dumps(result, indent=2))

# asyncio.run(test_parameter_validation())
```

### Task 3: Execution Logic Evaluator (90 minutes)

Implement the `ExecutionLogicEvaluator` class for multi-step reasoning assessment.

```python
class ExecutionLogicEvaluator:
    """
    Evaluates the logic and sequence of tool execution in multi-step scenarios.
    """
    
    def __init__(self):
        self.execution_patterns = {}
        self.logic_history = []
        
    def evaluate_execution_sequence(self, 
                                  task_description: str,
                                  execution_sequence: List[Dict[str, Any]],
                                  expected_outcome: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the logic and appropriateness of tool execution sequence.
        
        TODO: Implement sequence evaluation:
        1. Dependency analysis
        2. Order appropriateness
        3. Efficiency assessment
        4. Completeness evaluation
        """
        # YOUR CODE HERE
        pass
    
    def _analyze_dependencies(self, 
                            execution_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze dependencies between tool executions.
        
        TODO: Identify and validate execution dependencies.
        """
        # YOUR CODE HERE
        pass
    
    def _assess_execution_order(self, 
                              task_description: str,
                              execution_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess whether execution order is logical and efficient.
        
        TODO: Evaluate execution order appropriateness.
        """
        # YOUR CODE HERE
        pass
    
    async def create_logic_evaluation_prompt(self, 
                                           task: str,
                                           sequence: List[Dict[str, Any]]) -> str:
        """
        Create prompt for LLM-based execution logic evaluation.
        
        TODO: Build comprehensive logic evaluation prompt.
        """
        # YOUR CODE HERE
        pass

# Test execution logic evaluation
async def test_execution_logic():
    with open('evaluation_scenarios.json', 'r') as f:
        scenarios = json.load(f)
    
    evaluator = ExecutionLogicEvaluator()
    
    for scenario in scenarios:
        print(f"\nEvaluating Execution Logic for: {scenario['task']}")
        
        result = evaluator.evaluate_execution_sequence(
            task_description=scenario['task'],
            execution_sequence=scenario['sample_execution']
        )
        
        print("Logic Evaluation Result:")
        print(json.dumps(result, indent=2))

# asyncio.run(test_execution_logic())
```

### Task 4: Comprehensive Tool Calling Evaluation System (120 minutes)

Integrate all components into a comprehensive evaluation system.

```python
class ToolCallingEvaluationSystem:
    """
    Comprehensive evaluation system for tool calling and function execution.
    """
    
    def __init__(self, tool_definitions: Dict[str, Any]):
        self.tool_definitions = tool_definitions
        self.function_evaluator = FunctionSelectionEvaluator(tool_definitions)
        self.parameter_validator = ParameterValidator(tool_definitions)
        self.logic_evaluator = ExecutionLogicEvaluator()
        self.evaluation_history = []
        
    async def comprehensive_evaluation(self, 
                                     task_description: str,
                                     execution_plan: List[Dict[str, Any]],
                                     expected_outcome: Optional[str] = None) -> Dict[str, Any]:
        """
        Conduct comprehensive evaluation of tool calling scenario.
        
        TODO: Implement integrated evaluation that:
        1. Evaluates tool selection
        2. Validates all parameters
        3. Assesses execution logic
        4. Provides overall assessment
        5. Generates improvement recommendations
        """
        # YOUR CODE HERE
        pass
    
    def _calculate_overall_score(self, 
                               selection_result: Dict[str, Any],
                               validation_results: List[Dict[str, Any]],
                               logic_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall evaluation score from component assessments.
        
        TODO: Implement weighted scoring system.
        """
        # YOUR CODE HERE
        pass
    
    def _generate_improvement_recommendations(self, 
                                            evaluation_results: Dict[str, Any]) -> List[str]:
        """
        Generate specific improvement recommendations.
        
        TODO: Analyze results and provide actionable feedback.
        """
        # YOUR CODE HERE
        pass

# Comprehensive integration test
async def comprehensive_tool_evaluation_test():
    with open('sample_tools.json', 'r') as f:
        tools = json.load(f)
    
    with open('evaluation_scenarios.json', 'r') as f:
        scenarios = json.load(f)
    
    system = ToolCallingEvaluationSystem(tools)
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE EVALUATION: {scenario['id']}")
        print(f"Task: {scenario['task']}")
        print(f"{'='*60}")
        
        result = await system.comprehensive_evaluation(
            task_description=scenario['task'],
            execution_plan=scenario['sample_execution']
        )
        
        print("\nCOMPREHENSIVE EVALUATION RESULTS:")
        print(json.dumps(result, indent=2))
        
        # Compare with expected criteria
        expected = scenario['success_criteria']
        print(f"\nEXPECTED SUCCESS CRITERIA:")
        for criterion, description in expected.items():
            print(f"- {criterion}: {description}")

# asyncio.run(comprehensive_tool_evaluation_test())
```

## Testing and Validation

### Unit Tests
```python
import pytest

class TestFunctionSelectionEvaluator:
    def test_task_analysis(self):
        """Test task requirement analysis."""
        # YOUR TEST CODE HERE
        pass
    
    def test_tool_coverage_assessment(self):
        """Test tool coverage analysis."""
        # YOUR TEST CODE HERE
        pass

class TestParameterValidator:
    def test_schema_validation(self):
        """Test parameter schema validation."""
        # YOUR TEST CODE HERE
        pass
    
    def test_semantic_assessment(self):
        """Test semantic appropriateness evaluation."""
        # YOUR TEST CODE HERE
        pass

class TestExecutionLogicEvaluator:
    def test_dependency_analysis(self):
        """Test execution dependency analysis."""
        # YOUR TEST CODE HERE
        pass
    
    def test_order_assessment(self):
        """Test execution order evaluation."""
        # YOUR TEST CODE HERE
        pass

# Run tests
# pytest test_tool_calling_evaluation.py -v
```

### Performance Tests
```python
async def performance_test():
    """Test system performance with multiple tool calling scenarios."""
    
    system = ToolCallingEvaluationSystem(tools)
    
    # Generate multiple test scenarios
    test_scenarios = [
        {
            "task": f"Test scenario {i}",
            "execution_plan": [
                {"tool": "web_search", "parameters": {"query": f"test query {i}"}}
            ]
        }
        for i in range(20)
    ]
    
    start_time = time.time()
    
    tasks = [
        system.comprehensive_evaluation(
            scenario["task"], 
            scenario["execution_plan"]
        )
        for scenario in test_scenarios
    ]
    
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Evaluated {len(test_scenarios)} tool calling scenarios in {total_time:.2f} seconds")
    print(f"Average time per evaluation: {total_time/len(test_scenarios):.2f} seconds")

# asyncio.run(performance_test())
```

## Extension Challenges

### Challenge 1: Dynamic Tool Discovery
Implement evaluation for scenarios where tools are discovered dynamically during execution.

### Challenge 2: Error Handling Evaluation
Create evaluation frameworks for assessing how well systems handle tool execution errors.

### Challenge 3: Context-Aware Parameter Generation
Implement evaluation for parameter generation that adapts based on previous tool outputs.

### Challenge 4: Multi-Agent Tool Coordination
Extend the system to evaluate tool calling in multi-agent scenarios.

## Reflection Questions

1. **Tool Selection Strategy**: What factors are most important when evaluating tool selection appropriateness?

2. **Parameter Quality**: How do you balance schema compliance with semantic appropriateness in parameter validation?

3. **Execution Logic**: What makes a tool execution sequence efficient and logical?

4. **Error Scenarios**: How would you evaluate tool calling performance when tools fail or return unexpected results?

5. **Scalability**: How would you adapt this evaluation system for scenarios with hundreds of available tools?

## Solution Guidelines

### Key Implementation Points
- Use schema validation for structural parameter checking
- Implement semantic evaluation using LLMs for context appropriateness
- Analyze execution dependencies to validate logical flow
- Provide detailed feedback for each evaluation dimension
- Cache tool definitions and validation results for performance

### Common Pitfalls
- Over-relying on schema validation without semantic checking
- Ignoring execution order dependencies
- Insufficient error handling for malformed tool calls
- Missing context consideration in parameter evaluation
- Poor performance due to redundant LLM calls

### Best Practices
- Combine automated validation with LLM-based assessment
- Provide specific, actionable feedback for improvements
- Consider task context in all evaluation dimensions
- Implement comprehensive error handling and recovery
- Use caching and batching for performance optimization

---

**Estimated Completion Time**: 4-5 hours  
**Difficulty Level**: Advanced  
**Skills Developed**: Tool calling evaluation, parameter validation, multi-step reasoning assessment

This exercise provides comprehensive experience with evaluating one of the most complex aspects of AI systems: tool calling and function execution.

