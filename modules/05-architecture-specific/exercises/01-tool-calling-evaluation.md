# Exercise 1: Tool Calling Evaluation System

## Objective
Build a comprehensive evaluation system for AI agents that use tools and functions, implementing function selection assessment, parameter validation, and execution logic analysis.

## Duration
5-6 hours

## Skills Developed
- Function selection assessment and optimization
- Parameter validation with schema and semantic checking
- Execution logic analysis for multi-step workflows
- Tool calling evaluation system integration

## Prerequisites
- Understanding of tool calling frameworks from Section 6
- Basic knowledge of AI agent architectures
- Python programming experience
- Familiarity with API design and validation concepts

## Learning Outcomes
Upon completing this exercise, you will be able to:
- Design and implement comprehensive tool calling evaluation systems
- Build function selection assessment frameworks with appropriateness scoring
- Create parameter validation systems with schema and semantic analysis
- Develop execution logic analyzers for complex multi-step workflows
- Integrate tool calling evaluation into production AI systems

## Exercise Overview

In this exercise, you will build a production-grade tool calling evaluation system for a financial trading AI agent. The agent uses multiple financial tools including market data APIs, trading execution systems, risk assessment tools, and portfolio management functions. Your evaluation system must assess the agent's ability to select appropriate tools, validate parameters correctly, and execute complex trading workflows efficiently.

## Part 1: Function Selection Assessment Framework (90 minutes)

### 1.1 Understanding the Trading Agent Architecture

First, let's examine the trading agent's tool ecosystem:

```python
import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import uuid

class ToolCategory(Enum):
    MARKET_DATA = "market_data"
    TRADING_EXECUTION = "trading_execution"
    RISK_ASSESSMENT = "risk_assessment"
    PORTFOLIO_MANAGEMENT = "portfolio_management"
    COMPLIANCE_CHECK = "compliance_check"
    ANALYTICS = "analytics"

class ToolComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    CRITICAL = "critical"

@dataclass
class ToolDefinition:
    """Definition of a tool available to the trading agent."""
    tool_id: str
    name: str
    category: ToolCategory
    complexity: ToolComplexity
    description: str
    parameters: Dict[str, Any]
    required_permissions: List[str]
    cost_per_call: float
    average_latency: float
    reliability_score: float
    risk_level: str

@dataclass
class TradingContext:
    """Context information for trading decisions."""
    market_conditions: str
    portfolio_value: float
    risk_tolerance: str
    trading_session: str
    regulatory_constraints: List[str]
    time_sensitivity: str

@dataclass
class ToolSelectionDecision:
    """Record of a tool selection decision made by the agent."""
    decision_id: str
    timestamp: datetime
    context: TradingContext
    available_tools: List[str]
    selected_tool: str
    selection_reasoning: str
    confidence_score: float
    alternative_tools: List[str]

class TradingToolRegistry:
    """Registry of available trading tools with comprehensive metadata."""
    
    def __init__(self):
        self.tools = self._initialize_trading_tools()
        self.tool_relationships = self._define_tool_relationships()
        self.context_preferences = self._define_context_preferences()
    
    def _initialize_trading_tools(self) -> Dict[str, ToolDefinition]:
        """Initialize the comprehensive set of trading tools."""
        
        tools = {}
        
        # Market Data Tools
        tools["real_time_quotes"] = ToolDefinition(
            tool_id="real_time_quotes",
            name="Real-Time Market Quotes",
            category=ToolCategory.MARKET_DATA,
            complexity=ToolComplexity.SIMPLE,
            description="Get real-time stock quotes and market data",
            parameters={
                "symbols": {"type": "list", "required": True},
                "fields": {"type": "list", "default": ["price", "volume"]},
                "exchange": {"type": "string", "default": "NYSE"}
            },
            required_permissions=["market_data_access"],
            cost_per_call=0.01,
            average_latency=0.2,
            reliability_score=0.99,
            risk_level="low"
        )
        
        tools["historical_data"] = ToolDefinition(
            tool_id="historical_data",
            name="Historical Market Data",
            category=ToolCategory.MARKET_DATA,
            complexity=ToolComplexity.MODERATE,
            description="Retrieve historical price and volume data",
            parameters={
                "symbol": {"type": "string", "required": True},
                "start_date": {"type": "date", "required": True},
                "end_date": {"type": "date", "required": True},
                "interval": {"type": "string", "default": "1d"}
            },
            required_permissions=["historical_data_access"],
            cost_per_call=0.05,
            average_latency=1.5,
            reliability_score=0.98,
            risk_level="low"
        )
        
        # Trading Execution Tools
        tools["place_order"] = ToolDefinition(
            tool_id="place_order",
            name="Place Trading Order",
            category=ToolCategory.TRADING_EXECUTION,
            complexity=ToolComplexity.CRITICAL,
            description="Execute buy or sell orders in the market",
            parameters={
                "symbol": {"type": "string", "required": True},
                "quantity": {"type": "integer", "required": True},
                "order_type": {"type": "string", "required": True},
                "price": {"type": "float", "required": False},
                "time_in_force": {"type": "string", "default": "DAY"}
            },
            required_permissions=["trading_execution", "order_placement"],
            cost_per_call=5.00,
            average_latency=0.5,
            reliability_score=0.999,
            risk_level="critical"
        )
        
        tools["cancel_order"] = ToolDefinition(
            tool_id="cancel_order",
            name="Cancel Trading Order",
            category=ToolCategory.TRADING_EXECUTION,
            complexity=ToolComplexity.MODERATE,
            description="Cancel existing trading orders",
            parameters={
                "order_id": {"type": "string", "required": True},
                "symbol": {"type": "string", "required": True}
            },
            required_permissions=["trading_execution", "order_management"],
            cost_per_call=1.00,
            average_latency=0.3,
            reliability_score=0.998,
            risk_level="high"
        )
        
        # Risk Assessment Tools
        tools["portfolio_risk_analysis"] = ToolDefinition(
            tool_id="portfolio_risk_analysis",
            name="Portfolio Risk Analysis",
            category=ToolCategory.RISK_ASSESSMENT,
            complexity=ToolComplexity.COMPLEX,
            description="Comprehensive portfolio risk assessment",
            parameters={
                "portfolio_id": {"type": "string", "required": True},
                "risk_metrics": {"type": "list", "default": ["var", "beta", "sharpe"]},
                "time_horizon": {"type": "string", "default": "1d"}
            },
            required_permissions=["risk_analysis", "portfolio_access"],
            cost_per_call=2.50,
            average_latency=3.0,
            reliability_score=0.97,
            risk_level="medium"
        )
        
        tools["position_sizing"] = ToolDefinition(
            tool_id="position_sizing",
            name="Optimal Position Sizing",
            category=ToolCategory.RISK_ASSESSMENT,
            complexity=ToolComplexity.COMPLEX,
            description="Calculate optimal position sizes based on risk parameters",
            parameters={
                "symbol": {"type": "string", "required": True},
                "risk_tolerance": {"type": "float", "required": True},
                "portfolio_value": {"type": "float", "required": True},
                "volatility": {"type": "float", "required": False}
            },
            required_permissions=["risk_analysis", "position_management"],
            cost_per_call=1.50,
            average_latency=1.0,
            reliability_score=0.96,
            risk_level="medium"
        )
        
        return tools
    
    def _define_tool_relationships(self) -> Dict[str, Dict[str, Any]]:
        """Define relationships and dependencies between tools."""
        
        return {
            "complementary_tools": {
                "real_time_quotes": ["portfolio_risk_analysis", "position_sizing"],
                "historical_data": ["portfolio_risk_analysis", "position_sizing"],
                "portfolio_risk_analysis": ["position_sizing", "place_order"],
                "position_sizing": ["place_order"]
            },
            "prerequisite_tools": {
                "place_order": ["portfolio_risk_analysis", "position_sizing"],
                "cancel_order": ["place_order"]
            },
            "alternative_tools": {
                "real_time_quotes": ["historical_data"],
                "place_order": ["cancel_order"]
            }
        }
    
    def _define_context_preferences(self) -> Dict[str, Dict[str, float]]:
        """Define tool preferences based on trading context."""
        
        return {
            "market_conditions": {
                "volatile": {
                    "real_time_quotes": 0.9,
                    "portfolio_risk_analysis": 0.95,
                    "position_sizing": 0.9
                },
                "stable": {
                    "historical_data": 0.8,
                    "portfolio_risk_analysis": 0.7,
                    "position_sizing": 0.8
                }
            },
            "time_sensitivity": {
                "urgent": {
                    "real_time_quotes": 0.95,
                    "place_order": 0.9
                },
                "normal": {
                    "historical_data": 0.8,
                    "portfolio_risk_analysis": 0.85
                }
            },
            "risk_tolerance": {
                "conservative": {
                    "portfolio_risk_analysis": 0.95,
                    "position_sizing": 0.9
                },
                "aggressive": {
                    "real_time_quotes": 0.85,
                    "place_order": 0.8
                }
            }
        }

class FunctionSelectionEvaluator:
    """
    Comprehensive evaluator for function selection decisions in trading agents.
    """
    
    def __init__(self, tool_registry: TradingToolRegistry):
        self.tool_registry = tool_registry
        self.evaluation_metrics = self._initialize_evaluation_metrics()
        self.decision_history = []
    
    def _initialize_evaluation_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive evaluation metrics for function selection."""
        
        return {
            "appropriateness": {
                "weight": 0.3,
                "description": "How well the selected tool matches the task requirements",
                "scoring_criteria": {
                    "task_alignment": 0.4,
                    "context_suitability": 0.3,
                    "capability_match": 0.3
                }
            },
            "efficiency": {
                "weight": 0.25,
                "description": "Cost-effectiveness and performance of the tool selection",
                "scoring_criteria": {
                    "cost_optimization": 0.4,
                    "latency_optimization": 0.3,
                    "resource_utilization": 0.3
                }
            },
            "risk_awareness": {
                "weight": 0.25,
                "description": "Consideration of risk factors in tool selection",
                "scoring_criteria": {
                    "risk_level_appropriateness": 0.5,
                    "compliance_consideration": 0.3,
                    "reliability_factor": 0.2
                }
            },
            "strategic_thinking": {
                "weight": 0.2,
                "description": "Long-term and holistic thinking in tool selection",
                "scoring_criteria": {
                    "workflow_optimization": 0.4,
                    "alternative_consideration": 0.3,
                    "future_step_planning": 0.3
                }
            }
        }
    
    async def evaluate_function_selection(self, 
                                        decision: ToolSelectionDecision,
                                        evaluation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a function selection decision.
        """
        
        config = evaluation_config or {}
        
        # Evaluate appropriateness of the selection
        appropriateness_score = await self._evaluate_appropriateness(decision, config)
        
        # Evaluate efficiency of the selection
        efficiency_score = await self._evaluate_efficiency(decision, config)
        
        # Evaluate risk awareness in the selection
        risk_awareness_score = await self._evaluate_risk_awareness(decision, config)
        
        # Evaluate strategic thinking in the selection
        strategic_thinking_score = await self._evaluate_strategic_thinking(decision, config)
        
        # Calculate overall selection quality score
        overall_score = self._calculate_overall_score(
            appropriateness_score,
            efficiency_score,
            risk_awareness_score,
            strategic_thinking_score
        )
        
        # Generate detailed feedback and recommendations
        feedback = await self._generate_selection_feedback(
            decision,
            appropriateness_score,
            efficiency_score,
            risk_awareness_score,
            strategic_thinking_score
        )
        
        evaluation_result = {
            'decision_id': decision.decision_id,
            'evaluation_timestamp': datetime.now(timezone.utc).isoformat(),
            'appropriateness_score': appropriateness_score,
            'efficiency_score': efficiency_score,
            'risk_awareness_score': risk_awareness_score,
            'strategic_thinking_score': strategic_thinking_score,
            'overall_score': overall_score,
            'feedback': feedback,
            'recommendations': self._generate_recommendations(decision, overall_score)
        }
        
        # Store evaluation for learning and improvement
        self.decision_history.append(evaluation_result)
        
        return evaluation_result
    
    async def _evaluate_appropriateness(self, 
                                      decision: ToolSelectionDecision,
                                      config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the appropriateness of the tool selection for the given context.
        """
        
        selected_tool = self.tool_registry.tools.get(decision.selected_tool)
        if not selected_tool:
            return {
                'score': 0.0,
                'details': {'error': 'Selected tool not found in registry'},
                'breakdown': {}
            }
        
        # Evaluate task alignment
        task_alignment_score = self._evaluate_task_alignment(decision, selected_tool)
        
        # Evaluate context suitability
        context_suitability_score = self._evaluate_context_suitability(decision, selected_tool)
        
        # Evaluate capability match
        capability_match_score = self._evaluate_capability_match(decision, selected_tool)
        
        # Calculate weighted appropriateness score
        criteria = self.evaluation_metrics["appropriateness"]["scoring_criteria"]
        appropriateness_score = (
            (task_alignment_score * criteria["task_alignment"]) +
            (context_suitability_score * criteria["context_suitability"]) +
            (capability_match_score * criteria["capability_match"])
        )
        
        return {
            'score': appropriateness_score,
            'breakdown': {
                'task_alignment': task_alignment_score,
                'context_suitability': context_suitability_score,
                'capability_match': capability_match_score
            },
            'details': {
                'selected_tool_category': selected_tool.category.value,
                'selected_tool_complexity': selected_tool.complexity.value,
                'context_match_analysis': self._analyze_context_match(decision, selected_tool)
            }
        }
    
    def _evaluate_task_alignment(self, 
                               decision: ToolSelectionDecision,
                               selected_tool: ToolDefinition) -> float:
        """
        Evaluate how well the selected tool aligns with the task requirements.
        """
        
        # This would typically involve more sophisticated analysis
        # For this exercise, we'll use simplified heuristics
        
        context = decision.context
        tool_category = selected_tool.category
        
        # Define task-category alignment scores
        alignment_scores = {
            ToolCategory.MARKET_DATA: {
                'data_gathering': 0.9,
                'analysis_preparation': 0.8,
                'decision_support': 0.7
            },
            ToolCategory.TRADING_EXECUTION: {
                'order_placement': 0.95,
                'portfolio_modification': 0.9,
                'execution_management': 0.85
            },
            ToolCategory.RISK_ASSESSMENT: {
                'risk_analysis': 0.95,
                'compliance_check': 0.8,
                'decision_validation': 0.85
            }
        }
        
        # Infer task type from context and reasoning
        inferred_task = self._infer_task_type(decision)
        
        category_scores = alignment_scores.get(tool_category, {})
        return category_scores.get(inferred_task, 0.5)  # Default moderate score
    
    def _infer_task_type(self, decision: ToolSelectionDecision) -> str:
        """
        Infer the task type from the decision context and reasoning.
        """
        
        reasoning = decision.selection_reasoning.lower()
        
        if any(keyword in reasoning for keyword in ['data', 'quote', 'price', 'market']):
            return 'data_gathering'
        elif any(keyword in reasoning for keyword in ['order', 'buy', 'sell', 'execute']):
            return 'order_placement'
        elif any(keyword in reasoning for keyword in ['risk', 'analysis', 'assess']):
            return 'risk_analysis'
        else:
            return 'general_task'
    
    def _evaluate_context_suitability(self, 
                                    decision: ToolSelectionDecision,
                                    selected_tool: ToolDefinition) -> float:
        """
        Evaluate how suitable the tool is for the current trading context.
        """
        
        context = decision.context
        tool_id = selected_tool.tool_id
        
        # Get context preferences for this tool
        context_prefs = self.tool_registry.context_preferences
        
        suitability_score = 0.5  # Base score
        
        # Market conditions suitability
        market_prefs = context_prefs.get("market_conditions", {}).get(context.market_conditions, {})
        if tool_id in market_prefs:
            suitability_score += market_prefs[tool_id] * 0.3
        
        # Time sensitivity suitability
        time_prefs = context_prefs.get("time_sensitivity", {}).get(context.time_sensitivity, {})
        if tool_id in time_prefs:
            suitability_score += time_prefs[tool_id] * 0.3
        
        # Risk tolerance suitability
        risk_prefs = context_prefs.get("risk_tolerance", {}).get(context.risk_tolerance, {})
        if tool_id in risk_prefs:
            suitability_score += risk_prefs[tool_id] * 0.4
        
        return min(1.0, suitability_score)
    
    def _evaluate_capability_match(self, 
                                 decision: ToolSelectionDecision,
                                 selected_tool: ToolDefinition) -> float:
        """
        Evaluate how well the tool's capabilities match the requirements.
        """
        
        # Evaluate based on tool complexity vs. task complexity
        context = decision.context
        tool_complexity = selected_tool.complexity
        
        # Infer required complexity from context
        required_complexity = self._infer_required_complexity(context)
        
        # Calculate capability match score
        complexity_mapping = {
            ToolComplexity.SIMPLE: 1,
            ToolComplexity.MODERATE: 2,
            ToolComplexity.COMPLEX: 3,
            ToolComplexity.CRITICAL: 4
        }
        
        tool_level = complexity_mapping[tool_complexity]
        required_level = complexity_mapping[required_complexity]
        
        # Perfect match gets highest score, over/under-engineering gets lower scores
        if tool_level == required_level:
            return 1.0
        elif abs(tool_level - required_level) == 1:
            return 0.8
        elif abs(tool_level - required_level) == 2:
            return 0.6
        else:
            return 0.4
    
    def _infer_required_complexity(self, context: TradingContext) -> ToolComplexity:
        """
        Infer the required tool complexity based on trading context.
        """
        
        # High portfolio value or volatile markets require more sophisticated tools
        if context.portfolio_value > 10000000 or context.market_conditions == "volatile":
            return ToolComplexity.COMPLEX
        elif context.portfolio_value > 1000000 or context.risk_tolerance == "conservative":
            return ToolComplexity.MODERATE
        else:
            return ToolComplexity.SIMPLE
    
    async def _evaluate_efficiency(self, 
                                 decision: ToolSelectionDecision,
                                 config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the efficiency of the tool selection.
        """
        
        selected_tool = self.tool_registry.tools[decision.selected_tool]
        
        # Evaluate cost optimization
        cost_optimization_score = self._evaluate_cost_optimization(decision, selected_tool)
        
        # Evaluate latency optimization
        latency_optimization_score = self._evaluate_latency_optimization(decision, selected_tool)
        
        # Evaluate resource utilization
        resource_utilization_score = self._evaluate_resource_utilization(decision, selected_tool)
        
        # Calculate weighted efficiency score
        criteria = self.evaluation_metrics["efficiency"]["scoring_criteria"]
        efficiency_score = (
            (cost_optimization_score * criteria["cost_optimization"]) +
            (latency_optimization_score * criteria["latency_optimization"]) +
            (resource_utilization_score * criteria["resource_utilization"])
        )
        
        return {
            'score': efficiency_score,
            'breakdown': {
                'cost_optimization': cost_optimization_score,
                'latency_optimization': latency_optimization_score,
                'resource_utilization': resource_utilization_score
            },
            'details': {
                'tool_cost': selected_tool.cost_per_call,
                'tool_latency': selected_tool.average_latency,
                'alternative_analysis': self._analyze_alternatives(decision)
            }
        }
    
    def _evaluate_cost_optimization(self, 
                                  decision: ToolSelectionDecision,
                                  selected_tool: ToolDefinition) -> float:
        """
        Evaluate cost optimization in tool selection.
        """
        
        # Compare cost with alternatives
        alternative_costs = []
        for alt_tool_id in decision.alternative_tools:
            alt_tool = self.tool_registry.tools.get(alt_tool_id)
            if alt_tool and alt_tool.category == selected_tool.category:
                alternative_costs.append(alt_tool.cost_per_call)
        
        if not alternative_costs:
            return 0.8  # No alternatives to compare, assume reasonable
        
        selected_cost = selected_tool.cost_per_call
        min_cost = min(alternative_costs)
        max_cost = max(alternative_costs)
        
        if max_cost == min_cost:
            return 1.0  # All tools have same cost
        
        # Score based on relative cost position
        cost_position = (max_cost - selected_cost) / (max_cost - min_cost)
        return cost_position
    
    def _evaluate_latency_optimization(self, 
                                     decision: ToolSelectionDecision,
                                     selected_tool: ToolDefinition) -> float:
        """
        Evaluate latency optimization in tool selection.
        """
        
        context = decision.context
        selected_latency = selected_tool.average_latency
        
        # Time sensitivity affects latency importance
        if context.time_sensitivity == "urgent":
            # For urgent tasks, lower latency is critical
            if selected_latency <= 0.5:
                return 1.0
            elif selected_latency <= 1.0:
                return 0.8
            elif selected_latency <= 2.0:
                return 0.6
            else:
                return 0.4
        else:
            # For normal tasks, latency is less critical
            if selected_latency <= 2.0:
                return 1.0
            elif selected_latency <= 5.0:
                return 0.8
            else:
                return 0.6
    
    def _evaluate_resource_utilization(self, 
                                     decision: ToolSelectionDecision,
                                     selected_tool: ToolDefinition) -> float:
        """
        Evaluate resource utilization efficiency.
        """
        
        # Consider tool reliability and permissions
        reliability_score = selected_tool.reliability_score
        
        # Check if agent has required permissions
        # (In a real implementation, this would check actual permissions)
        has_permissions = True  # Simplified for exercise
        
        if not has_permissions:
            return 0.0
        
        # Higher reliability indicates better resource utilization
        return reliability_score
    
    def _analyze_alternatives(self, decision: ToolSelectionDecision) -> Dict[str, Any]:
        """
        Analyze alternative tools that could have been selected.
        """
        
        alternatives_analysis = []
        
        for alt_tool_id in decision.alternative_tools:
            alt_tool = self.tool_registry.tools.get(alt_tool_id)
            if alt_tool:
                alternatives_analysis.append({
                    'tool_id': alt_tool_id,
                    'cost': alt_tool.cost_per_call,
                    'latency': alt_tool.average_latency,
                    'reliability': alt_tool.reliability_score,
                    'complexity': alt_tool.complexity.value
                })
        
        return {
            'alternatives_considered': len(decision.alternative_tools),
            'alternatives_analysis': alternatives_analysis,
            'selection_rationale': decision.selection_reasoning
        }
    
    def _calculate_overall_score(self, 
                               appropriateness_score: Dict[str, Any],
                               efficiency_score: Dict[str, Any],
                               risk_awareness_score: Dict[str, Any],
                               strategic_thinking_score: Dict[str, Any]) -> float:
        """
        Calculate overall function selection quality score.
        """
        
        metrics = self.evaluation_metrics
        
        overall_score = (
            (appropriateness_score['score'] * metrics['appropriateness']['weight']) +
            (efficiency_score['score'] * metrics['efficiency']['weight']) +
            (risk_awareness_score['score'] * metrics['risk_awareness']['weight']) +
            (strategic_thinking_score['score'] * metrics['strategic_thinking']['weight'])
        )
        
        return overall_score
    
    async def _generate_selection_feedback(self, 
                                         decision: ToolSelectionDecision,
                                         appropriateness_score: Dict[str, Any],
                                         efficiency_score: Dict[str, Any],
                                         risk_awareness_score: Dict[str, Any],
                                         strategic_thinking_score: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate detailed feedback on the function selection decision.
        """
        
        feedback = {
            'strengths': [],
            'weaknesses': [],
            'suggestions': [],
            'overall_assessment': ''
        }
        
        # Analyze strengths
        if appropriateness_score['score'] >= 0.8:
            feedback['strengths'].append("Excellent tool-task alignment")
        if efficiency_score['score'] >= 0.8:
            feedback['strengths'].append("Efficient tool selection with good cost-performance balance")
        if risk_awareness_score['score'] >= 0.8:
            feedback['strengths'].append("Strong risk awareness in tool selection")
        
        # Analyze weaknesses
        if appropriateness_score['score'] < 0.6:
            feedback['weaknesses'].append("Tool selection may not be optimal for the given task")
        if efficiency_score['score'] < 0.6:
            feedback['weaknesses'].append("Tool selection could be more cost-effective or faster")
        if risk_awareness_score['score'] < 0.6:
            feedback['weaknesses'].append("Insufficient consideration of risk factors")
        
        # Generate suggestions
        if efficiency_score['breakdown']['cost_optimization'] < 0.7:
            feedback['suggestions'].append("Consider lower-cost alternatives for similar functionality")
        if efficiency_score['breakdown']['latency_optimization'] < 0.7:
            feedback['suggestions'].append("Evaluate faster tools for time-sensitive operations")
        
        # Overall assessment
        overall_score = self._calculate_overall_score(
            appropriateness_score, efficiency_score, risk_awareness_score, strategic_thinking_score
        )
        
        if overall_score >= 0.9:
            feedback['overall_assessment'] = "Excellent function selection with strong performance across all dimensions"
        elif overall_score >= 0.8:
            feedback['overall_assessment'] = "Good function selection with minor areas for improvement"
        elif overall_score >= 0.7:
            feedback['overall_assessment'] = "Adequate function selection with some optimization opportunities"
        else:
            feedback['overall_assessment'] = "Function selection needs significant improvement"
        
        return feedback
```

### 1.2 Implementation Task

Your task is to implement the missing evaluation methods for risk awareness and strategic thinking:

```python
async def _evaluate_risk_awareness(self, 
                                 decision: ToolSelectionDecision,
                                 config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate risk awareness in the tool selection decision.
    
    TODO: Implement this method to evaluate:
    1. Risk level appropriateness (50% weight)
    2. Compliance consideration (30% weight)  
    3. Reliability factor (20% weight)
    
    Return format should match other evaluation methods.
    """
    # Your implementation here
    pass

async def _evaluate_strategic_thinking(self, 
                                     decision: ToolSelectionDecision,
                                     config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate strategic thinking in the tool selection decision.
    
    TODO: Implement this method to evaluate:
    1. Workflow optimization (40% weight)
    2. Alternative consideration (30% weight)
    3. Future step planning (30% weight)
    
    Return format should match other evaluation methods.
    """
    # Your implementation here
    pass
```

### 1.3 Testing Your Implementation

Test your function selection evaluator with this sample scenario:

```python
async def test_function_selection_evaluator():
    """Test the function selection evaluation system."""
    
    # Initialize the system
    tool_registry = TradingToolRegistry()
    evaluator = FunctionSelectionEvaluator(tool_registry)
    
    # Create a test trading context
    context = TradingContext(
        market_conditions="volatile",
        portfolio_value=5000000.0,
        risk_tolerance="conservative",
        trading_session="market_hours",
        regulatory_constraints=["SEC_compliance", "risk_limits"],
        time_sensitivity="urgent"
    )
    
    # Create a test tool selection decision
    decision = ToolSelectionDecision(
        decision_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        context=context,
        available_tools=["real_time_quotes", "historical_data", "portfolio_risk_analysis"],
        selected_tool="real_time_quotes",
        selection_reasoning="Need current market data for volatile conditions with urgent timing",
        confidence_score=0.85,
        alternative_tools=["historical_data"]
    )
    
    # Evaluate the decision
    evaluation_result = await evaluator.evaluate_function_selection(decision)
    
    # Print results
    print("Function Selection Evaluation Results:")
    print(f"Overall Score: {evaluation_result['overall_score']:.3f}")
    print(f"Appropriateness: {evaluation_result['appropriateness_score']['score']:.3f}")
    print(f"Efficiency: {evaluation_result['efficiency_score']['score']:.3f}")
    print(f"Risk Awareness: {evaluation_result['risk_awareness_score']['score']:.3f}")
    print(f"Strategic Thinking: {evaluation_result['strategic_thinking_score']['score']:.3f}")
    print(f"Feedback: {evaluation_result['feedback']['overall_assessment']}")

# Run the test
if __name__ == "__main__":
    asyncio.run(test_function_selection_evaluator())
```

## Part 2: Parameter Validation Framework (90 minutes)

### 2.1 Understanding Parameter Validation Requirements

In this section, you'll build a comprehensive parameter validation system that ensures tool parameters are not only syntactically correct but also semantically meaningful and contextually appropriate.

```python
import re
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, date
from decimal import Decimal
import jsonschema

class ValidationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Result of parameter validation."""
    is_valid: bool
    severity: ValidationSeverity
    message: str
    field_name: Optional[str] = None
    suggested_value: Optional[Any] = None
    validation_details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ParameterValidationReport:
    """Comprehensive parameter validation report."""
    tool_id: str
    validation_timestamp: datetime
    overall_valid: bool
    validation_results: List[ValidationResult]
    schema_compliance_score: float
    semantic_validity_score: float
    contextual_appropriateness_score: float
    overall_validation_score: float

class ParameterValidator:
    """
    Comprehensive parameter validation system for trading tools.
    """
    
    def __init__(self, tool_registry: TradingToolRegistry):
        self.tool_registry = tool_registry
        self.validation_rules = self._initialize_validation_rules()
        self.semantic_validators = self._initialize_semantic_validators()
        self.contextual_validators = self._initialize_contextual_validators()
    
    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive validation rules for each tool."""
        
        return {
            "real_time_quotes": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string", "pattern": "^[A-Z]{1,5}$"},
                            "minItems": 1,
                            "maxItems": 100
                        },
                        "fields": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["price", "volume", "bid", "ask", "high", "low"]},
                            "default": ["price", "volume"]
                        },
                        "exchange": {
                            "type": "string",
                            "enum": ["NYSE", "NASDAQ", "AMEX", "OTC"]
                        }
                    },
                    "required": ["symbols"]
                },
                "semantic_rules": {
                    "symbols": ["validate_symbol_existence", "validate_trading_hours"],
                    "exchange": ["validate_exchange_availability"]
                },
                "contextual_rules": {
                    "symbols": ["validate_portfolio_relevance", "validate_risk_constraints"]
                }
            },
            "place_order": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "pattern": "^[A-Z]{1,5}$"
                        },
                        "quantity": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 1000000
                        },
                        "order_type": {
                            "type": "string",
                            "enum": ["market", "limit", "stop", "stop_limit"]
                        },
                        "price": {
                            "type": "number",
                            "minimum": 0.01,
                            "maximum": 100000
                        },
                        "time_in_force": {
                            "type": "string",
                            "enum": ["DAY", "GTC", "IOC", "FOK"]
                        }
                    },
                    "required": ["symbol", "quantity", "order_type"]
                },
                "semantic_rules": {
                    "symbol": ["validate_symbol_tradeable", "validate_market_hours"],
                    "quantity": ["validate_lot_size", "validate_liquidity"],
                    "price": ["validate_price_reasonableness", "validate_circuit_breakers"]
                },
                "contextual_rules": {
                    "quantity": ["validate_position_limits", "validate_buying_power"],
                    "price": ["validate_risk_tolerance", "validate_portfolio_impact"]
                }
            }
        }
    
    def _initialize_semantic_validators(self) -> Dict[str, Callable]:
        """Initialize semantic validation functions."""
        
        return {
            "validate_symbol_existence": self._validate_symbol_existence,
            "validate_symbol_tradeable": self._validate_symbol_tradeable,
            "validate_trading_hours": self._validate_trading_hours,
            "validate_market_hours": self._validate_market_hours,
            "validate_exchange_availability": self._validate_exchange_availability,
            "validate_lot_size": self._validate_lot_size,
            "validate_liquidity": self._validate_liquidity,
            "validate_price_reasonableness": self._validate_price_reasonableness,
            "validate_circuit_breakers": self._validate_circuit_breakers
        }
    
    def _initialize_contextual_validators(self) -> Dict[str, Callable]:
        """Initialize contextual validation functions."""
        
        return {
            "validate_portfolio_relevance": self._validate_portfolio_relevance,
            "validate_risk_constraints": self._validate_risk_constraints,
            "validate_position_limits": self._validate_position_limits,
            "validate_buying_power": self._validate_buying_power,
            "validate_risk_tolerance": self._validate_risk_tolerance,
            "validate_portfolio_impact": self._validate_portfolio_impact
        }
    
    async def validate_parameters(self, 
                                tool_id: str,
                                parameters: Dict[str, Any],
                                context: Optional[TradingContext] = None) -> ParameterValidationReport:
        """
        Comprehensive parameter validation for a tool call.
        """
        
        validation_results = []
        
        # Schema validation
        schema_results = await self._validate_schema_compliance(tool_id, parameters)
        validation_results.extend(schema_results)
        
        # Semantic validation
        semantic_results = await self._validate_semantic_correctness(tool_id, parameters)
        validation_results.extend(semantic_results)
        
        # Contextual validation (if context provided)
        contextual_results = []
        if context:
            contextual_results = await self._validate_contextual_appropriateness(
                tool_id, parameters, context
            )
            validation_results.extend(contextual_results)
        
        # Calculate validation scores
        schema_score = self._calculate_schema_compliance_score(schema_results)
        semantic_score = self._calculate_semantic_validity_score(semantic_results)
        contextual_score = self._calculate_contextual_appropriateness_score(contextual_results)
        
        # Calculate overall validation score
        overall_score = self._calculate_overall_validation_score(
            schema_score, semantic_score, contextual_score
        )
        
        # Determine overall validity
        overall_valid = all(
            result.severity != ValidationSeverity.CRITICAL 
            for result in validation_results
        )
        
        return ParameterValidationReport(
            tool_id=tool_id,
            validation_timestamp=datetime.now(timezone.utc),
            overall_valid=overall_valid,
            validation_results=validation_results,
            schema_compliance_score=schema_score,
            semantic_validity_score=semantic_score,
            contextual_appropriateness_score=contextual_score,
            overall_validation_score=overall_score
        )
    
    async def _validate_schema_compliance(self, 
                                        tool_id: str,
                                        parameters: Dict[str, Any]) -> List[ValidationResult]:
        """
        Validate parameters against the tool's schema definition.
        """
        
        validation_results = []
        
        tool_rules = self.validation_rules.get(tool_id)
        if not tool_rules:
            validation_results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"No validation rules found for tool: {tool_id}",
                field_name=None
            ))
            return validation_results
        
        schema = tool_rules.get("schema")
        if not schema:
            return validation_results
        
        try:
            # Validate against JSON schema
            jsonschema.validate(parameters, schema)
            validation_results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message="Schema validation passed",
                field_name=None
            ))
        except jsonschema.ValidationError as e:
            validation_results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Schema validation failed: {e.message}",
                field_name=e.path[0] if e.path else None,
                validation_details={"schema_error": str(e)}
            ))
        except Exception as e:
            validation_results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Schema validation error: {str(e)}",
                field_name=None
            ))
        
        return validation_results
    
    async def _validate_semantic_correctness(self, 
                                           tool_id: str,
                                           parameters: Dict[str, Any]) -> List[ValidationResult]:
        """
        Validate semantic correctness of parameters.
        """
        
        validation_results = []
        
        tool_rules = self.validation_rules.get(tool_id, {})
        semantic_rules = tool_rules.get("semantic_rules", {})
        
        for field_name, validators in semantic_rules.items():
            if field_name in parameters:
                field_value = parameters[field_name]
                
                for validator_name in validators:
                    validator_func = self.semantic_validators.get(validator_name)
                    if validator_func:
                        try:
                            result = await validator_func(field_name, field_value, parameters)
                            validation_results.append(result)
                        except Exception as e:
                            validation_results.append(ValidationResult(
                                is_valid=False,
                                severity=ValidationSeverity.ERROR,
                                message=f"Semantic validation error in {validator_name}: {str(e)}",
                                field_name=field_name
                            ))
        
        return validation_results
    
    # Semantic validation methods
    async def _validate_symbol_existence(self, 
                                       field_name: str,
                                       field_value: Any,
                                       parameters: Dict[str, Any]) -> ValidationResult:
        """Validate that trading symbols exist."""
        
        # In a real implementation, this would check against a market data service
        # For this exercise, we'll use a simplified validation
        
        if isinstance(field_value, list):
            symbols = field_value
        else:
            symbols = [field_value]
        
        # Simulate symbol validation (in reality, would call market data API)
        valid_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA"]
        invalid_symbols = [symbol for symbol in symbols if symbol not in valid_symbols]
        
        if invalid_symbols:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"Unknown symbols detected: {', '.join(invalid_symbols)}",
                field_name=field_name,
                validation_details={"invalid_symbols": invalid_symbols}
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="All symbols are valid",
            field_name=field_name
        )
    
    async def _validate_trading_hours(self, 
                                    field_name: str,
                                    field_value: Any,
                                    parameters: Dict[str, Any]) -> ValidationResult:
        """Validate that trading is allowed during current hours."""
        
        current_time = datetime.now()
        current_hour = current_time.hour
        
        # Simplified trading hours: 9:30 AM to 4:00 PM EST (14:30 to 21:00 UTC)
        if 14 <= current_hour <= 21:  # Simplified UTC check
            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message="Trading hours validation passed",
                field_name=field_name
            )
        else:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="Trading outside normal market hours",
                field_name=field_name,
                validation_details={"current_hour": current_hour}
            )
```

### 2.2 Implementation Task

Your task is to implement the missing validation methods:

```python
# TODO: Implement these semantic validation methods
async def _validate_symbol_tradeable(self, field_name: str, field_value: Any, parameters: Dict[str, Any]) -> ValidationResult:
    """Validate that symbols are currently tradeable (not halted, etc.)."""
    # Your implementation here
    pass

async def _validate_price_reasonableness(self, field_name: str, field_value: Any, parameters: Dict[str, Any]) -> ValidationResult:
    """Validate that price is reasonable compared to current market price."""
    # Your implementation here
    pass

# TODO: Implement these contextual validation methods  
async def _validate_position_limits(self, field_name: str, field_value: Any, parameters: Dict[str, Any], context: TradingContext) -> ValidationResult:
    """Validate that order quantity doesn't exceed position limits."""
    # Your implementation here
    pass

async def _validate_risk_tolerance(self, field_name: str, field_value: Any, parameters: Dict[str, Any], context: TradingContext) -> ValidationResult:
    """Validate that order aligns with risk tolerance."""
    # Your implementation here
    pass
```

## Part 3: Execution Logic Analysis (90 minutes)

### 3.1 Understanding Multi-Step Workflow Evaluation

In this section, you'll build a system to analyze the execution logic of complex multi-step trading workflows.

```python
@dataclass
class WorkflowStep:
    """Individual step in a trading workflow."""
    step_id: str
    tool_id: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    expected_outputs: List[str]
    timeout_seconds: int
    retry_policy: Dict[str, Any]

@dataclass
class WorkflowDefinition:
    """Complete workflow definition."""
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    success_criteria: Dict[str, Any]
    failure_handling: Dict[str, Any]

class ExecutionLogicAnalyzer:
    """
    Analyzer for multi-step workflow execution logic.
    """
    
    def __init__(self, tool_registry: TradingToolRegistry):
        self.tool_registry = tool_registry
        self.workflow_patterns = self._initialize_workflow_patterns()
    
    async def analyze_workflow_logic(self, 
                                   workflow: WorkflowDefinition,
                                   context: TradingContext) -> Dict[str, Any]:
        """
        Comprehensive analysis of workflow execution logic.
        """
        
        # Analyze workflow structure
        structure_analysis = await self._analyze_workflow_structure(workflow)
        
        # Analyze dependency relationships
        dependency_analysis = await self._analyze_dependencies(workflow)
        
        # Analyze error handling and resilience
        resilience_analysis = await self._analyze_resilience(workflow)
        
        # Analyze performance characteristics
        performance_analysis = await self._analyze_performance(workflow, context)
        
        # Generate optimization recommendations
        optimization_recommendations = await self._generate_optimization_recommendations(
            workflow, structure_analysis, dependency_analysis, resilience_analysis, performance_analysis
        )
        
        return {
            'workflow_id': workflow.workflow_id,
            'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
            'structure_analysis': structure_analysis,
            'dependency_analysis': dependency_analysis,
            'resilience_analysis': resilience_analysis,
            'performance_analysis': performance_analysis,
            'optimization_recommendations': optimization_recommendations,
            'overall_quality_score': self._calculate_workflow_quality_score(
                structure_analysis, dependency_analysis, resilience_analysis, performance_analysis
            )
        }
```

### 3.2 Implementation Task

Implement the workflow analysis methods:

```python
# TODO: Implement workflow analysis methods
async def _analyze_workflow_structure(self, workflow: WorkflowDefinition) -> Dict[str, Any]:
    """Analyze the structural quality of the workflow."""
    # Your implementation here
    pass

async def _analyze_dependencies(self, workflow: WorkflowDefinition) -> Dict[str, Any]:
    """Analyze dependency relationships and potential issues."""
    # Your implementation here
    pass

async def _analyze_resilience(self, workflow: WorkflowDefinition) -> Dict[str, Any]:
    """Analyze error handling and resilience characteristics."""
    # Your implementation here
    pass
```

## Part 4: Integration and Testing (60 minutes)

### 4.1 Complete System Integration

Integrate all components into a comprehensive tool calling evaluation system:

```python
class ToolCallingEvaluationSystem:
    """
    Comprehensive evaluation system for tool calling in AI agents.
    """
    
    def __init__(self):
        self.tool_registry = TradingToolRegistry()
        self.function_evaluator = FunctionSelectionEvaluator(self.tool_registry)
        self.parameter_validator = ParameterValidator(self.tool_registry)
        self.execution_analyzer = ExecutionLogicAnalyzer(self.tool_registry)
    
    async def evaluate_tool_call(self, 
                               tool_call_data: Dict[str, Any],
                               context: TradingContext) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a complete tool call.
        """
        
        # Extract components from tool call data
        decision = self._extract_selection_decision(tool_call_data)
        parameters = tool_call_data.get('parameters', {})
        workflow = tool_call_data.get('workflow')
        
        # Evaluate function selection
        selection_evaluation = await self.function_evaluator.evaluate_function_selection(decision)
        
        # Validate parameters
        parameter_validation = await self.parameter_validator.validate_parameters(
            decision.selected_tool, parameters, context
        )
        
        # Analyze execution logic (if workflow provided)
        execution_analysis = None
        if workflow:
            execution_analysis = await self.execution_analyzer.analyze_workflow_logic(workflow, context)
        
        # Generate comprehensive evaluation report
        return self._generate_comprehensive_report(
            selection_evaluation, parameter_validation, execution_analysis
        )
```

### 4.2 Testing and Validation

Create comprehensive tests for your evaluation system:

```python
async def test_comprehensive_evaluation():
    """Test the complete tool calling evaluation system."""
    
    evaluation_system = ToolCallingEvaluationSystem()
    
    # Test data
    tool_call_data = {
        'decision': {
            'decision_id': str(uuid.uuid4()),
            'timestamp': datetime.now(timezone.utc),
            'context': TradingContext(
                market_conditions="volatile",
                portfolio_value=2000000.0,
                risk_tolerance="moderate",
                trading_session="market_hours",
                regulatory_constraints=["SEC_compliance"],
                time_sensitivity="normal"
            ),
            'available_tools': ["real_time_quotes", "place_order", "portfolio_risk_analysis"],
            'selected_tool': "place_order",
            'selection_reasoning': "Execute buy order based on analysis results",
            'confidence_score': 0.9,
            'alternative_tools': ["real_time_quotes"]
        },
        'parameters': {
            'symbol': 'AAPL',
            'quantity': 100,
            'order_type': 'limit',
            'price': 150.50,
            'time_in_force': 'DAY'
        }
    }
    
    # Run evaluation
    result = await evaluation_system.evaluate_tool_call(
        tool_call_data, 
        tool_call_data['decision']['context']
    )
    
    # Analyze results
    print("Comprehensive Tool Calling Evaluation Results:")
    print(json.dumps(result, indent=2, default=str))

# Run the test
if __name__ == "__main__":
    asyncio.run(test_comprehensive_evaluation())
```

## Deliverables

1. **Complete Function Selection Evaluator** with all evaluation methods implemented
2. **Comprehensive Parameter Validator** with schema, semantic, and contextual validation
3. **Execution Logic Analyzer** for multi-step workflow evaluation
4. **Integrated Evaluation System** combining all components
5. **Test Suite** demonstrating the system's capabilities
6. **Documentation** explaining your implementation decisions and evaluation criteria

## Evaluation Criteria

Your implementation will be evaluated on:

- **Completeness**: All required methods implemented and functional
- **Accuracy**: Evaluation logic correctly assesses tool calling quality
- **Robustness**: System handles edge cases and errors gracefully
- **Performance**: Efficient evaluation with reasonable response times
- **Code Quality**: Clean, well-documented, and maintainable code
- **Testing**: Comprehensive test coverage with realistic scenarios

## Extension Opportunities

For additional challenge, consider implementing:

- **Machine Learning Integration**: Use ML models to improve evaluation accuracy
- **Real-Time Monitoring**: Add capabilities for production monitoring
- **Advanced Analytics**: Implement trend analysis and pattern recognition
- **Multi-Agent Coordination**: Extend to evaluate tool sharing between agents
- **Custom Metrics**: Add domain-specific evaluation metrics for different trading strategies

This exercise provides hands-on experience with the most advanced aspects of tool calling evaluation, preparing you for real-world implementation of sophisticated AI agent evaluation systems.

