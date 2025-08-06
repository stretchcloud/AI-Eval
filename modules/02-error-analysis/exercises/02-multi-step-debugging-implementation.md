# Exercise 2: Multi-Step Debugging Implementation

## Objective
Implement an advanced debugging framework for complex multi-step AI agent workflows with comprehensive trace analysis, dependency mapping, failure propagation detection, and optimization identification.

## Duration
4-5 hours

## Skills Developed
- Multi-step workflow trace analysis and validation
- Dependency mapping and performance bottleneck identification
- Failure detection and propagation analysis
- Pattern recognition and optimization opportunity identification

## Prerequisites
- Understanding of multi-step debugging frameworks from Section 7
- Knowledge of workflow orchestration and dependency management
- Python programming experience with async/await patterns
- Familiarity with performance profiling and optimization techniques

## Learning Outcomes
Upon completing this exercise, you will be able to:
- Design and implement comprehensive trace analysis systems for complex workflows
- Build dependency mapping frameworks with performance optimization
- Create failure detection systems with propagation analysis
- Develop pattern recognition systems for workflow optimization
- Integrate debugging frameworks into production AI agent systems

## Exercise Overview

In this exercise, you will build a production-grade multi-step debugging system for a healthcare AI diagnostic assistant. The assistant uses multiple AI agents working together to analyze patient data, consult medical databases, perform differential diagnosis, and generate treatment recommendations. Your debugging system must trace complex multi-step workflows, identify performance bottlenecks, detect failure propagation patterns, and suggest optimization opportunities.

## Part 1: Trace Analysis and Validation Framework (75 minutes)

### 1.1 Understanding the Healthcare AI Workflow Architecture

First, let's examine the healthcare diagnostic assistant's multi-step workflow:

```python
import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
import uuid
import hashlib
from collections import defaultdict, deque

class WorkflowStepType(Enum):
    DATA_INGESTION = "data_ingestion"
    PREPROCESSING = "preprocessing"
    ANALYSIS = "analysis"
    CONSULTATION = "consultation"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    REPORTING = "reporting"

class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"

class FailureType(Enum):
    TIMEOUT = "timeout"
    VALIDATION_ERROR = "validation_error"
    DEPENDENCY_FAILURE = "dependency_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    EXTERNAL_SERVICE_ERROR = "external_service_error"
    DATA_QUALITY_ISSUE = "data_quality_issue"
    LOGIC_ERROR = "logic_error"

@dataclass
class ExecutionTrace:
    """Individual execution trace for a workflow step."""
    trace_id: str
    step_id: str
    step_type: WorkflowStepType
    start_time: datetime
    end_time: Optional[datetime] = None
    status: StepStatus = StepStatus.PENDING
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_details: Optional[Dict[str, Any]] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    dependencies_resolved: List[str] = field(default_factory=list)
    dependencies_failed: List[str] = field(default_factory=list)

@dataclass
class WorkflowExecution:
    """Complete workflow execution with all traces."""
    execution_id: str
    workflow_name: str
    patient_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: StepStatus = StepStatus.PENDING
    traces: List[ExecutionTrace] = field(default_factory=list)
    global_context: Dict[str, Any] = field(default_factory=dict)
    performance_summary: Dict[str, float] = field(default_factory=dict)

@dataclass
class DependencyRelationship:
    """Dependency relationship between workflow steps."""
    source_step: str
    target_step: str
    dependency_type: str
    data_flow: List[str]
    timing_constraint: Optional[float] = None
    criticality: str = "normal"

class HealthcareDiagnosticWorkflow:
    """
    Healthcare diagnostic workflow with complex multi-step processing.
    """
    
    def __init__(self):
        self.workflow_definition = self._define_diagnostic_workflow()
        self.execution_history = []
        self.performance_baselines = self._initialize_performance_baselines()
    
    def _define_diagnostic_workflow(self) -> Dict[str, Any]:
        """Define the complete healthcare diagnostic workflow."""
        
        return {
            "workflow_name": "comprehensive_diagnostic_analysis",
            "description": "Multi-step AI diagnostic workflow for complex medical cases",
            "steps": {
                "patient_data_ingestion": {
                    "type": WorkflowStepType.DATA_INGESTION,
                    "description": "Ingest and validate patient medical records",
                    "dependencies": [],
                    "expected_duration": 2.0,
                    "resource_requirements": {"memory": 512, "cpu": 0.5},
                    "failure_tolerance": "low"
                },
                "medical_history_preprocessing": {
                    "type": WorkflowStepType.PREPROCESSING,
                    "description": "Process and structure medical history data",
                    "dependencies": ["patient_data_ingestion"],
                    "expected_duration": 5.0,
                    "resource_requirements": {"memory": 1024, "cpu": 1.0},
                    "failure_tolerance": "medium"
                },
                "symptom_analysis": {
                    "type": WorkflowStepType.ANALYSIS,
                    "description": "Analyze patient symptoms using AI models",
                    "dependencies": ["medical_history_preprocessing"],
                    "expected_duration": 8.0,
                    "resource_requirements": {"memory": 2048, "cpu": 2.0, "gpu": 1.0},
                    "failure_tolerance": "low"
                },
                "lab_results_analysis": {
                    "type": WorkflowStepType.ANALYSIS,
                    "description": "Analyze laboratory test results",
                    "dependencies": ["patient_data_ingestion"],
                    "expected_duration": 6.0,
                    "resource_requirements": {"memory": 1024, "cpu": 1.5},
                    "failure_tolerance": "medium"
                },
                "medical_literature_consultation": {
                    "type": WorkflowStepType.CONSULTATION,
                    "description": "Consult medical literature and guidelines",
                    "dependencies": ["symptom_analysis", "lab_results_analysis"],
                    "expected_duration": 12.0,
                    "resource_requirements": {"memory": 1536, "cpu": 1.0},
                    "failure_tolerance": "high"
                },
                "differential_diagnosis": {
                    "type": WorkflowStepType.SYNTHESIS,
                    "description": "Generate differential diagnosis based on all analyses",
                    "dependencies": ["symptom_analysis", "lab_results_analysis", "medical_literature_consultation"],
                    "expected_duration": 10.0,
                    "resource_requirements": {"memory": 2048, "cpu": 2.5, "gpu": 0.5},
                    "failure_tolerance": "low"
                },
                "treatment_recommendation": {
                    "type": WorkflowStepType.SYNTHESIS,
                    "description": "Generate treatment recommendations",
                    "dependencies": ["differential_diagnosis"],
                    "expected_duration": 8.0,
                    "resource_requirements": {"memory": 1536, "cpu": 2.0},
                    "failure_tolerance": "low"
                },
                "clinical_validation": {
                    "type": WorkflowStepType.VALIDATION,
                    "description": "Validate recommendations against clinical guidelines",
                    "dependencies": ["treatment_recommendation"],
                    "expected_duration": 5.0,
                    "resource_requirements": {"memory": 1024, "cpu": 1.0},
                    "failure_tolerance": "low"
                },
                "report_generation": {
                    "type": WorkflowStepType.REPORTING,
                    "description": "Generate comprehensive diagnostic report",
                    "dependencies": ["clinical_validation"],
                    "expected_duration": 3.0,
                    "resource_requirements": {"memory": 512, "cpu": 0.5},
                    "failure_tolerance": "medium"
                }
            },
            "success_criteria": {
                "total_duration_max": 60.0,
                "critical_steps_success_rate": 0.95,
                "data_quality_threshold": 0.9
            }
        }
    
    def _initialize_performance_baselines(self) -> Dict[str, Dict[str, float]]:
        """Initialize performance baselines for each workflow step."""
        
        return {
            "patient_data_ingestion": {
                "duration_p50": 1.8,
                "duration_p95": 3.2,
                "memory_usage_avg": 480,
                "cpu_usage_avg": 0.45,
                "success_rate": 0.98
            },
            "medical_history_preprocessing": {
                "duration_p50": 4.5,
                "duration_p95": 7.8,
                "memory_usage_avg": 950,
                "cpu_usage_avg": 0.85,
                "success_rate": 0.96
            },
            "symptom_analysis": {
                "duration_p50": 7.2,
                "duration_p95": 12.5,
                "memory_usage_avg": 1950,
                "cpu_usage_avg": 1.8,
                "gpu_usage_avg": 0.9,
                "success_rate": 0.94
            },
            "lab_results_analysis": {
                "duration_p50": 5.5,
                "duration_p95": 9.2,
                "memory_usage_avg": 980,
                "cpu_usage_avg": 1.3,
                "success_rate": 0.97
            },
            "medical_literature_consultation": {
                "duration_p50": 11.0,
                "duration_p95": 18.5,
                "memory_usage_avg": 1450,
                "cpu_usage_avg": 0.9,
                "success_rate": 0.92
            },
            "differential_diagnosis": {
                "duration_p50": 9.2,
                "duration_p95": 15.8,
                "memory_usage_avg": 1980,
                "cpu_usage_avg": 2.2,
                "gpu_usage_avg": 0.45,
                "success_rate": 0.95
            },
            "treatment_recommendation": {
                "duration_p50": 7.5,
                "duration_p95": 12.2,
                "memory_usage_avg": 1480,
                "cpu_usage_avg": 1.8,
                "success_rate": 0.96
            },
            "clinical_validation": {
                "duration_p50": 4.8,
                "duration_p95": 7.5,
                "memory_usage_avg": 980,
                "cpu_usage_avg": 0.9,
                "success_rate": 0.98
            },
            "report_generation": {
                "duration_p50": 2.8,
                "duration_p95": 4.5,
                "memory_usage_avg": 490,
                "cpu_usage_avg": 0.45,
                "success_rate": 0.99
            }
        }

class TraceAnalysisEngine:
    """
    Comprehensive trace analysis engine for multi-step workflows.
    """
    
    def __init__(self, workflow: HealthcareDiagnosticWorkflow):
        self.workflow = workflow
        self.trace_validators = self._initialize_trace_validators()
        self.integrity_checkers = self._initialize_integrity_checkers()
        self.anomaly_detectors = self._initialize_anomaly_detectors()
    
    def _initialize_trace_validators(self) -> Dict[str, Any]:
        """Initialize trace validation rules and functions."""
        
        return {
            "temporal_consistency": {
                "description": "Validate temporal ordering and timing constraints",
                "validator": self._validate_temporal_consistency,
                "severity": "critical"
            },
            "data_flow_integrity": {
                "description": "Validate data flow between workflow steps",
                "validator": self._validate_data_flow_integrity,
                "severity": "critical"
            },
            "resource_constraints": {
                "description": "Validate resource usage within acceptable limits",
                "validator": self._validate_resource_constraints,
                "severity": "warning"
            },
            "dependency_resolution": {
                "description": "Validate proper dependency resolution",
                "validator": self._validate_dependency_resolution,
                "severity": "critical"
            },
            "performance_baselines": {
                "description": "Validate performance against established baselines",
                "validator": self._validate_performance_baselines,
                "severity": "warning"
            }
        }
    
    def _initialize_integrity_checkers(self) -> Dict[str, Any]:
        """Initialize integrity checking functions."""
        
        return {
            "trace_completeness": self._check_trace_completeness,
            "data_consistency": self._check_data_consistency,
            "execution_ordering": self._check_execution_ordering,
            "error_propagation": self._check_error_propagation
        }
    
    def _initialize_anomaly_detectors(self) -> Dict[str, Any]:
        """Initialize anomaly detection algorithms."""
        
        return {
            "performance_anomalies": self._detect_performance_anomalies,
            "resource_anomalies": self._detect_resource_anomalies,
            "pattern_anomalies": self._detect_pattern_anomalies,
            "failure_anomalies": self._detect_failure_anomalies
        }
    
    async def analyze_execution_traces(self, 
                                     execution: WorkflowExecution,
                                     analysis_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of workflow execution traces.
        """
        
        config = analysis_config or {}
        
        # Validate trace integrity
        integrity_results = await self._validate_trace_integrity(execution, config)
        
        # Analyze execution flow
        flow_analysis = await self._analyze_execution_flow(execution, config)
        
        # Detect anomalies
        anomaly_analysis = await self._detect_execution_anomalies(execution, config)
        
        # Analyze performance characteristics
        performance_analysis = await self._analyze_performance_characteristics(execution, config)
        
        # Generate insights and recommendations
        insights = await self._generate_trace_insights(
            execution, integrity_results, flow_analysis, anomaly_analysis, performance_analysis
        )
        
        return {
            'execution_id': execution.execution_id,
            'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
            'integrity_results': integrity_results,
            'flow_analysis': flow_analysis,
            'anomaly_analysis': anomaly_analysis,
            'performance_analysis': performance_analysis,
            'insights': insights,
            'overall_health_score': self._calculate_execution_health_score(
                integrity_results, flow_analysis, anomaly_analysis, performance_analysis
            )
        }
    
    async def _validate_trace_integrity(self, 
                                      execution: WorkflowExecution,
                                      config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the integrity of execution traces.
        """
        
        validation_results = {}
        
        for validator_name, validator_config in self.trace_validators.items():
            try:
                validator_func = validator_config["validator"]
                result = await validator_func(execution, config)
                validation_results[validator_name] = {
                    'result': result,
                    'severity': validator_config["severity"],
                    'description': validator_config["description"]
                }
            except Exception as e:
                validation_results[validator_name] = {
                    'result': {'valid': False, 'error': str(e)},
                    'severity': 'error',
                    'description': f"Validation error in {validator_name}"
                }
        
        return validation_results
    
    async def _validate_temporal_consistency(self, 
                                           execution: WorkflowExecution,
                                           config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate temporal consistency of execution traces.
        """
        
        issues = []
        
        # Check that start times are before end times
        for trace in execution.traces:
            if trace.end_time and trace.start_time >= trace.end_time:
                issues.append({
                    'type': 'invalid_time_range',
                    'step_id': trace.step_id,
                    'start_time': trace.start_time.isoformat(),
                    'end_time': trace.end_time.isoformat()
                })
        
        # Check dependency ordering
        step_completion_times = {}
        for trace in execution.traces:
            if trace.end_time and trace.status == StepStatus.COMPLETED:
                step_completion_times[trace.step_id] = trace.end_time
        
        workflow_steps = self.workflow.workflow_definition["steps"]
        for step_id, step_config in workflow_steps.items():
            if step_id in step_completion_times:
                step_completion_time = step_completion_times[step_id]
                
                for dependency in step_config.get("dependencies", []):
                    if dependency in step_completion_times:
                        dep_completion_time = step_completion_times[dependency]
                        if dep_completion_time >= step_completion_time:
                            issues.append({
                                'type': 'dependency_timing_violation',
                                'step_id': step_id,
                                'dependency': dependency,
                                'step_completion': step_completion_time.isoformat(),
                                'dependency_completion': dep_completion_time.isoformat()
                            })
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'total_issues': len(issues)
        }
    
    async def _validate_data_flow_integrity(self, 
                                          execution: WorkflowExecution,
                                          config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data flow integrity between workflow steps.
        """
        
        issues = []
        data_flow_map = {}
        
        # Build data flow map
        for trace in execution.traces:
            data_flow_map[trace.step_id] = {
                'inputs': trace.input_data,
                'outputs': trace.output_data,
                'status': trace.status
            }
        
        # Validate data flow between dependent steps
        workflow_steps = self.workflow.workflow_definition["steps"]
        for step_id, step_config in workflow_steps.items():
            if step_id in data_flow_map:
                step_data = data_flow_map[step_id]
                
                for dependency in step_config.get("dependencies", []):
                    if dependency in data_flow_map:
                        dep_data = data_flow_map[dependency]
                        
                        # Check if dependency produced required outputs
                        if dep_data['status'] == StepStatus.COMPLETED:
                            if not dep_data['outputs']:
                                issues.append({
                                    'type': 'missing_dependency_output',
                                    'step_id': step_id,
                                    'dependency': dependency
                                })
                        
                        # Check if step received required inputs
                        if step_data['status'] != StepStatus.FAILED:
                            if not step_data['inputs']:
                                issues.append({
                                    'type': 'missing_step_input',
                                    'step_id': step_id,
                                    'dependency': dependency
                                })
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'data_flow_map': data_flow_map,
            'total_issues': len(issues)
        }
    
    async def _validate_resource_constraints(self, 
                                           execution: WorkflowExecution,
                                           config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate resource usage within acceptable constraints.
        """
        
        issues = []
        resource_violations = []
        
        workflow_steps = self.workflow.workflow_definition["steps"]
        
        for trace in execution.traces:
            step_config = workflow_steps.get(trace.step_id, {})
            resource_requirements = step_config.get("resource_requirements", {})
            
            # Check memory usage
            if "memory" in resource_requirements and "memory_usage" in trace.resource_usage:
                required_memory = resource_requirements["memory"]
                actual_memory = trace.resource_usage["memory_usage"]
                
                if actual_memory > required_memory * 1.5:  # 50% tolerance
                    resource_violations.append({
                        'type': 'memory_overuse',
                        'step_id': trace.step_id,
                        'required': required_memory,
                        'actual': actual_memory,
                        'ratio': actual_memory / required_memory
                    })
            
            # Check CPU usage
            if "cpu" in resource_requirements and "cpu_usage" in trace.resource_usage:
                required_cpu = resource_requirements["cpu"]
                actual_cpu = trace.resource_usage["cpu_usage"]
                
                if actual_cpu > required_cpu * 1.3:  # 30% tolerance
                    resource_violations.append({
                        'type': 'cpu_overuse',
                        'step_id': trace.step_id,
                        'required': required_cpu,
                        'actual': actual_cpu,
                        'ratio': actual_cpu / required_cpu
                    })
        
        return {
            'valid': len(resource_violations) == 0,
            'violations': resource_violations,
            'total_violations': len(resource_violations)
        }
    
    async def _validate_dependency_resolution(self, 
                                            execution: WorkflowExecution,
                                            config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate proper dependency resolution.
        """
        
        issues = []
        
        for trace in execution.traces:
            # Check if all dependencies were resolved
            workflow_steps = self.workflow.workflow_definition["steps"]
            step_config = workflow_steps.get(trace.step_id, {})
            required_dependencies = set(step_config.get("dependencies", []))
            resolved_dependencies = set(trace.dependencies_resolved)
            failed_dependencies = set(trace.dependencies_failed)
            
            # Missing dependencies
            missing_dependencies = required_dependencies - resolved_dependencies - failed_dependencies
            if missing_dependencies:
                issues.append({
                    'type': 'missing_dependencies',
                    'step_id': trace.step_id,
                    'missing': list(missing_dependencies)
                })
            
            # Failed dependencies for critical steps
            if failed_dependencies and step_config.get("failure_tolerance") == "low":
                issues.append({
                    'type': 'critical_dependency_failure',
                    'step_id': trace.step_id,
                    'failed_dependencies': list(failed_dependencies)
                })
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'total_issues': len(issues)
        }
    
    async def _validate_performance_baselines(self, 
                                            execution: WorkflowExecution,
                                            config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate performance against established baselines.
        """
        
        performance_issues = []
        
        for trace in execution.traces:
            if trace.step_id in self.workflow.performance_baselines:
                baseline = self.workflow.performance_baselines[trace.step_id]
                
                # Check duration
                if trace.end_time:
                    actual_duration = (trace.end_time - trace.start_time).total_seconds()
                    baseline_p95 = baseline.get("duration_p95", float('inf'))
                    
                    if actual_duration > baseline_p95:
                        performance_issues.append({
                            'type': 'duration_exceeded',
                            'step_id': trace.step_id,
                            'actual_duration': actual_duration,
                            'baseline_p95': baseline_p95,
                            'ratio': actual_duration / baseline_p95
                        })
                
                # Check resource usage
                for resource, baseline_value in baseline.items():
                    if resource.endswith('_avg') and resource in trace.resource_usage:
                        actual_value = trace.resource_usage[resource]
                        if actual_value > baseline_value * 1.5:  # 50% tolerance
                            performance_issues.append({
                                'type': 'resource_baseline_exceeded',
                                'step_id': trace.step_id,
                                'resource': resource,
                                'actual': actual_value,
                                'baseline': baseline_value,
                                'ratio': actual_value / baseline_value
                            })
        
        return {
            'valid': len(performance_issues) == 0,
            'issues': performance_issues,
            'total_issues': len(performance_issues)
        }
```

### 1.2 Implementation Task

Your task is to implement the missing analysis methods:

```python
async def _analyze_execution_flow(self, 
                                execution: WorkflowExecution,
                                config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the execution flow and identify patterns.
    
    TODO: Implement this method to analyze:
    1. Execution path analysis and optimization opportunities
    2. Parallel execution potential and bottleneck identification
    3. Critical path analysis and timing optimization
    4. Flow efficiency metrics and improvement recommendations
    
    Return comprehensive flow analysis results.
    """
    # Your implementation here
    pass

async def _detect_execution_anomalies(self, 
                                    execution: WorkflowExecution,
                                    config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect anomalies in workflow execution.
    
    TODO: Implement this method to detect:
    1. Performance anomalies (unusual timing, resource usage)
    2. Pattern anomalies (unexpected execution patterns)
    3. Failure anomalies (unusual failure patterns)
    4. Resource anomalies (unexpected resource consumption)
    
    Return comprehensive anomaly detection results.
    """
    # Your implementation here
    pass

async def _analyze_performance_characteristics(self, 
                                             execution: WorkflowExecution,
                                             config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze performance characteristics of the workflow execution.
    
    TODO: Implement this method to analyze:
    1. Overall workflow performance metrics
    2. Individual step performance analysis
    3. Resource utilization efficiency
    4. Scalability and optimization opportunities
    
    Return comprehensive performance analysis results.
    """
    # Your implementation here
    pass
```

### 1.3 Testing Your Implementation

Test your trace analysis engine with this sample scenario:

```python
async def test_trace_analysis_engine():
    """Test the trace analysis engine with sample execution data."""
    
    # Initialize the system
    workflow = HealthcareDiagnosticWorkflow()
    trace_analyzer = TraceAnalysisEngine(workflow)
    
    # Create sample execution with traces
    execution = WorkflowExecution(
        execution_id=str(uuid.uuid4()),
        workflow_name="comprehensive_diagnostic_analysis",
        patient_id="patient_12345",
        start_time=datetime.now(timezone.utc) - timedelta(minutes=30),
        end_time=datetime.now(timezone.utc),
        status=StepStatus.COMPLETED
    )
    
    # Add sample traces
    base_time = execution.start_time
    
    traces = [
        ExecutionTrace(
            trace_id=str(uuid.uuid4()),
            step_id="patient_data_ingestion",
            step_type=WorkflowStepType.DATA_INGESTION,
            start_time=base_time,
            end_time=base_time + timedelta(seconds=2.1),
            status=StepStatus.COMPLETED,
            input_data={"patient_id": "patient_12345"},
            output_data={"medical_records": "processed"},
            performance_metrics={"duration": 2.1},
            resource_usage={"memory_usage": 490, "cpu_usage": 0.48},
            dependencies_resolved=[]
        ),
        ExecutionTrace(
            trace_id=str(uuid.uuid4()),
            step_id="medical_history_preprocessing",
            step_type=WorkflowStepType.PREPROCESSING,
            start_time=base_time + timedelta(seconds=2.1),
            end_time=base_time + timedelta(seconds=7.3),
            status=StepStatus.COMPLETED,
            input_data={"medical_records": "processed"},
            output_data={"structured_history": "completed"},
            performance_metrics={"duration": 5.2},
            resource_usage={"memory_usage": 1050, "cpu_usage": 0.92},
            dependencies_resolved=["patient_data_ingestion"]
        ),
        ExecutionTrace(
            trace_id=str(uuid.uuid4()),
            step_id="symptom_analysis",
            step_type=WorkflowStepType.ANALYSIS,
            start_time=base_time + timedelta(seconds=7.3),
            end_time=base_time + timedelta(seconds=16.8),
            status=StepStatus.COMPLETED,
            input_data={"structured_history": "completed"},
            output_data={"symptom_analysis": "completed"},
            performance_metrics={"duration": 9.5},
            resource_usage={"memory_usage": 2100, "cpu_usage": 1.95, "gpu_usage": 0.88},
            dependencies_resolved=["medical_history_preprocessing"]
        )
    ]
    
    execution.traces = traces
    
    # Run analysis
    analysis_result = await trace_analyzer.analyze_execution_traces(execution)
    
    # Print results
    print("Trace Analysis Results:")
    print(f"Overall Health Score: {analysis_result['overall_health_score']:.3f}")
    print(f"Integrity Issues: {len(analysis_result['integrity_results'])}")
    print(f"Flow Analysis: {analysis_result['flow_analysis']}")
    print(f"Anomalies Detected: {analysis_result['anomaly_analysis']}")

# Run the test
if __name__ == "__main__":
    asyncio.run(test_trace_analysis_engine())
```

## Part 2: Dependency and Performance Analysis (75 minutes)

### 2.1 Understanding Dependency Mapping and Performance Optimization

In this section, you'll build a system to map dependencies and identify performance optimization opportunities.

```python
class DependencyAnalyzer:
    """
    Advanced dependency analysis and performance optimization system.
    """
    
    def __init__(self, workflow: HealthcareDiagnosticWorkflow):
        self.workflow = workflow
        self.dependency_graph = self._build_dependency_graph()
        self.performance_optimizer = PerformanceOptimizer(workflow)
    
    def _build_dependency_graph(self) -> Dict[str, Any]:
        """Build comprehensive dependency graph for the workflow."""
        
        graph = {
            'nodes': {},
            'edges': [],
            'critical_path': [],
            'parallel_opportunities': []
        }
        
        workflow_steps = self.workflow.workflow_definition["steps"]
        
        # Add nodes
        for step_id, step_config in workflow_steps.items():
            graph['nodes'][step_id] = {
                'type': step_config['type'].value,
                'expected_duration': step_config['expected_duration'],
                'resource_requirements': step_config['resource_requirements'],
                'failure_tolerance': step_config['failure_tolerance'],
                'dependencies': step_config.get('dependencies', [])
            }
        
        # Add edges
        for step_id, step_config in workflow_steps.items():
            for dependency in step_config.get('dependencies', []):
                graph['edges'].append({
                    'source': dependency,
                    'target': step_id,
                    'type': 'dependency'
                })
        
        return graph
    
    async def analyze_dependencies(self, 
                                 execution: WorkflowExecution) -> Dict[str, Any]:
        """
        Comprehensive dependency analysis for workflow execution.
        """
        
        # Analyze dependency resolution patterns
        resolution_analysis = await self._analyze_dependency_resolution(execution)
        
        # Identify bottlenecks
        bottleneck_analysis = await self._identify_bottlenecks(execution)
        
        # Analyze critical path
        critical_path_analysis = await self._analyze_critical_path(execution)
        
        # Identify parallelization opportunities
        parallelization_analysis = await self._identify_parallelization_opportunities(execution)
        
        return {
            'dependency_resolution': resolution_analysis,
            'bottlenecks': bottleneck_analysis,
            'critical_path': critical_path_analysis,
            'parallelization_opportunities': parallelization_analysis,
            'optimization_recommendations': await self._generate_dependency_optimization_recommendations(
                resolution_analysis, bottleneck_analysis, critical_path_analysis, parallelization_analysis
            )
        }

class PerformanceOptimizer:
    """
    Performance optimization system for multi-step workflows.
    """
    
    def __init__(self, workflow: HealthcareDiagnosticWorkflow):
        self.workflow = workflow
        self.optimization_strategies = self._initialize_optimization_strategies()
    
    def _initialize_optimization_strategies(self) -> Dict[str, Any]:
        """Initialize performance optimization strategies."""
        
        return {
            'caching': {
                'description': 'Implement intelligent caching for repeated operations',
                'applicability': ['data_ingestion', 'consultation', 'analysis'],
                'expected_improvement': 0.3
            },
            'parallelization': {
                'description': 'Execute independent steps in parallel',
                'applicability': ['analysis', 'consultation'],
                'expected_improvement': 0.4
            },
            'resource_optimization': {
                'description': 'Optimize resource allocation and usage',
                'applicability': ['all'],
                'expected_improvement': 0.2
            },
            'pipeline_optimization': {
                'description': 'Optimize data pipeline and flow',
                'applicability': ['preprocessing', 'analysis', 'synthesis'],
                'expected_improvement': 0.25
            }
        }
```

### 2.2 Implementation Task

Implement the dependency and performance analysis methods:

```python
# TODO: Implement dependency analysis methods
async def _analyze_dependency_resolution(self, execution: WorkflowExecution) -> Dict[str, Any]:
    """Analyze dependency resolution patterns and efficiency."""
    # Your implementation here
    pass

async def _identify_bottlenecks(self, execution: WorkflowExecution) -> Dict[str, Any]:
    """Identify performance bottlenecks in the workflow."""
    # Your implementation here
    pass

async def _analyze_critical_path(self, execution: WorkflowExecution) -> Dict[str, Any]:
    """Analyze the critical path and timing optimization opportunities."""
    # Your implementation here
    pass

async def _identify_parallelization_opportunities(self, execution: WorkflowExecution) -> Dict[str, Any]:
    """Identify opportunities for parallel execution."""
    # Your implementation here
    pass
```

## Part 3: Failure Detection and Pattern Recognition (60 minutes)

### 3.1 Understanding Failure Propagation and Pattern Analysis

Build a system to detect failure patterns and propagation:

```python
class FailureAnalyzer:
    """
    Advanced failure detection and pattern analysis system.
    """
    
    def __init__(self, workflow: HealthcareDiagnosticWorkflow):
        self.workflow = workflow
        self.failure_patterns = self._initialize_failure_patterns()
        self.propagation_rules = self._initialize_propagation_rules()
    
    async def analyze_failures(self, 
                             execution: WorkflowExecution,
                             historical_executions: List[WorkflowExecution]) -> Dict[str, Any]:
        """
        Comprehensive failure analysis and pattern recognition.
        """
        
        # Analyze current execution failures
        current_failures = await self._analyze_current_failures(execution)
        
        # Analyze failure propagation
        propagation_analysis = await self._analyze_failure_propagation(execution)
        
        # Detect failure patterns across executions
        pattern_analysis = await self._detect_failure_patterns(execution, historical_executions)
        
        # Generate failure prevention recommendations
        prevention_recommendations = await self._generate_failure_prevention_recommendations(
            current_failures, propagation_analysis, pattern_analysis
        )
        
        return {
            'current_failures': current_failures,
            'propagation_analysis': propagation_analysis,
            'pattern_analysis': pattern_analysis,
            'prevention_recommendations': prevention_recommendations
        }
```

### 3.2 Implementation Task

Implement the failure analysis methods:

```python
# TODO: Implement failure analysis methods
async def _analyze_current_failures(self, execution: WorkflowExecution) -> Dict[str, Any]:
    """Analyze failures in the current execution."""
    # Your implementation here
    pass

async def _analyze_failure_propagation(self, execution: WorkflowExecution) -> Dict[str, Any]:
    """Analyze how failures propagate through the workflow."""
    # Your implementation here
    pass

async def _detect_failure_patterns(self, 
                                 execution: WorkflowExecution,
                                 historical_executions: List[WorkflowExecution]) -> Dict[str, Any]:
    """Detect patterns in failures across multiple executions."""
    # Your implementation here
    pass
```

## Part 4: Integration and Comprehensive Testing (60 minutes)

### 4.1 Complete System Integration

Integrate all components into a comprehensive multi-step debugging system:

```python
class MultiStepDebuggingSystem:
    """
    Comprehensive multi-step debugging system for AI agent workflows.
    """
    
    def __init__(self):
        self.workflow = HealthcareDiagnosticWorkflow()
        self.trace_analyzer = TraceAnalysisEngine(self.workflow)
        self.dependency_analyzer = DependencyAnalyzer(self.workflow)
        self.failure_analyzer = FailureAnalyzer(self.workflow)
        self.execution_history = []
    
    async def debug_workflow_execution(self, 
                                     execution: WorkflowExecution,
                                     debug_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Comprehensive debugging of a workflow execution.
        """
        
        config = debug_config or {}
        
        # Trace analysis
        trace_analysis = await self.trace_analyzer.analyze_execution_traces(execution, config)
        
        # Dependency analysis
        dependency_analysis = await self.dependency_analyzer.analyze_dependencies(execution)
        
        # Failure analysis
        failure_analysis = await self.failure_analyzer.analyze_failures(
            execution, self.execution_history
        )
        
        # Generate comprehensive debugging report
        debugging_report = await self._generate_debugging_report(
            execution, trace_analysis, dependency_analysis, failure_analysis
        )
        
        # Store execution for historical analysis
        self.execution_history.append(execution)
        
        return debugging_report
    
    async def _generate_debugging_report(self, 
                                       execution: WorkflowExecution,
                                       trace_analysis: Dict[str, Any],
                                       dependency_analysis: Dict[str, Any],
                                       failure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive debugging report."""
        
        return {
            'execution_summary': {
                'execution_id': execution.execution_id,
                'workflow_name': execution.workflow_name,
                'patient_id': execution.patient_id,
                'total_duration': (execution.end_time - execution.start_time).total_seconds() if execution.end_time else None,
                'status': execution.status.value,
                'steps_completed': len([t for t in execution.traces if t.status == StepStatus.COMPLETED]),
                'steps_failed': len([t for t in execution.traces if t.status == StepStatus.FAILED])
            },
            'trace_analysis': trace_analysis,
            'dependency_analysis': dependency_analysis,
            'failure_analysis': failure_analysis,
            'overall_assessment': self._generate_overall_assessment(
                trace_analysis, dependency_analysis, failure_analysis
            ),
            'action_items': self._generate_action_items(
                trace_analysis, dependency_analysis, failure_analysis
            )
        }
```

### 4.2 Testing and Validation

Create comprehensive tests for your debugging system:

```python
async def test_comprehensive_debugging():
    """Test the complete multi-step debugging system."""
    
    debugging_system = MultiStepDebuggingSystem()
    
    # Create test execution with various scenarios
    execution = create_test_execution_with_issues()
    
    # Run debugging
    debugging_report = await debugging_system.debug_workflow_execution(execution)
    
    # Analyze results
    print("Multi-Step Debugging Report:")
    print(json.dumps(debugging_report, indent=2, default=str))

def create_test_execution_with_issues():
    """Create a test execution with various issues for debugging."""
    # Implementation of test data creation
    pass

# Run the test
if __name__ == "__main__":
    asyncio.run(test_comprehensive_debugging())
```

## Deliverables

1. **Complete Trace Analysis Engine** with integrity validation and anomaly detection
2. **Comprehensive Dependency Analyzer** with bottleneck identification and optimization
3. **Advanced Failure Analyzer** with pattern recognition and prevention recommendations
4. **Integrated Debugging System** combining all components
5. **Test Suite** demonstrating the system's capabilities with realistic healthcare scenarios
6. **Documentation** explaining your implementation decisions and debugging methodologies

## Evaluation Criteria

Your implementation will be evaluated on:

- **Completeness**: All required methods implemented and functional
- **Accuracy**: Debugging logic correctly identifies issues and optimization opportunities
- **Depth**: Comprehensive analysis covering all aspects of multi-step workflows
- **Performance**: Efficient analysis with reasonable response times
- **Code Quality**: Clean, well-documented, and maintainable code
- **Testing**: Comprehensive test coverage with realistic scenarios

## Extension Opportunities

For additional challenge, consider implementing:

- **Machine Learning Integration**: Use ML models to predict failures and optimize performance
- **Real-Time Monitoring**: Add capabilities for live workflow monitoring and debugging
- **Visual Debugging**: Implement visualization tools for workflow execution and dependencies
- **Automated Optimization**: Add capabilities for automatic workflow optimization
- **Multi-Workflow Analysis**: Extend to analyze patterns across different workflow types

This exercise provides hands-on experience with the most sophisticated aspects of multi-step debugging, preparing you for real-world implementation of advanced AI agent debugging systems in production environments.

