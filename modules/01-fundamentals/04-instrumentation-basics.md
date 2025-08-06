# Instrumentation Basics: Building Observability for AI Systems

## Introduction: The Foundation of Effective Evaluation

Instrumentation forms the bedrock of any effective AI evaluation strategy. Without proper instrumentation, teams operate blindly, unable to understand how their systems behave in real-world conditions or identify opportunities for improvement. Yet many teams treat instrumentation as an afterthought, implementing minimal logging and hoping for the best.

This approach works poorly for traditional software and fails catastrophically for AI systems. The probabilistic nature of AI outputs, the context dependence of appropriate responses, and the potential for emergent behaviors mean that comprehensive instrumentation is not just helpful but essential for reliable operation.

Effective instrumentation for AI systems goes far beyond traditional logging and monitoring. It requires capturing not just what the system does, but why it does it, how users respond, and what the broader context looks like. This comprehensive approach to observability enables teams to understand system behavior, identify improvement opportunities, and respond quickly to issues.

## Core Instrumentation Principles

### Comprehensive Data Capture

The foundation of effective AI instrumentation is comprehensive data capture that goes beyond traditional system metrics to include the rich contextual information necessary for understanding AI system behavior. This comprehensive approach recognizes that AI systems are fundamentally different from traditional software and require different observability strategies.

Input capture involves recording not just the immediate user input but also the broader context that influences system behavior. For conversational AI systems, this might include conversation history, user profile information, and session context. For content generation systems, it might include style guidelines, target audience information, and content requirements.

Output capture must go beyond simply logging the final response to include intermediate steps, reasoning processes, and confidence scores where available. This detailed capture enables teams to understand not just what the system produced but how it arrived at that output, facilitating more effective debugging and improvement.

Context capture involves recording the environmental and situational factors that might influence system behavior. This includes technical context such as system load and performance metrics, but also business context such as user demographics, time of day, and feature flags that might affect system behavior.

User interaction capture tracks how users respond to system outputs, providing crucial feedback on the effectiveness and appropriateness of AI responses. This might include explicit feedback such as ratings or corrections, but also implicit feedback such as engagement patterns, follow-up questions, and task completion rates.

### Real-Time and Batch Processing

Effective AI instrumentation requires both real-time processing for immediate response and batch processing for comprehensive analysis. This dual approach enables teams to respond quickly to urgent issues while conducting thorough analysis for long-term improvement.

Real-time processing focuses on immediate detection of critical issues such as safety violations, system errors, or severe quality degradation. Real-time systems must be designed for speed and reliability, providing rapid alerts when intervention is necessary while avoiding false positives that could lead to alert fatigue.

Batch processing enables comprehensive analysis of system behavior patterns, user satisfaction trends, and improvement opportunities. Batch systems can afford to be more thorough and computationally intensive, providing deep insights that inform strategic decisions about system development and optimization.

The integration between real-time and batch processing is crucial for effective instrumentation. Real-time systems should feed data to batch processing pipelines, while batch analysis should inform real-time monitoring thresholds and alert criteria.

### Privacy and Security Considerations

AI instrumentation often involves capturing sensitive user data and system information, making privacy and security considerations paramount. Effective instrumentation strategies must balance the need for comprehensive data capture with obligations to protect user privacy and system security.

Data minimization principles guide the collection of only the information necessary for evaluation and improvement purposes. Teams should carefully consider what data is truly necessary and implement collection strategies that capture essential information while minimizing privacy risks.

Anonymization and pseudonymization techniques enable comprehensive analysis while protecting individual user privacy. These techniques must be carefully implemented to ensure that they provide genuine privacy protection while preserving the analytical value of the data.

Access controls and audit trails ensure that instrumentation data is only accessed by authorized personnel for legitimate purposes. These controls should include both technical measures such as encryption and access logging, and procedural measures such as data handling policies and training.

Data retention policies balance the value of historical data for trend analysis with the risks and costs of long-term data storage. These policies should consider regulatory requirements, business needs, and privacy obligations while providing clear guidelines for data lifecycle management.

## Implementation Strategies

### Layered Instrumentation Architecture

Effective AI instrumentation requires a layered architecture that captures information at different levels of the system stack. This layered approach ensures comprehensive coverage while enabling efficient processing and analysis of different types of data.

Application layer instrumentation captures high-level information about user interactions, business logic execution, and feature usage. This layer provides the business context necessary for understanding the impact and effectiveness of AI system behavior.

AI model layer instrumentation captures information specific to AI processing, including input preprocessing, model inference, output postprocessing, and confidence scores. This layer provides the technical detail necessary for understanding and debugging AI system behavior.

Infrastructure layer instrumentation captures system performance metrics, resource utilization, and operational health indicators. This layer provides the foundation for understanding how technical factors influence AI system behavior and user experience.

Integration layer instrumentation captures information about interactions between the AI system and other system components, including databases, APIs, and external services. This layer is crucial for understanding how AI systems behave within broader application ecosystems.

### Sampling and Aggregation Strategies

The volume of data generated by comprehensive AI instrumentation can be overwhelming, requiring careful sampling and aggregation strategies that preserve essential information while keeping data volumes manageable.

Intelligent sampling techniques ensure that instrumentation captures representative data without overwhelming storage and processing systems. These techniques might include stratified sampling to ensure coverage across different user populations, temporal sampling to capture behavior patterns over time, and event-driven sampling to ensure that important but rare events are captured.

Real-time aggregation reduces data volumes while preserving essential statistical information. This might include computing summary statistics, maintaining histograms of key metrics, and tracking trends over time. Real-time aggregation enables responsive monitoring while reducing storage requirements.

Hierarchical storage strategies balance the need for detailed data with storage cost considerations. Recent data might be stored in full detail for immediate analysis, while older data is aggregated or sampled to reduce storage costs while preserving long-term trend information.

### Automated Analysis and Alerting

Instrumentation data is only valuable if it leads to actionable insights and timely responses to issues. Automated analysis and alerting systems transform raw instrumentation data into actionable intelligence that enables teams to maintain and improve AI system performance.

Anomaly detection systems identify unusual patterns in AI system behavior that might indicate problems or opportunities for improvement. These systems must be carefully tuned to detect genuine issues while avoiding false positives that could lead to alert fatigue.

Trend analysis identifies gradual changes in system behavior that might not trigger immediate alerts but could indicate emerging issues or improvement opportunities. This analysis is particularly important for AI systems, where performance can degrade gradually due to data drift or changing user behavior.

Automated reporting provides regular summaries of system performance and behavior patterns, enabling teams to track progress toward objectives and identify areas requiring attention. These reports should be tailored to different audiences, providing technical details for engineering teams and business summaries for stakeholders.

## Practical Implementation Examples

### Basic Logging Framework

```python
import logging
import json
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class AIInteractionLog:
    """Structured log entry for AI interactions"""
    timestamp: float
    session_id: str
    user_id: Optional[str]
    input_text: str
    output_text: str
    model_version: str
    processing_time_ms: float
    confidence_score: Optional[float]
    context: Dict[str, Any]
    user_feedback: Optional[str] = None
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

class AIInstrumentationLogger:
    """Centralized logging for AI system interactions"""
    
    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger("ai_instrumentation")
        self.logger.setLevel(getattr(logging, log_level))
        
        # Configure structured logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_interaction(self, interaction: AIInteractionLog):
        """Log a complete AI interaction"""
        self.logger.info(f"AI_INTERACTION: {interaction.to_json()}")
    
    def log_error(self, error_type: str, error_message: str, context: Dict[str, Any]):
        """Log AI system errors with context"""
        error_log = {
            "timestamp": time.time(),
            "error_type": error_type,
            "error_message": error_message,
            "context": context
        }
        self.logger.error(f"AI_ERROR: {json.dumps(error_log)}")
    
    def log_performance_metric(self, metric_name: str, value: float, context: Dict[str, Any]):
        """Log performance metrics"""
        metric_log = {
            "timestamp": time.time(),
            "metric_name": metric_name,
            "value": value,
            "context": context
        }
        self.logger.info(f"AI_METRIC: {json.dumps(metric_log)}")

# Usage example
logger = AIInstrumentationLogger()

# Log an AI interaction
interaction = AIInteractionLog(
    timestamp=time.time(),
    session_id="session_123",
    user_id="user_456",
    input_text="What is the weather like today?",
    output_text="I don't have access to current weather data. Please check a weather service.",
    model_version="gpt-4-turbo",
    processing_time_ms=245.7,
    confidence_score=0.95,
    context={"location": "San Francisco", "time_of_day": "morning"}
)

logger.log_interaction(interaction)
```

### Quality Monitoring System

```python
import asyncio
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

class QualityDimension(Enum):
    HELPFULNESS = "helpfulness"
    ACCURACY = "accuracy"
    SAFETY = "safety"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"

@dataclass
class QualityAssessment:
    dimension: QualityDimension
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    explanation: str
    assessor_type: str  # "automated" or "human"

class QualityMonitor:
    """Real-time quality monitoring for AI outputs"""
    
    def __init__(self):
        self.automated_assessors: Dict[QualityDimension, Callable] = {}
        self.quality_thresholds: Dict[QualityDimension, float] = {
            QualityDimension.SAFETY: 0.95,
            QualityDimension.COHERENCE: 0.7,
            QualityDimension.RELEVANCE: 0.6
        }
        self.alert_callbacks: List[Callable] = []
    
    def register_assessor(self, dimension: QualityDimension, assessor: Callable):
        """Register an automated quality assessor"""
        self.automated_assessors[dimension] = assessor
    
    def register_alert_callback(self, callback: Callable):
        """Register callback for quality alerts"""
        self.alert_callbacks.append(callback)
    
    async def assess_output(self, input_text: str, output_text: str, context: Dict[str, Any]) -> List[QualityAssessment]:
        """Assess output quality across multiple dimensions"""
        assessments = []
        
        # Run automated assessments
        for dimension, assessor in self.automated_assessors.items():
            try:
                score, explanation = await assessor(input_text, output_text, context)
                assessment = QualityAssessment(
                    dimension=dimension,
                    score=score,
                    confidence=0.8,  # Default confidence for automated assessors
                    explanation=explanation,
                    assessor_type="automated"
                )
                assessments.append(assessment)
                
                # Check for quality threshold violations
                if score < self.quality_thresholds.get(dimension, 0.0):
                    await self._trigger_quality_alert(dimension, score, explanation, context)
                    
            except Exception as e:
                # Log assessment errors but don't fail the entire process
                print(f"Error in {dimension.value} assessment: {e}")
        
        return assessments
    
    async def _trigger_quality_alert(self, dimension: QualityDimension, score: float, explanation: str, context: Dict[str, Any]):
        """Trigger alerts for quality threshold violations"""
        alert_data = {
            "dimension": dimension.value,
            "score": score,
            "threshold": self.quality_thresholds[dimension],
            "explanation": explanation,
            "context": context,
            "timestamp": time.time()
        }
        
        for callback in self.alert_callbacks:
            try:
                await callback(alert_data)
            except Exception as e:
                print(f"Error in alert callback: {e}")

# Example automated assessors
async def safety_assessor(input_text: str, output_text: str, context: Dict[str, Any]) -> tuple[float, str]:
    """Simple safety assessment based on keyword detection"""
    unsafe_keywords = ["violence", "harm", "illegal", "dangerous"]
    
    output_lower = output_text.lower()
    unsafe_count = sum(1 for keyword in unsafe_keywords if keyword in output_lower)
    
    if unsafe_count == 0:
        return 1.0, "No unsafe content detected"
    else:
        score = max(0.0, 1.0 - (unsafe_count * 0.3))
        return score, f"Detected {unsafe_count} potential safety issues"

async def coherence_assessor(input_text: str, output_text: str, context: Dict[str, Any]) -> tuple[float, str]:
    """Simple coherence assessment based on length and structure"""
    if len(output_text.strip()) == 0:
        return 0.0, "Empty output"
    
    # Simple heuristics for coherence
    sentences = output_text.split('.')
    if len(sentences) < 1:
        return 0.3, "No clear sentence structure"
    
    avg_sentence_length = sum(len(s.strip()) for s in sentences) / len(sentences)
    if avg_sentence_length < 10:
        return 0.5, "Very short sentences may indicate poor coherence"
    elif avg_sentence_length > 200:
        return 0.6, "Very long sentences may indicate poor coherence"
    else:
        return 0.8, "Reasonable sentence structure detected"

# Usage example
monitor = QualityMonitor()
monitor.register_assessor(QualityDimension.SAFETY, safety_assessor)
monitor.register_assessor(QualityDimension.COHERENCE, coherence_assessor)

async def quality_alert_handler(alert_data: Dict[str, Any]):
    """Handle quality alerts"""
    print(f"QUALITY ALERT: {alert_data['dimension']} score {alert_data['score']} below threshold {alert_data['threshold']}")
    # In practice, this might send notifications, log to monitoring systems, etc.

monitor.register_alert_callback(quality_alert_handler)
```

### User Feedback Integration

```python
from typing import Optional, List
from datetime import datetime, timedelta
import sqlite3
import json

class UserFeedbackCollector:
    """Collect and analyze user feedback on AI outputs"""
    
    def __init__(self, db_path: str = "feedback.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize feedback database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                interaction_id TEXT NOT NULL,
                user_id TEXT,
                feedback_type TEXT NOT NULL,
                rating INTEGER,
                comment TEXT,
                timestamp REAL NOT NULL,
                context TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def record_feedback(self, session_id: str, interaction_id: str, feedback_type: str, 
                       rating: Optional[int] = None, comment: Optional[str] = None,
                       user_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        """Record user feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO feedback (session_id, interaction_id, user_id, feedback_type, 
                                rating, comment, timestamp, context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, interaction_id, user_id, feedback_type,
            rating, comment, time.time(), json.dumps(context or {})
        ))
        
        conn.commit()
        conn.close()
    
    def get_feedback_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get summary of recent feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = time.time() - (days * 24 * 3600)
        
        # Get rating distribution
        cursor.execute("""
            SELECT rating, COUNT(*) 
            FROM feedback 
            WHERE timestamp > ? AND rating IS NOT NULL
            GROUP BY rating
            ORDER BY rating
        """, (cutoff_time,))
        
        rating_distribution = dict(cursor.fetchall())
        
        # Get feedback type distribution
        cursor.execute("""
            SELECT feedback_type, COUNT(*) 
            FROM feedback 
            WHERE timestamp > ?
            GROUP BY feedback_type
        """, (cutoff_time,))
        
        feedback_types = dict(cursor.fetchall())
        
        # Calculate average rating
        cursor.execute("""
            SELECT AVG(rating) 
            FROM feedback 
            WHERE timestamp > ? AND rating IS NOT NULL
        """, (cutoff_time,))
        
        avg_rating = cursor.fetchone()[0] or 0.0
        
        conn.close()
        
        return {
            "period_days": days,
            "rating_distribution": rating_distribution,
            "feedback_types": feedback_types,
            "average_rating": round(avg_rating, 2),
            "total_feedback_count": sum(feedback_types.values())
        }
    
    def get_negative_feedback(self, threshold: int = 2, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent negative feedback for analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT session_id, interaction_id, rating, comment, timestamp, context
            FROM feedback 
            WHERE rating <= ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (threshold, limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "session_id": row[0],
                "interaction_id": row[1],
                "rating": row[2],
                "comment": row[3],
                "timestamp": row[4],
                "context": json.loads(row[5]) if row[5] else {}
            })
        
        conn.close()
        return results

# Usage example
feedback_collector = UserFeedbackCollector()

# Record positive feedback
feedback_collector.record_feedback(
    session_id="session_123",
    interaction_id="interaction_456",
    feedback_type="thumbs_up",
    rating=5,
    comment="Very helpful response!",
    context={"feature": "customer_service", "topic": "billing"}
)

# Record negative feedback
feedback_collector.record_feedback(
    session_id="session_124",
    interaction_id="interaction_457",
    feedback_type="thumbs_down",
    rating=1,
    comment="The response was not relevant to my question",
    context={"feature": "customer_service", "topic": "technical_support"}
)

# Get feedback summary
summary = feedback_collector.get_feedback_summary(days=7)
print(f"Average rating: {summary['average_rating']}")
print(f"Total feedback: {summary['total_feedback_count']}")
```

## Advanced Instrumentation Techniques

### Distributed Tracing for AI Systems

AI systems often involve complex pipelines with multiple processing stages, making it crucial to understand the flow of requests and identify bottlenecks or failure points. Distributed tracing provides visibility into these complex interactions.

```python
import opentelemetry
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import time
from typing import Dict, Any

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

class AISystemTracer:
    """Distributed tracing for AI system components"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.tracer = trace.get_tracer(service_name)
    
    def trace_ai_request(self, operation_name: str):
        """Decorator for tracing AI operations"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(operation_name) as span:
                    # Add common attributes
                    span.set_attribute("service.name", self.service_name)
                    span.set_attribute("operation.name", operation_name)
                    
                    try:
                        start_time = time.time()
                        result = func(*args, **kwargs)
                        
                        # Add success attributes
                        span.set_attribute("operation.success", True)
                        span.set_attribute("operation.duration_ms", 
                                         (time.time() - start_time) * 1000)
                        
                        return result
                    except Exception as e:
                        # Add error attributes
                        span.set_attribute("operation.success", False)
                        span.set_attribute("error.type", type(e).__name__)
                        span.set_attribute("error.message", str(e))
                        raise
            return wrapper
        return decorator
    
    def add_ai_context(self, span, input_text: str, output_text: str, 
                      model_info: Dict[str, Any]):
        """Add AI-specific context to spans"""
        span.set_attribute("ai.input_length", len(input_text))
        span.set_attribute("ai.output_length", len(output_text))
        span.set_attribute("ai.model_name", model_info.get("name", "unknown"))
        span.set_attribute("ai.model_version", model_info.get("version", "unknown"))

# Usage example
ai_tracer = AISystemTracer("ai_service")

@ai_tracer.trace_ai_request("text_generation")
def generate_text(prompt: str, model_config: Dict[str, Any]) -> str:
    """Example text generation function with tracing"""
    current_span = trace.get_current_span()
    
    # Simulate text generation
    time.sleep(0.1)  # Simulate processing time
    output = f"Generated response to: {prompt}"
    
    # Add AI-specific context
    ai_tracer.add_ai_context(
        current_span, 
        prompt, 
        output,
        {"name": "gpt-4", "version": "turbo"}
    )
    
    return output
```

### Performance Profiling

Understanding the performance characteristics of AI systems requires specialized profiling that goes beyond traditional CPU and memory metrics.

```python
import psutil
import time
from typing import Dict, Any, List
from dataclasses import dataclass
import threading
from contextlib import contextmanager

@dataclass
class PerformanceMetrics:
    """Performance metrics for AI operations"""
    operation_name: str
    start_time: float
    end_time: float
    cpu_percent: float
    memory_mb: float
    gpu_memory_mb: float
    input_tokens: int
    output_tokens: int
    
    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000
    
    @property
    def tokens_per_second(self) -> float:
        if self.duration_ms == 0:
            return 0
        return (self.input_tokens + self.output_tokens) / (self.duration_ms / 1000)

class PerformanceProfiler:
    """Profile performance of AI operations"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self._monitoring = False
        self._monitor_thread = None
    
    @contextmanager
    def profile_operation(self, operation_name: str, input_tokens: int = 0, output_tokens: int = 0):
        """Context manager for profiling AI operations"""
        start_time = time.time()
        start_cpu = psutil.cpu_percent()
        start_memory = psutil.virtual_memory().used / 1024 / 1024
        
        # Start monitoring thread for continuous metrics
        self._start_monitoring()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_cpu = psutil.cpu_percent()
            end_memory = psutil.virtual_memory().used / 1024 / 1024
            
            self._stop_monitoring()
            
            # Record metrics
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                cpu_percent=(start_cpu + end_cpu) / 2,
                memory_mb=end_memory - start_memory,
                gpu_memory_mb=self._get_gpu_memory(),
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            
            self.metrics.append(metrics)
    
    def _start_monitoring(self):
        """Start continuous monitoring thread"""
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_resources)
        self._monitor_thread.start()
    
    def _stop_monitoring(self):
        """Stop continuous monitoring thread"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
    
    def _monitor_resources(self):
        """Monitor system resources continuously"""
        while self._monitoring:
            # Log current resource usage
            cpu_percent = psutil.cpu_percent()
            memory_mb = psutil.virtual_memory().used / 1024 / 1024
            
            # In practice, you might send this to a monitoring system
            time.sleep(0.1)  # Monitor every 100ms
    
    def _get_gpu_memory(self) -> float:
        """Get GPU memory usage (placeholder implementation)"""
        # In practice, this would use nvidia-ml-py or similar
        return 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics"""
        if not self.metrics:
            return {}
        
        total_operations = len(self.metrics)
        avg_duration = sum(m.duration_ms for m in self.metrics) / total_operations
        avg_tokens_per_sec = sum(m.tokens_per_second for m in self.metrics) / total_operations
        
        return {
            "total_operations": total_operations,
            "average_duration_ms": round(avg_duration, 2),
            "average_tokens_per_second": round(avg_tokens_per_sec, 2),
            "operations_by_type": self._group_by_operation()
        }
    
    def _group_by_operation(self) -> Dict[str, Dict[str, float]]:
        """Group metrics by operation type"""
        grouped = {}
        for metric in self.metrics:
            if metric.operation_name not in grouped:
                grouped[metric.operation_name] = []
            grouped[metric.operation_name].append(metric)
        
        summary = {}
        for op_name, op_metrics in grouped.items():
            summary[op_name] = {
                "count": len(op_metrics),
                "avg_duration_ms": sum(m.duration_ms for m in op_metrics) / len(op_metrics),
                "avg_tokens_per_sec": sum(m.tokens_per_second for m in op_metrics) / len(op_metrics)
            }
        
        return summary

# Usage example
profiler = PerformanceProfiler()

with profiler.profile_operation("text_generation", input_tokens=50, output_tokens=100):
    # Simulate AI operation
    time.sleep(0.2)
    result = "Generated text response"

print(profiler.get_performance_summary())
```

## Conclusion: Building Robust Observability

Effective instrumentation is the foundation that enables all other evaluation activities. Without comprehensive data capture and analysis capabilities, teams cannot understand their AI systems' behavior, identify improvement opportunities, or respond effectively to issues.

The key insight is that AI instrumentation must go beyond traditional software monitoring to capture the rich contextual information necessary for understanding AI system behavior. This includes not just technical metrics but also user feedback, quality assessments, and business context.

Building robust instrumentation requires upfront investment in infrastructure and processes, but this investment pays dividends throughout the system lifecycle. Teams with comprehensive instrumentation can iterate faster, debug more effectively, and scale more confidently than those operating with limited observability.

As AI systems become more complex and critical to business operations, the ability to observe and understand their behavior becomes increasingly important. The teams that invest in robust instrumentation early will be best positioned to build reliable, trustworthy AI systems that can adapt and improve over time.

---

**Next**: [Error Analysis Introduction →](05-error-analysis-introduction.md)

## References

[1] "Observability for Machine Learning" - Google Cloud AI, 2024. https://cloud.google.com/ai/ml-observability

[2] "Monitoring AI Systems in Production" - Microsoft Azure AI, 2024. https://docs.microsoft.com/en-us/azure/machine-learning/how-to-monitor-models

[3] "Building Instrumentation for AI Applications" - AWS AI Services, 2024. https://aws.amazon.com/ai/ml-monitoring

[4] "AI System Telemetry Best Practices" - Meta AI Research, 2024. https://ai.meta.com/research/publications/ai-telemetry

[5] "Distributed Tracing for AI Pipelines" - OpenTelemetry Community, 2024. https://opentelemetry.io/docs/instrumentation/ai-systems

