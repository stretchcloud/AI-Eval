# Exercise 3: RAG System with Tool Integration

## Objective
Build and evaluate a comprehensive RAG (Retrieval-Augmented Generation) system that integrates multiple tools for enhanced information retrieval, processing, and generation capabilities.

## Duration
5-6 hours

## Skills Developed
- RAG system architecture design and implementation
- Tool integration and orchestration in RAG workflows
- Multi-modal retrieval and generation evaluation
- Performance optimization for RAG systems with tools
- Quality assessment for tool-augmented generation

## Prerequisites
- Understanding of RAG system evaluation from Section 1
- Knowledge of tool calling evaluation from Section 6
- Python programming experience with vector databases and LLMs
- Familiarity with information retrieval and generation metrics

## Learning Outcomes
Upon completing this exercise, you will be able to:
- Design and implement sophisticated RAG systems with tool integration
- Build comprehensive evaluation frameworks for tool-augmented RAG
- Create performance optimization strategies for complex RAG workflows
- Develop quality assessment protocols for multi-tool generation systems
- Integrate RAG systems with external tools and APIs

## Exercise Overview

In this exercise, you will build a sophisticated RAG system for a legal research platform that integrates multiple tools including document analysis, case law search, regulatory compliance checking, and citation validation. Your system must handle complex legal queries, retrieve relevant information from multiple sources, use specialized tools for analysis, and generate comprehensive legal research reports.

## Part 1: RAG System Architecture with Tool Integration (90 minutes)

### 1.1 Understanding the Legal Research RAG Architecture

First, let's examine the legal research RAG system's architecture:

```python
import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
import uuid
import numpy as np
from collections import defaultdict, deque
import hashlib

class ToolType(Enum):
    DOCUMENT_ANALYZER = "document_analyzer"
    CASE_LAW_SEARCH = "case_law_search"
    REGULATORY_CHECKER = "regulatory_checker"
    CITATION_VALIDATOR = "citation_validator"
    LEGAL_CALCULATOR = "legal_calculator"
    PRECEDENT_ANALYZER = "precedent_analyzer"
    STATUTE_INTERPRETER = "statute_interpreter"

class RetrievalType(Enum):
    SEMANTIC_SEARCH = "semantic_search"
    KEYWORD_SEARCH = "keyword_search"
    HYBRID_SEARCH = "hybrid_search"
    CITATION_SEARCH = "citation_search"
    TEMPORAL_SEARCH = "temporal_search"

class GenerationType(Enum):
    LEGAL_BRIEF = "legal_brief"
    CASE_ANALYSIS = "case_analysis"
    REGULATORY_SUMMARY = "regulatory_summary"
    PRECEDENT_REPORT = "precedent_report"
    COMPLIANCE_ASSESSMENT = "compliance_assessment"

@dataclass
class ToolCall:
    """Individual tool call within the RAG workflow."""
    tool_id: str
    tool_type: ToolType
    input_parameters: Dict[str, Any]
    output_result: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    success: bool = False
    error_message: Optional[str] = None
    confidence_score: Optional[float] = None

@dataclass
class RetrievalResult:
    """Result from document retrieval operations."""
    document_id: str
    content: str
    relevance_score: float
    source_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieval_method: RetrievalType = RetrievalType.SEMANTIC_SEARCH
    tool_enhanced: bool = False
    tool_annotations: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class RAGExecution:
    """Complete RAG execution with tool integration."""
    execution_id: str
    query: str
    query_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    retrieval_results: List[RetrievalResult] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    generated_response: Optional[str] = None
    generation_type: Optional[GenerationType] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    quality_scores: Dict[str, float] = field(default_factory=dict)

class LegalResearchTool:
    """Base class for legal research tools."""
    
    def __init__(self, tool_id: str, tool_type: ToolType):
        self.tool_id = tool_id
        self.tool_type = tool_type
        self.performance_history = []
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        raise NotImplementedError("Subclasses must implement execute method")
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate input parameters for the tool."""
        raise NotImplementedError("Subclasses must implement validate_parameters method")

class DocumentAnalyzerTool(LegalResearchTool):
    """Tool for analyzing legal documents and extracting key information."""
    
    def __init__(self):
        super().__init__("doc_analyzer_v1", ToolType.DOCUMENT_ANALYZER)
        self.analysis_capabilities = [
            "contract_analysis", "case_law_analysis", "statute_analysis",
            "regulation_analysis", "brief_analysis", "opinion_analysis"
        ]
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for document analysis."""
        required_params = ["document_content", "analysis_type"]
        return all(param in parameters for param in required_params)
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute document analysis."""
        
        if not self.validate_parameters(parameters):
            raise ValueError("Invalid parameters for document analysis")
        
        document_content = parameters["document_content"]
        analysis_type = parameters["analysis_type"]
        
        # Simulate document analysis processing
        await asyncio.sleep(0.5)  # Simulate processing time
        
        # Mock analysis results
        analysis_result = {
            "document_type": analysis_type,
            "key_entities": [
                {"entity": "Contract Party A", "type": "organization", "confidence": 0.95},
                {"entity": "Effective Date", "type": "date", "confidence": 0.92},
                {"entity": "Termination Clause", "type": "legal_concept", "confidence": 0.88}
            ],
            "legal_concepts": [
                {"concept": "Force Majeure", "relevance": 0.85, "section": "Section 12"},
                {"concept": "Indemnification", "relevance": 0.78, "section": "Section 15"},
                {"concept": "Governing Law", "relevance": 0.92, "section": "Section 18"}
            ],
            "risk_factors": [
                {"risk": "Ambiguous termination conditions", "severity": "medium", "confidence": 0.82},
                {"risk": "Broad indemnification scope", "severity": "high", "confidence": 0.89}
            ],
            "compliance_issues": [
                {"issue": "Missing data protection clause", "severity": "medium", "regulation": "GDPR"}
            ],
            "summary": f"Analysis of {analysis_type} document reveals key legal concepts and potential risk factors.",
            "confidence_score": 0.87
        }
        
        return analysis_result

class CaseLawSearchTool(LegalResearchTool):
    """Tool for searching and analyzing case law precedents."""
    
    def __init__(self):
        super().__init__("case_search_v1", ToolType.CASE_LAW_SEARCH)
        self.jurisdictions = ["federal", "state", "international"]
        self.case_databases = ["westlaw", "lexis", "google_scholar", "courtlistener"]
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for case law search."""
        required_params = ["search_query", "jurisdiction"]
        return all(param in parameters for param in required_params)
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute case law search."""
        
        if not self.validate_parameters(parameters):
            raise ValueError("Invalid parameters for case law search")
        
        search_query = parameters["search_query"]
        jurisdiction = parameters["jurisdiction"]
        max_results = parameters.get("max_results", 10)
        
        # Simulate case law search processing
        await asyncio.sleep(1.2)  # Simulate search time
        
        # Mock search results
        search_results = {
            "query": search_query,
            "jurisdiction": jurisdiction,
            "total_results": 156,
            "cases": [
                {
                    "case_name": "Smith v. Johnson Corp",
                    "citation": "123 F.3d 456 (9th Cir. 2020)",
                    "court": "9th Circuit Court of Appeals",
                    "date": "2020-03-15",
                    "relevance_score": 0.94,
                    "key_holdings": [
                        "Contract interpretation requires consideration of industry standards",
                        "Ambiguous terms construed against drafter"
                    ],
                    "precedential_value": "binding",
                    "summary": "Court held that ambiguous contract terms must be interpreted in light of industry standards."
                },
                {
                    "case_name": "Tech Innovations LLC v. DataCorp",
                    "citation": "789 F.Supp.2d 123 (N.D. Cal. 2019)",
                    "court": "Northern District of California",
                    "date": "2019-11-22",
                    "relevance_score": 0.87,
                    "key_holdings": [
                        "Software licensing agreements subject to UCC Article 2",
                        "Implied warranty of merchantability applies to software"
                    ],
                    "precedential_value": "persuasive",
                    "summary": "District court applied UCC provisions to software licensing dispute."
                }
            ],
            "search_metadata": {
                "search_time": 1.2,
                "databases_searched": ["westlaw", "lexis"],
                "filters_applied": ["jurisdiction", "date_range"],
                "confidence_score": 0.91
            }
        }
        
        return search_results

class RegulatoryCheckerTool(LegalResearchTool):
    """Tool for checking regulatory compliance and requirements."""
    
    def __init__(self):
        super().__init__("reg_checker_v1", ToolType.REGULATORY_CHECKER)
        self.regulatory_domains = [
            "securities", "healthcare", "financial_services", "data_protection",
            "environmental", "employment", "antitrust", "international_trade"
        ]
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for regulatory checking."""
        required_params = ["business_activity", "jurisdiction", "regulatory_domain"]
        return all(param in parameters for param in required_params)
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute regulatory compliance check."""
        
        if not self.validate_parameters(parameters):
            raise ValueError("Invalid parameters for regulatory checking")
        
        business_activity = parameters["business_activity"]
        jurisdiction = parameters["jurisdiction"]
        regulatory_domain = parameters["regulatory_domain"]
        
        # Simulate regulatory analysis processing
        await asyncio.sleep(0.8)  # Simulate processing time
        
        # Mock regulatory analysis results
        compliance_result = {
            "business_activity": business_activity,
            "jurisdiction": jurisdiction,
            "regulatory_domain": regulatory_domain,
            "applicable_regulations": [
                {
                    "regulation": "Securities Act of 1933",
                    "section": "Section 5",
                    "applicability": "high",
                    "compliance_status": "requires_review",
                    "requirements": [
                        "Registration of securities offerings",
                        "Disclosure of material information",
                        "Anti-fraud provisions compliance"
                    ]
                },
                {
                    "regulation": "Investment Company Act of 1940",
                    "section": "Section 3(a)(1)",
                    "applicability": "medium",
                    "compliance_status": "compliant",
                    "requirements": [
                        "Investment company registration exemption",
                        "Beneficial ownership limitations"
                    ]
                }
            ],
            "compliance_gaps": [
                {
                    "gap": "Missing accredited investor verification procedures",
                    "severity": "high",
                    "regulation": "Regulation D",
                    "remediation": "Implement investor verification process"
                }
            ],
            "recommendations": [
                "Consult with securities attorney for offering structure",
                "Implement comprehensive compliance monitoring system",
                "Regular compliance audits and updates"
            ],
            "confidence_score": 0.89
        }
        
        return compliance_result

class LegalRAGSystem:
    """
    Comprehensive RAG system with integrated legal research tools.
    """
    
    def __init__(self):
        self.tools = self._initialize_tools()
        self.vector_store = self._initialize_vector_store()
        self.retrieval_strategies = self._initialize_retrieval_strategies()
        self.generation_templates = self._initialize_generation_templates()
        self.execution_history = []
    
    def _initialize_tools(self) -> Dict[str, LegalResearchTool]:
        """Initialize available legal research tools."""
        
        return {
            "document_analyzer": DocumentAnalyzerTool(),
            "case_law_search": CaseLawSearchTool(),
            "regulatory_checker": RegulatoryCheckerTool()
        }
    
    def _initialize_vector_store(self) -> Dict[str, Any]:
        """Initialize mock vector store for legal documents."""
        
        # Mock legal document database
        return {
            "documents": [
                {
                    "id": "doc_001",
                    "content": "Software licensing agreement between TechCorp and ClientCo...",
                    "type": "contract",
                    "jurisdiction": "california",
                    "date": "2023-01-15",
                    "embedding": np.random.rand(768).tolist()  # Mock embedding
                },
                {
                    "id": "doc_002",
                    "content": "Securities offering memorandum for private placement...",
                    "type": "securities_document",
                    "jurisdiction": "federal",
                    "date": "2023-03-22",
                    "embedding": np.random.rand(768).tolist()  # Mock embedding
                },
                {
                    "id": "doc_003",
                    "content": "Employment agreement with non-compete provisions...",
                    "type": "employment_contract",
                    "jurisdiction": "new_york",
                    "date": "2023-02-10",
                    "embedding": np.random.rand(768).tolist()  # Mock embedding
                }
            ],
            "index_metadata": {
                "total_documents": 3,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "embedding_model": "legal-bert-base"
            }
        }
    
    def _initialize_retrieval_strategies(self) -> Dict[str, Any]:
        """Initialize retrieval strategies for different query types."""
        
        return {
            "contract_analysis": {
                "primary_retrieval": RetrievalType.SEMANTIC_SEARCH,
                "secondary_retrieval": RetrievalType.KEYWORD_SEARCH,
                "tools_required": ["document_analyzer"],
                "max_documents": 5
            },
            "case_law_research": {
                "primary_retrieval": RetrievalType.CITATION_SEARCH,
                "secondary_retrieval": RetrievalType.SEMANTIC_SEARCH,
                "tools_required": ["case_law_search"],
                "max_documents": 10
            },
            "compliance_check": {
                "primary_retrieval": RetrievalType.KEYWORD_SEARCH,
                "secondary_retrieval": RetrievalType.SEMANTIC_SEARCH,
                "tools_required": ["regulatory_checker"],
                "max_documents": 8
            },
            "general_legal_research": {
                "primary_retrieval": RetrievalType.HYBRID_SEARCH,
                "secondary_retrieval": RetrievalType.SEMANTIC_SEARCH,
                "tools_required": ["document_analyzer", "case_law_search"],
                "max_documents": 12
            }
        }
    
    def _initialize_generation_templates(self) -> Dict[str, str]:
        """Initialize generation templates for different output types."""
        
        return {
            "legal_brief": """
            Based on the retrieved documents and tool analysis, here is a comprehensive legal brief:
            
            ## Executive Summary
            {executive_summary}
            
            ## Legal Analysis
            {legal_analysis}
            
            ## Relevant Precedents
            {precedents}
            
            ## Regulatory Considerations
            {regulatory_analysis}
            
            ## Recommendations
            {recommendations}
            
            ## Supporting Documentation
            {supporting_docs}
            """,
            "case_analysis": """
            ## Case Analysis Report
            
            ### Case Overview
            {case_overview}
            
            ### Key Legal Issues
            {legal_issues}
            
            ### Applicable Law
            {applicable_law}
            
            ### Analysis and Reasoning
            {analysis}
            
            ### Conclusion
            {conclusion}
            """,
            "compliance_assessment": """
            ## Regulatory Compliance Assessment
            
            ### Business Activity Overview
            {business_overview}
            
            ### Applicable Regulations
            {regulations}
            
            ### Compliance Status
            {compliance_status}
            
            ### Risk Assessment
            {risk_assessment}
            
            ### Remediation Plan
            {remediation_plan}
            """
        }
    
    async def execute_rag_query(self, 
                               query: str,
                               query_type: str,
                               generation_type: GenerationType,
                               config: Optional[Dict[str, Any]] = None) -> RAGExecution:
        """
        Execute a complete RAG query with tool integration.
        """
        
        execution = RAGExecution(
            execution_id=str(uuid.uuid4()),
            query=query,
            query_type=query_type,
            start_time=datetime.now(timezone.utc),
            generation_type=generation_type
        )
        
        try:
            # Step 1: Retrieve relevant documents
            retrieval_results = await self._retrieve_documents(query, query_type, config)
            execution.retrieval_results = retrieval_results
            
            # Step 2: Execute relevant tools
            tool_calls = await self._execute_tools(query, query_type, retrieval_results, config)
            execution.tool_calls = tool_calls
            
            # Step 3: Generate response
            generated_response = await self._generate_response(
                query, retrieval_results, tool_calls, generation_type, config
            )
            execution.generated_response = generated_response
            
            # Step 4: Calculate performance metrics
            execution.end_time = datetime.now(timezone.utc)
            execution.performance_metrics = self._calculate_performance_metrics(execution)
            
            # Step 5: Assess quality
            execution.quality_scores = await self._assess_quality(execution, config)
            
        except Exception as e:
            execution.end_time = datetime.now(timezone.utc)
            execution.performance_metrics = {"error": str(e)}
        
        # Store execution for analysis
        self.execution_history.append(execution)
        
        return execution
    
    async def _retrieve_documents(self, 
                                query: str,
                                query_type: str,
                                config: Optional[Dict[str, Any]]) -> List[RetrievalResult]:
        """
        Retrieve relevant documents using appropriate strategies.
        """
        
        strategy = self.retrieval_strategies.get(query_type, self.retrieval_strategies["general_legal_research"])
        max_docs = strategy["max_documents"]
        
        # Simulate document retrieval
        await asyncio.sleep(0.3)  # Simulate retrieval time
        
        # Mock retrieval results
        retrieval_results = []
        for i, doc in enumerate(self.vector_store["documents"][:max_docs]):
            result = RetrievalResult(
                document_id=doc["id"],
                content=doc["content"],
                relevance_score=0.9 - (i * 0.1),  # Mock decreasing relevance
                source_type=doc["type"],
                metadata={
                    "jurisdiction": doc["jurisdiction"],
                    "date": doc["date"],
                    "document_type": doc["type"]
                },
                retrieval_method=strategy["primary_retrieval"]
            )
            retrieval_results.append(result)
        
        return retrieval_results
    
    async def _execute_tools(self, 
                           query: str,
                           query_type: str,
                           retrieval_results: List[RetrievalResult],
                           config: Optional[Dict[str, Any]]) -> List[ToolCall]:
        """
        Execute relevant tools based on query type and retrieved documents.
        """
        
        strategy = self.retrieval_strategies.get(query_type, self.retrieval_strategies["general_legal_research"])
        required_tools = strategy["tools_required"]
        
        tool_calls = []
        
        for tool_name in required_tools:
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                
                # Prepare tool parameters based on query and retrieved documents
                parameters = self._prepare_tool_parameters(tool_name, query, retrieval_results)
                
                # Execute tool
                tool_call = ToolCall(
                    tool_id=tool.tool_id,
                    tool_type=tool.tool_type,
                    input_parameters=parameters
                )
                
                try:
                    start_time = time.time()
                    result = await tool.execute(parameters)
                    execution_time = time.time() - start_time
                    
                    tool_call.output_result = result
                    tool_call.execution_time = execution_time
                    tool_call.success = True
                    tool_call.confidence_score = result.get("confidence_score", 0.8)
                    
                except Exception as e:
                    tool_call.success = False
                    tool_call.error_message = str(e)
                
                tool_calls.append(tool_call)
        
        return tool_calls
    
    def _prepare_tool_parameters(self, 
                               tool_name: str,
                               query: str,
                               retrieval_results: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Prepare parameters for tool execution based on context.
        """
        
        if tool_name == "document_analyzer":
            # Use the most relevant document for analysis
            if retrieval_results:
                return {
                    "document_content": retrieval_results[0].content,
                    "analysis_type": retrieval_results[0].source_type
                }
            else:
                return {
                    "document_content": query,
                    "analysis_type": "general"
                }
        
        elif tool_name == "case_law_search":
            return {
                "search_query": query,
                "jurisdiction": "federal",
                "max_results": 5
            }
        
        elif tool_name == "regulatory_checker":
            return {
                "business_activity": query,
                "jurisdiction": "federal",
                "regulatory_domain": "securities"  # Default domain
            }
        
        else:
            return {"query": query}
    
    async def _generate_response(self, 
                               query: str,
                               retrieval_results: List[RetrievalResult],
                               tool_calls: List[ToolCall],
                               generation_type: GenerationType,
                               config: Optional[Dict[str, Any]]) -> str:
        """
        Generate response using retrieved documents and tool results.
        """
        
        # Simulate response generation
        await asyncio.sleep(1.0)  # Simulate generation time
        
        # Get appropriate template
        template_key = generation_type.value
        template = self.generation_templates.get(template_key, self.generation_templates["legal_brief"])
        
        # Extract information from retrieval results and tool calls
        context_info = self._extract_context_information(retrieval_results, tool_calls)
        
        # Generate response using template
        response = template.format(**context_info)
        
        return response
    
    def _extract_context_information(self, 
                                   retrieval_results: List[RetrievalResult],
                                   tool_calls: List[ToolCall]) -> Dict[str, str]:
        """
        Extract and organize information from retrieval results and tool calls.
        """
        
        context = {
            "executive_summary": "Based on the analysis of retrieved documents and tool outputs...",
            "legal_analysis": "The legal analysis reveals several key considerations...",
            "precedents": "Relevant case law precedents include...",
            "regulatory_analysis": "Regulatory compliance analysis indicates...",
            "recommendations": "Based on the comprehensive analysis, we recommend...",
            "supporting_docs": "Supporting documentation includes...",
            "case_overview": "The case presents the following key facts...",
            "legal_issues": "The primary legal issues identified are...",
            "applicable_law": "The applicable legal framework includes...",
            "analysis": "Our analysis of the legal issues reveals...",
            "conclusion": "Based on the analysis, we conclude...",
            "business_overview": "The business activity under review involves...",
            "regulations": "Applicable regulations include...",
            "compliance_status": "Current compliance status assessment...",
            "risk_assessment": "Risk assessment reveals the following concerns...",
            "remediation_plan": "Recommended remediation steps include..."
        }
        
        # Enhance context with actual tool results
        for tool_call in tool_calls:
            if tool_call.success and tool_call.output_result:
                if tool_call.tool_type == ToolType.CASE_LAW_SEARCH:
                    cases = tool_call.output_result.get("cases", [])
                    if cases:
                        precedents_text = "\n".join([
                            f"- {case['case_name']}: {case['summary']}"
                            for case in cases[:3]
                        ])
                        context["precedents"] = precedents_text
                
                elif tool_call.tool_type == ToolType.REGULATORY_CHECKER:
                    regulations = tool_call.output_result.get("applicable_regulations", [])
                    if regulations:
                        reg_text = "\n".join([
                            f"- {reg['regulation']}: {reg['compliance_status']}"
                            for reg in regulations
                        ])
                        context["regulations"] = reg_text
        
        return context
    
    def _calculate_performance_metrics(self, execution: RAGExecution) -> Dict[str, float]:
        """
        Calculate performance metrics for the RAG execution.
        """
        
        total_time = (execution.end_time - execution.start_time).total_seconds()
        
        # Calculate tool execution times
        tool_times = [tc.execution_time for tc in execution.tool_calls if tc.execution_time]
        total_tool_time = sum(tool_times) if tool_times else 0
        
        # Calculate retrieval metrics
        avg_relevance = np.mean([r.relevance_score for r in execution.retrieval_results]) if execution.retrieval_results else 0
        
        return {
            "total_execution_time": total_time,
            "tool_execution_time": total_tool_time,
            "retrieval_time": total_time - total_tool_time - 1.0,  # Subtract generation time
            "generation_time": 1.0,  # Mock generation time
            "documents_retrieved": len(execution.retrieval_results),
            "tools_executed": len(execution.tool_calls),
            "successful_tool_calls": len([tc for tc in execution.tool_calls if tc.success]),
            "average_relevance_score": avg_relevance,
            "response_length": len(execution.generated_response) if execution.generated_response else 0
        }
    
    async def _assess_quality(self, 
                            execution: RAGExecution,
                            config: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """
        Assess the quality of the RAG execution and generated response.
        """
        
        # Simulate quality assessment
        await asyncio.sleep(0.2)
        
        # Mock quality scores
        quality_scores = {
            "relevance": 0.85,
            "completeness": 0.78,
            "accuracy": 0.82,
            "coherence": 0.88,
            "legal_soundness": 0.79,
            "citation_quality": 0.83,
            "tool_integration": 0.86,
            "overall_quality": 0.83
        }
        
        # Adjust scores based on tool success
        successful_tools = len([tc for tc in execution.tool_calls if tc.success])
        total_tools = len(execution.tool_calls)
        
        if total_tools > 0:
            tool_success_rate = successful_tools / total_tools
            quality_scores["tool_integration"] = tool_success_rate * 0.9
            quality_scores["overall_quality"] = (
                quality_scores["overall_quality"] * 0.7 + 
                quality_scores["tool_integration"] * 0.3
            )
        
        return quality_scores
```

### 1.2 Implementation Task

Your task is to implement the RAG evaluation framework:

```python
class RAGToolEvaluationFramework:
    """
    Comprehensive evaluation framework for RAG systems with tool integration.
    """
    
    def __init__(self, rag_system: LegalRAGSystem):
        self.rag_system = rag_system
        self.evaluation_metrics = self._initialize_evaluation_metrics()
        self.benchmark_queries = self._initialize_benchmark_queries()
    
    def _initialize_evaluation_metrics(self) -> Dict[str, Any]:
        """Initialize evaluation metrics for RAG with tools."""
        
        return {
            "retrieval_metrics": {
                "precision_at_k": {"k_values": [1, 3, 5, 10], "weight": 0.3},
                "recall_at_k": {"k_values": [1, 3, 5, 10], "weight": 0.3},
                "mrr": {"weight": 0.2},  # Mean Reciprocal Rank
                "ndcg": {"weight": 0.2}  # Normalized Discounted Cumulative Gain
            },
            "tool_integration_metrics": {
                "tool_selection_accuracy": {"weight": 0.25},
                "tool_execution_success_rate": {"weight": 0.25},
                "tool_output_relevance": {"weight": 0.25},
                "tool_coordination_efficiency": {"weight": 0.25}
            },
            "generation_metrics": {
                "factual_accuracy": {"weight": 0.3},
                "legal_soundness": {"weight": 0.25},
                "completeness": {"weight": 0.2},
                "coherence": {"weight": 0.15},
                "citation_quality": {"weight": 0.1}
            },
            "performance_metrics": {
                "latency": {"weight": 0.4},
                "throughput": {"weight": 0.3},
                "resource_efficiency": {"weight": 0.3}
            }
        }
    
    async def evaluate_rag_execution(self, 
                                   execution: RAGExecution,
                                   ground_truth: Dict[str, Any],
                                   evaluation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a RAG execution.
        
        TODO: Implement this method to evaluate:
        1. Retrieval quality and relevance
        2. Tool integration effectiveness
        3. Generation quality and accuracy
        4. Overall system performance
        
        Return comprehensive evaluation results.
        """
        # Your implementation here
        pass
    
    async def evaluate_retrieval_quality(self, 
                                       execution: RAGExecution,
                                       ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the quality of document retrieval.
        
        TODO: Implement this method to evaluate:
        1. Precision and recall at different k values
        2. Mean reciprocal rank (MRR)
        3. Normalized discounted cumulative gain (NDCG)
        4. Diversity and coverage metrics
        
        Return retrieval quality metrics.
        """
        # Your implementation here
        pass
    
    async def evaluate_tool_integration(self, 
                                      execution: RAGExecution,
                                      ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the effectiveness of tool integration.
        
        TODO: Implement this method to evaluate:
        1. Tool selection accuracy and appropriateness
        2. Tool execution success rates and error handling
        3. Tool output relevance and quality
        4. Tool coordination and workflow efficiency
        
        Return tool integration metrics.
        """
        # Your implementation here
        pass
    
    async def evaluate_generation_quality(self, 
                                        execution: RAGExecution,
                                        ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the quality of generated responses.
        
        TODO: Implement this method to evaluate:
        1. Factual accuracy and legal soundness
        2. Completeness and comprehensiveness
        3. Coherence and readability
        4. Citation quality and attribution
        
        Return generation quality metrics.
        """
        # Your implementation here
        pass
```

### 1.3 Testing Your Implementation

Test your RAG evaluation framework with this sample scenario:

```python
async def test_rag_tool_evaluation():
    """Test the RAG tool evaluation framework."""
    
    # Initialize the system
    rag_system = LegalRAGSystem()
    evaluation_framework = RAGToolEvaluationFramework(rag_system)
    
    # Execute a sample query
    query = "Analyze the enforceability of non-compete clauses in software licensing agreements"
    execution = await rag_system.execute_rag_query(
        query=query,
        query_type="contract_analysis",
        generation_type=GenerationType.LEGAL_BRIEF
    )
    
    # Create ground truth for evaluation
    ground_truth = {
        "relevant_documents": ["doc_001", "doc_003"],
        "expected_tools": ["document_analyzer", "case_law_search"],
        "key_legal_concepts": ["non-compete", "enforceability", "software licensing"],
        "expected_citations": ["Smith v. Johnson Corp"],
        "quality_expectations": {
            "factual_accuracy": 0.9,
            "legal_soundness": 0.85,
            "completeness": 0.8
        }
    }
    
    # Evaluate the execution
    evaluation_result = await evaluation_framework.evaluate_rag_execution(
        execution, ground_truth
    )
    
    # Print results
    print("RAG Tool Evaluation Results:")
    print(json.dumps(evaluation_result, indent=2, default=str))

# Run the test
if __name__ == "__main__":
    asyncio.run(test_rag_tool_evaluation())
```

## Part 2: Performance Optimization and Caching (75 minutes)

### 2.1 Understanding RAG Performance Optimization

Build optimization strategies for RAG systems with tools:

```python
class RAGPerformanceOptimizer:
    """
    Performance optimization system for RAG with tool integration.
    """
    
    def __init__(self, rag_system: LegalRAGSystem):
        self.rag_system = rag_system
        self.cache_manager = CacheManager()
        self.optimization_strategies = self._initialize_optimization_strategies()
    
    async def optimize_rag_performance(self, 
                                     execution_history: List[RAGExecution],
                                     optimization_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize RAG system performance based on execution history.
        
        TODO: Implement this method to:
        1. Analyze performance bottlenecks
        2. Implement caching strategies
        3. Optimize tool execution order
        4. Improve retrieval efficiency
        
        Return optimization recommendations and implementations.
        """
        # Your implementation here
        pass

class CacheManager:
    """
    Intelligent caching system for RAG components.
    """
    
    def __init__(self):
        self.retrieval_cache = {}
        self.tool_cache = {}
        self.generation_cache = {}
        self.cache_stats = defaultdict(int)
    
    async def get_cached_retrieval(self, query_hash: str) -> Optional[List[RetrievalResult]]:
        """Get cached retrieval results."""
        # Your implementation here
        pass
    
    async def cache_retrieval_results(self, 
                                    query_hash: str,
                                    results: List[RetrievalResult]) -> None:
        """Cache retrieval results."""
        # Your implementation here
        pass
```

### 2.2 Implementation Task

Implement the performance optimization methods:

```python
# TODO: Implement performance optimization methods
async def _analyze_performance_bottlenecks(self, 
                                         execution_history: List[RAGExecution]) -> Dict[str, Any]:
    """Analyze performance bottlenecks in RAG executions."""
    # Your implementation here
    pass

async def _implement_caching_strategies(self, 
                                      optimization_config: Dict[str, Any]) -> Dict[str, Any]:
    """Implement intelligent caching strategies."""
    # Your implementation here
    pass

async def _optimize_tool_execution(self, 
                                 execution_history: List[RAGExecution]) -> Dict[str, Any]:
    """Optimize tool execution order and coordination."""
    # Your implementation here
    pass
```

## Part 3: Quality Assessment and Validation (60 minutes)

### 3.1 Understanding Quality Assessment for Tool-Augmented RAG

Build comprehensive quality assessment:

```python
class RAGQualityAssessor:
    """
    Comprehensive quality assessment for RAG systems with tools.
    """
    
    def __init__(self):
        self.quality_dimensions = self._initialize_quality_dimensions()
        self.assessment_methods = self._initialize_assessment_methods()
    
    async def assess_comprehensive_quality(self, 
                                         execution: RAGExecution,
                                         assessment_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive quality assessment of RAG execution.
        
        TODO: Implement this method to assess:
        1. Multi-dimensional quality evaluation
        2. Tool-specific quality metrics
        3. Integration quality assessment
        4. User satisfaction prediction
        
        Return comprehensive quality assessment.
        """
        # Your implementation here
        pass
```

### 3.2 Implementation Task

Implement the quality assessment methods:

```python
# TODO: Implement quality assessment methods
async def _assess_factual_accuracy(self, execution: RAGExecution) -> float:
    """Assess factual accuracy of generated response."""
    # Your implementation here
    pass

async def _assess_legal_soundness(self, execution: RAGExecution) -> float:
    """Assess legal soundness of analysis and recommendations."""
    # Your implementation here
    pass

async def _assess_tool_integration_quality(self, execution: RAGExecution) -> float:
    """Assess quality of tool integration and coordination."""
    # Your implementation here
    pass
```

## Part 4: Integration and Comprehensive Testing (60 minutes)

### 4.1 Complete System Integration

Integrate all components into a comprehensive RAG evaluation system:

```python
class ComprehensiveRAGEvaluationSystem:
    """
    Complete evaluation system for RAG with tool integration.
    """
    
    def __init__(self):
        self.rag_system = LegalRAGSystem()
        self.evaluation_framework = RAGToolEvaluationFramework(self.rag_system)
        self.performance_optimizer = RAGPerformanceOptimizer(self.rag_system)
        self.quality_assessor = RAGQualityAssessor()
    
    async def comprehensive_rag_evaluation(self, 
                                         test_queries: List[Dict[str, Any]],
                                         evaluation_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive evaluation of RAG system across multiple queries.
        """
        
        evaluation_results = []
        
        for query_data in test_queries:
            # Execute RAG query
            execution = await self.rag_system.execute_rag_query(
                query=query_data["query"],
                query_type=query_data["query_type"],
                generation_type=query_data["generation_type"]
            )
            
            # Evaluate execution
            evaluation_result = await self.evaluation_framework.evaluate_rag_execution(
                execution, query_data["ground_truth"], evaluation_config
            )
            
            # Assess quality
            quality_assessment = await self.quality_assessor.assess_comprehensive_quality(
                execution, evaluation_config
            )
            
            evaluation_results.append({
                'query': query_data["query"],
                'execution': execution,
                'evaluation': evaluation_result,
                'quality': quality_assessment
            })
        
        # Generate comprehensive report
        comprehensive_report = await self._generate_comprehensive_report(
            evaluation_results, evaluation_config
        )
        
        return comprehensive_report
```

### 4.2 Testing and Validation

Create comprehensive tests for your RAG evaluation system:

```python
async def test_comprehensive_rag_evaluation():
    """Test the complete RAG evaluation system."""
    
    evaluation_system = ComprehensiveRAGEvaluationSystem()
    
    # Create test queries
    test_queries = [
        {
            "query": "Analyze enforceability of non-compete clauses in California",
            "query_type": "contract_analysis",
            "generation_type": GenerationType.LEGAL_BRIEF,
            "ground_truth": {
                "relevant_documents": ["doc_001"],
                "expected_tools": ["document_analyzer", "case_law_search"],
                "key_concepts": ["non-compete", "California law"]
            }
        },
        {
            "query": "Securities compliance requirements for private placement",
            "query_type": "compliance_check",
            "generation_type": GenerationType.COMPLIANCE_ASSESSMENT,
            "ground_truth": {
                "relevant_documents": ["doc_002"],
                "expected_tools": ["regulatory_checker"],
                "key_concepts": ["securities", "private placement", "compliance"]
            }
        }
    ]
    
    # Run comprehensive evaluation
    evaluation_config = {
        "include_performance_analysis": True,
        "include_quality_assessment": True,
        "optimization_recommendations": True
    }
    
    results = await evaluation_system.comprehensive_rag_evaluation(
        test_queries, evaluation_config
    )
    
    # Analyze results
    print("Comprehensive RAG Evaluation Results:")
    print(json.dumps(results, indent=2, default=str))

# Run the test
if __name__ == "__main__":
    asyncio.run(test_comprehensive_rag_evaluation())
```

## Deliverables

1. **Complete RAG System** with integrated legal research tools
2. **Comprehensive Evaluation Framework** for tool-augmented RAG
3. **Performance Optimization System** with intelligent caching
4. **Quality Assessment Framework** for multi-dimensional evaluation
5. **Integrated Evaluation System** combining all components
6. **Test Suite** demonstrating capabilities with legal research scenarios
7. **Documentation** explaining implementation decisions and evaluation methodologies

## Evaluation Criteria

Your implementation will be evaluated on:

- **Completeness**: All required components implemented and functional
- **Integration Quality**: Effective coordination between RAG and tools
- **Evaluation Depth**: Comprehensive assessment across multiple dimensions
- **Performance**: Efficient execution with optimization strategies
- **Code Quality**: Clean, well-documented, and maintainable code
- **Testing**: Comprehensive test coverage with realistic scenarios

## Extension Opportunities

For additional challenge, consider implementing:

- **Multi-Modal RAG**: Extend to handle documents, images, and audio
- **Real-Time Learning**: Add capabilities for continuous improvement
- **Advanced Caching**: Implement sophisticated caching strategies
- **User Feedback Integration**: Add user feedback loops for quality improvement
- **Distributed RAG**: Extend to distributed and federated RAG systems

This exercise provides hands-on experience with the most sophisticated aspects of RAG system evaluation, preparing you for real-world implementation of advanced tool-augmented RAG systems in production legal and professional environments.

