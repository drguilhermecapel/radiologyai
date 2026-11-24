#!/usr/bin/env python3
"""
MedAI Natural Language Interface System
Advanced AI-Human collaboration with natural language processing for radiological workflows
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import re
from pathlib import Path

class ConversationContext(Enum):
    """Types of conversation contexts"""
    CASE_REVIEW = "case_review"
    TEACHING = "teaching"
    CONSULTATION = "consultation"
    QUALITY_ASSURANCE = "quality_assurance"
    RESEARCH = "research"
    EMERGENCY = "emergency"

class IntentType(Enum):
    """Types of user intents"""
    QUESTION = "question"
    REQUEST_ANALYSIS = "request_analysis"
    PROVIDE_FEEDBACK = "provide_feedback"
    REQUEST_EXPLANATION = "request_explanation"
    COMPARE_CASES = "compare_cases"
    TEACHING_MOMENT = "teaching_moment"
    EMERGENCY_CONSULT = "emergency_consult"

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation"""
    timestamp: datetime
    speaker: str  # 'radiologist' or 'ai'
    message: str
    intent: Optional[IntentType]
    context: ConversationContext
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class ClinicalQuery:
    """Structured representation of a clinical query"""
    query_id: str
    timestamp: datetime
    radiologist_id: str
    query_text: str
    intent: IntentType
    context: ConversationContext
    case_references: List[str]
    urgency_level: str
    expected_response_type: str

@dataclass
class AIResponse:
    """AI-generated response to clinical queries"""
    response_id: str
    query_id: str
    timestamp: datetime
    response_text: str
    confidence: float
    evidence_references: List[str]
    visual_aids: List[str]
    follow_up_suggestions: List[str]
    uncertainty_indicators: List[str]

class NaturalLanguageProcessor:
    """Advanced NLP processor for medical conversations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.medical_vocabulary = self._load_medical_vocabulary()
        self.intent_classifier = self._initialize_intent_classifier()
        self.entity_extractor = self._initialize_entity_extractor()
        self.logger = logging.getLogger(__name__)
        
    def _load_medical_vocabulary(self) -> Dict[str, List[str]]:
        """Load medical terminology and synonyms"""
        return {
            'anatomy': ['lung', 'heart', 'brain', 'liver', 'kidney', 'spine', 'chest', 'abdomen'],
            'pathology': ['pneumonia', 'tumor', 'fracture', 'hemorrhage', 'infarct', 'mass', 'nodule'],
            'modalities': ['CT', 'MRI', 'X-ray', 'ultrasound', 'PET', 'mammography'],
            'findings': ['normal', 'abnormal', 'suspicious', 'benign', 'malignant', 'acute', 'chronic'],
            'urgency': ['urgent', 'emergent', 'routine', 'stat', 'critical', 'immediate']
        }
    
    def _initialize_intent_classifier(self):
        """Initialize intent classification model"""
        return None  # Placeholder for actual model
    
    def _initialize_entity_extractor(self):
        """Initialize medical entity extraction model"""
        return None  # Placeholder for actual model
    
    def process_query(self, query_text: str, context: ConversationContext) -> ClinicalQuery:
        """Process natural language query from radiologist"""
        
        intent = self._classify_intent(query_text)
        entities = self._extract_entities(query_text)
        urgency = self._assess_urgency(query_text)
        case_refs = self._extract_case_references(query_text)
        
        query = ClinicalQuery(
            query_id=f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            radiologist_id="current_user",  # Would be from session
            query_text=query_text,
            intent=intent,
            context=context,
            case_references=case_refs,
            urgency_level=urgency,
            expected_response_type=self._determine_response_type(intent)
        )
        
        return query
    
    def _classify_intent(self, text: str) -> IntentType:
        """Classify the intent of the user's message"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['what', 'why', 'how', 'explain']):
            return IntentType.QUESTION
        elif any(word in text_lower for word in ['analyze', 'look at', 'check', 'examine']):
            return IntentType.REQUEST_ANALYSIS
        elif any(word in text_lower for word in ['wrong', 'correct', 'actually', 'disagree']):
            return IntentType.PROVIDE_FEEDBACK
        elif any(word in text_lower for word in ['compare', 'similar', 'difference']):
            return IntentType.COMPARE_CASES
        elif any(word in text_lower for word in ['emergency', 'urgent', 'stat', 'critical']):
            return IntentType.EMERGENCY_CONSULT
        else:
            return IntentType.QUESTION
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities from text"""
        entities = {
            'anatomy': [],
            'pathology': [],
            'modalities': [],
            'findings': []
        }
        
        text_lower = text.lower()
        for category, terms in self.medical_vocabulary.items():
            for term in terms:
                if term in text_lower:
                    entities[category].append(term)
        
        return entities
    
    def _assess_urgency(self, text: str) -> str:
        """Assess urgency level from text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['emergency', 'urgent', 'stat', 'critical', 'immediate']):
            return 'high'
        elif any(word in text_lower for word in ['soon', 'priority', 'important']):
            return 'medium'
        else:
            return 'low'
    
    def _extract_case_references(self, text: str) -> List[str]:
        """Extract case/study references from text"""
        case_pattern = r'case\s+(\w+)|study\s+(\w+)|patient\s+(\w+)'
        matches = re.findall(case_pattern, text.lower())
        return [match for group in matches for match in group if match]
    
    def _determine_response_type(self, intent: IntentType) -> str:
        """Determine expected response type based on intent"""
        response_types = {
            IntentType.QUESTION: 'explanation',
            IntentType.REQUEST_ANALYSIS: 'analysis_results',
            IntentType.PROVIDE_FEEDBACK: 'acknowledgment',
            IntentType.REQUEST_EXPLANATION: 'detailed_explanation',
            IntentType.COMPARE_CASES: 'comparison_report',
            IntentType.TEACHING_MOMENT: 'educational_content',
            IntentType.EMERGENCY_CONSULT: 'urgent_analysis'
        }
        return response_types.get(intent, 'general_response')

class ConversationalAI:
    """AI system for natural language conversations with radiologists"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nlp_processor = NaturalLanguageProcessor(config)
        self.knowledge_base = self._initialize_knowledge_base()
        self.conversation_history = []
        self.logger = logging.getLogger(__name__)
        
    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """Initialize medical knowledge base"""
        return {
            'guidelines': {},
            'case_studies': {},
            'research_papers': {},
            'protocols': {}
        }
    
    def generate_response(self, query: ClinicalQuery, 
                         conversation_history: List[ConversationTurn]) -> AIResponse:
        """Generate AI response to clinical query"""
        
        response_text = self._generate_response_text(query, conversation_history)
        
        confidence = self._calculate_response_confidence(query, response_text)
        
        evidence_refs = self._gather_evidence_references(query)
        
        visual_aids = self._suggest_visual_aids(query)
        
        follow_ups = self._generate_follow_up_suggestions(query)
        
        uncertainties = self._identify_uncertainties(query, response_text)
        
        response = AIResponse(
            response_id=f"resp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            query_id=query.query_id,
            timestamp=datetime.now(),
            response_text=response_text,
            confidence=confidence,
            evidence_references=evidence_refs,
            visual_aids=visual_aids,
            follow_up_suggestions=follow_ups,
            uncertainty_indicators=uncertainties
        )
        
        return response
    
    def _generate_response_text(self, query: ClinicalQuery, 
                               history: List[ConversationTurn]) -> str:
        """Generate natural language response text"""
        
        if query.intent == IntentType.QUESTION:
            return self._generate_explanatory_response(query)
        elif query.intent == IntentType.REQUEST_ANALYSIS:
            return self._generate_analysis_response(query)
        elif query.intent == IntentType.PROVIDE_FEEDBACK:
            return self._generate_feedback_acknowledgment(query)
        elif query.intent == IntentType.COMPARE_CASES:
            return self._generate_comparison_response(query)
        elif query.intent == IntentType.EMERGENCY_CONSULT:
            return self._generate_emergency_response(query)
        else:
            return self._generate_general_response(query)
    
    def _generate_explanatory_response(self, query: ClinicalQuery) -> str:
        """Generate explanatory response for questions"""
        return f"Based on your question about {query.query_text}, I can provide the following explanation..."
    
    def _generate_analysis_response(self, query: ClinicalQuery) -> str:
        """Generate analysis response"""
        return f"I've analyzed the case you mentioned. Here are my findings..."
    
    def _generate_feedback_acknowledgment(self, query: ClinicalQuery) -> str:
        """Generate feedback acknowledgment"""
        return f"Thank you for the feedback. I understand your correction and will incorporate this learning..."
    
    def _generate_comparison_response(self, query: ClinicalQuery) -> str:
        """Generate case comparison response"""
        return f"Comparing the cases you mentioned, I observe the following similarities and differences..."
    
    def _generate_emergency_response(self, query: ClinicalQuery) -> str:
        """Generate emergency consultation response"""
        return f"This appears to be an urgent case. Based on the findings, I recommend immediate..."
    
    def _generate_general_response(self, query: ClinicalQuery) -> str:
        """Generate general response"""
        return f"I understand your query. Let me provide you with relevant information..."
    
    def _calculate_response_confidence(self, query: ClinicalQuery, response: str) -> float:
        """Calculate confidence in the response"""
        base_confidence = 0.8
        
        if query.urgency_level == 'high':
            base_confidence *= 0.9  # Slightly lower confidence for urgent cases
        
        if len(query.case_references) > 0:
            base_confidence *= 1.1  # Higher confidence with case references
        
        return min(0.95, base_confidence)
    
    def _gather_evidence_references(self, query: ClinicalQuery) -> List[str]:
        """Gather relevant evidence references"""
        return [
            "Radiology Guidelines 2024",
            "ACR Appropriateness Criteria",
            "Recent Literature Review"
        ]
    
    def _suggest_visual_aids(self, query: ClinicalQuery) -> List[str]:
        """Suggest relevant visual aids"""
        return [
            "attention_heatmap.png",
            "comparison_overlay.png",
            "measurement_annotations.png"
        ]
    
    def _generate_follow_up_suggestions(self, query: ClinicalQuery) -> List[str]:
        """Generate follow-up suggestions"""
        suggestions = []
        
        if query.intent == IntentType.REQUEST_ANALYSIS:
            suggestions.append("Would you like me to analyze additional views?")
            suggestions.append("Should I compare with prior studies?")
        
        if query.urgency_level == 'high':
            suggestions.append("Should I alert the attending physician?")
            suggestions.append("Do you need additional emergency protocols?")
        
        return suggestions
    
    def _identify_uncertainties(self, query: ClinicalQuery, response: str) -> List[str]:
        """Identify areas of uncertainty in the response"""
        uncertainties = []
        
        if query.urgency_level == 'high':
            uncertainties.append("High urgency case - recommend human verification")
        
        if len(query.case_references) == 0:
            uncertainties.append("No specific case referenced - general guidance provided")
        
        return uncertainties

class ConversationManager:
    """Manages conversation flow and context"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nlp_processor = NaturalLanguageProcessor(config)
        self.conversational_ai = ConversationalAI(config)
        self.active_conversations = {}
        self.logger = logging.getLogger(__name__)
        
    def start_conversation(self, radiologist_id: str, 
                          context: ConversationContext) -> str:
        """Start a new conversation session"""
        conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{radiologist_id}"
        
        self.active_conversations[conversation_id] = {
            'radiologist_id': radiologist_id,
            'context': context,
            'start_time': datetime.now(),
            'turns': [],
            'current_cases': []
        }
        
        return conversation_id
    
    def process_message(self, conversation_id: str, message: str) -> AIResponse:
        """Process a message in an ongoing conversation"""
        
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        conversation = self.active_conversations[conversation_id]
        
        query = self.nlp_processor.process_query(message, conversation['context'])
        
        response = self.conversational_ai.generate_response(
            query, conversation['turns']
        )
        
        user_turn = ConversationTurn(
            timestamp=datetime.now(),
            speaker='radiologist',
            message=message,
            intent=query.intent,
            context=conversation['context'],
            confidence=1.0,
            metadata={'query_id': query.query_id}
        )
        
        ai_turn = ConversationTurn(
            timestamp=datetime.now(),
            speaker='ai',
            message=response.response_text,
            intent=None,
            context=conversation['context'],
            confidence=response.confidence,
            metadata={'response_id': response.response_id}
        )
        
        conversation['turns'].extend([user_turn, ai_turn])
        
        return response
    
    def end_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """End conversation and return summary"""
        
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        conversation = self.active_conversations[conversation_id]
        
        summary = {
            'conversation_id': conversation_id,
            'duration': datetime.now() - conversation['start_time'],
            'total_turns': len(conversation['turns']),
            'context': conversation['context'],
            'key_topics': self._extract_key_topics(conversation['turns']),
            'action_items': self._extract_action_items(conversation['turns'])
        }
        
        del self.active_conversations[conversation_id]
        
        return summary
    
    def _extract_key_topics(self, turns: List[ConversationTurn]) -> List[str]:
        """Extract key topics from conversation"""
        topics = set()
        
        for turn in turns:
            if turn.speaker == 'radiologist':
                entities = self.nlp_processor._extract_entities(turn.message)
                for category, terms in entities.items():
                    topics.update(terms)
        
        return list(topics)
    
    def _extract_action_items(self, turns: List[ConversationTurn]) -> List[str]:
        """Extract action items from conversation"""
        action_items = []
        
        for turn in turns:
            if turn.speaker == 'ai' and 'follow_up_suggestions' in turn.metadata:
                action_items.extend(turn.metadata['follow_up_suggestions'])
        
        return action_items

class SmartWorkflowEngine:
    """Intelligent workflow optimization for radiological tasks"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.workflow_patterns = self._initialize_workflow_patterns()
        self.performance_metrics = {}
        self.logger = logging.getLogger(__name__)
        
    def _initialize_workflow_patterns(self) -> Dict[str, Any]:
        """Initialize common workflow patterns"""
        return {
            'emergency_workflow': {
                'steps': ['triage', 'urgent_analysis', 'notification', 'documentation'],
                'time_limits': {'triage': 2, 'urgent_analysis': 5, 'notification': 1}
            },
            'routine_workflow': {
                'steps': ['queue_management', 'analysis', 'review', 'reporting'],
                'optimization_targets': ['throughput', 'accuracy', 'consistency']
            },
            'teaching_workflow': {
                'steps': ['case_selection', 'guided_analysis', 'discussion', 'assessment'],
                'learning_objectives': ['pattern_recognition', 'differential_diagnosis']
            }
        }
    
    def optimize_workflow(self, context: ConversationContext, 
                         current_workload: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize workflow based on context and current workload"""
        
        if context == ConversationContext.EMERGENCY:
            return self._optimize_emergency_workflow(current_workload)
        elif context == ConversationContext.TEACHING:
            return self._optimize_teaching_workflow(current_workload)
        else:
            return self._optimize_routine_workflow(current_workload)
    
    def _optimize_emergency_workflow(self, workload: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize emergency workflow"""
        return {
            'priority_queue': 'emergency_first',
            'resource_allocation': 'maximum_available',
            'notification_channels': ['immediate_alert', 'phone_call'],
            'escalation_rules': 'automatic_after_2min'
        }
    
    def _optimize_teaching_workflow(self, workload: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize teaching workflow"""
        return {
            'case_selection': 'educational_value_high',
            'interaction_mode': 'guided_discovery',
            'feedback_frequency': 'real_time',
            'assessment_integration': 'continuous'
        }
    
    def _optimize_routine_workflow(self, workload: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize routine workflow"""
        return {
            'batch_processing': 'enabled',
            'quality_checks': 'automated',
            'reporting_schedule': 'optimized',
            'resource_balancing': 'dynamic'
        }

class NaturalLanguageInterface:
    """Main interface for natural language interactions"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.conversation_manager = ConversationManager(self.config)
        self.workflow_engine = SmartWorkflowEngine(self.config)
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load system configuration"""
        default_config = {
            'max_conversation_length': 100,
            'response_timeout': 30,
            'confidence_threshold': 0.7,
            'enable_learning': True,
            'log_conversations': True,
            'workflow_optimization': True
        }
        
        if config_path and Path(config_path).exists():
            pass
            
        return default_config
    
    def start_session(self, radiologist_id: str, 
                     context: ConversationContext = ConversationContext.CASE_REVIEW) -> str:
        """Start a new conversation session"""
        return self.conversation_manager.start_conversation(radiologist_id, context)
    
    def chat(self, conversation_id: str, message: str) -> Dict[str, Any]:
        """Process a chat message and return response"""
        
        response = self.conversation_manager.process_message(conversation_id, message)
        
        workflow_suggestions = {}
        if self.config.get('workflow_optimization', False):
            conversation = self.conversation_manager.active_conversations.get(conversation_id)
            if conversation:
                workflow_suggestions = self.workflow_engine.optimize_workflow(
                    conversation['context'], {'current_cases': conversation['current_cases']}
                )
        
        return {
            'response_text': response.response_text,
            'confidence': response.confidence,
            'visual_aids': response.visual_aids,
            'follow_up_suggestions': response.follow_up_suggestions,
            'uncertainty_indicators': response.uncertainty_indicators,
            'evidence_references': response.evidence_references,
            'workflow_suggestions': workflow_suggestions
        }
    
    def end_session(self, conversation_id: str) -> Dict[str, Any]:
        """End conversation session"""
        return self.conversation_manager.end_conversation(conversation_id)
    
    def get_conversation_analytics(self, conversation_id: str) -> Dict[str, Any]:
        """Get analytics for ongoing conversation"""
        
        if conversation_id not in self.conversation_manager.active_conversations:
            return {}
        
        conversation = self.conversation_manager.active_conversations[conversation_id]
        
        return {
            'duration': datetime.now() - conversation['start_time'],
            'turn_count': len(conversation['turns']),
            'context': conversation['context'].value,
            'avg_confidence': np.mean([turn.confidence for turn in conversation['turns'] if turn.speaker == 'ai']),
            'intent_distribution': self._calculate_intent_distribution(conversation['turns'])
        }
    
    def _calculate_intent_distribution(self, turns: List[ConversationTurn]) -> Dict[str, int]:
        """Calculate distribution of intents in conversation"""
        intent_counts = {}
        
        for turn in turns:
            if turn.intent:
                intent_name = turn.intent.value
                intent_counts[intent_name] = intent_counts.get(intent_name, 0) + 1
        
        return intent_counts

def main():
    """Example usage of natural language interface"""
    
    interface = NaturalLanguageInterface()
    
    session_id = interface.start_session("radiologist_001", ConversationContext.CASE_REVIEW)
    
    messages = [
        "Can you analyze this chest X-ray for pneumonia?",
        "What makes you think there's consolidation in the right lower lobe?",
        "I disagree, I think this is just atelectasis",
        "Can you compare this with the patient's prior study from last week?"
    ]
    
    for message in messages:
        response = interface.chat(session_id, message)
        print(f"User: {message}")
        print(f"AI: {response['response_text']}")
        print(f"Confidence: {response['confidence']:.2f}")
        if response['workflow_suggestions']:
            print(f"Workflow: {response['workflow_suggestions']}")
        print("---")
    
    analytics = interface.get_conversation_analytics(session_id)
    print(f"Conversation Analytics: {analytics}")
    
    summary = interface.end_session(session_id)
    print(f"Conversation Summary: {summary}")

if __name__ == "__main__":
    main()
