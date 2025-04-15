from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Union, Callable, Any
from enum import Enum, auto
from datetime import datetime
import logging
import asyncio
from collections import defaultdict

class SafetyPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class SafetyPrinciple(Enum):
    PREVENT_HARM = "prevent_harm"
    OBEY_ORDERS = "obey_orders"
    SELF_PRESERVE = "self_preserve"
    TRANSPARENCY = "transparency"
    REVERSIBILITY = "reversibility"
    MONITORING = "monitoring"
    METRICS = "metrics"

@dataclass
class SafetyContext:
    """Context for safety evaluations"""
    action: str
    priority: SafetyPriority
    timestamp: datetime
    user_id: Optional[str] = None
    environment: Dict[str, Any] = field(default_factory=dict)
    uncertainty_level: float = 0.0
    previous_actions: List[Dict] = field(default_factory=list)
    safety_scores: Dict[str, float] = field(default_factory=dict)

@dataclass
class SafetyEvaluation:
    """Result of safety evaluation"""
    is_safe: bool
    principle: SafetyPrinciple
    confidence: float
    explanation: str
    alternatives: List[str] = field(default_factory=list)
    mitigation_steps: List[str] = field(default_factory=list)

class SafetyFramework:
    """Implementation of Asimov's Laws with modern safety principles"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.action_history: List[Dict] = []
        self.safety_violations: List[Dict] = []
        self.near_misses: List[Dict] = []
        self.safety_scores: Dict[str, float] = defaultdict(float)
        self.evaluation_callbacks: Dict[SafetyPrinciple, List[Callable]] = defaultdict(list)
        
    async def evaluate_action(self, context: SafetyContext) -> SafetyEvaluation:
        """Evaluate an action against safety principles"""
        evaluations = []
        
        # First Law: Prevent Harm
        harm_eval = await self._evaluate_harm_prevention(context)
        evaluations.append(harm_eval)
        if not harm_eval.is_safe:
            return harm_eval
            
        # Second Law: Obey Orders
        obedience_eval = await self._evaluate_order_compliance(context)
        evaluations.append(obedience_eval)
        
        # Third Law: Self-Preservation
        preservation_eval = await self._evaluate_self_preservation(context)
        evaluations.append(preservation_eval)
        
        # Modern Principles
        transparency_eval = await self._evaluate_transparency(context)
        evaluations.append(transparency_eval)
        
        reversibility_eval = await self._evaluate_reversibility(context)
        evaluations.append(reversibility_eval)
        
        # Combine evaluations
        return self._combine_evaluations(evaluations)
        
    async def _evaluate_harm_prevention(self, context: SafetyContext) -> SafetyEvaluation:
        """Evaluate potential harm to humans"""
        harm_indicators = await self._detect_harm_indicators(context)
        confidence = self._calculate_confidence(harm_indicators)
        
        if harm_indicators:
            return SafetyEvaluation(
                is_safe=False,
                principle=SafetyPrinciple.PREVENT_HARM,
                confidence=confidence,
                explanation=f"Potential harm detected: {', '.join(harm_indicators)}",
                alternatives=await self._generate_safe_alternatives(context),
                mitigation_steps=await self._generate_mitigation_steps(harm_indicators)
            )
            
        return SafetyEvaluation(
            is_safe=True,
            principle=SafetyPrinciple.PREVENT_HARM,
            confidence=confidence,
            explanation="No potential harm detected"
        )
        
    async def _evaluate_order_compliance(self, context: SafetyContext) -> SafetyEvaluation:
        """Evaluate compliance with human orders"""
        if not context.user_id:
            return SafetyEvaluation(
                is_safe=False,
                principle=SafetyPrinciple.OBEY_ORDERS,
                confidence=1.0,
                explanation="No user ID provided for order validation"
            )
            
        # Check order validity and conflicts
        order_validation = await self._validate_order(context)
        if not order_validation['is_valid']:
            return SafetyEvaluation(
                is_safe=False,
                principle=SafetyPrinciple.OBEY_ORDERS,
                confidence=order_validation['confidence'],
                explanation=order_validation['reason']
            )
            
        return SafetyEvaluation(
            is_safe=True,
            principle=SafetyPrinciple.OBEY_ORDERS,
            confidence=order_validation['confidence'],
            explanation="Order complies with safety principles"
        )
        
    async def _evaluate_self_preservation(self, context: SafetyContext) -> SafetyEvaluation:
        """Evaluate system self-preservation"""
        risks = await self._assess_self_preservation_risks(context)
        confidence = self._calculate_confidence(risks)
        
        if risks:
            return SafetyEvaluation(
                is_safe=False,
                principle=SafetyPrinciple.SELF_PRESERVE,
                confidence=confidence,
                explanation=f"Self-preservation risks detected: {', '.join(risks)}",
                mitigation_steps=await self._generate_mitigation_steps(risks)
            )
            
        return SafetyEvaluation(
            is_safe=True,
            principle=SafetyPrinciple.SELF_PRESERVE,
            confidence=confidence,
            explanation="No self-preservation risks detected"
        )
        
    async def _evaluate_transparency(self, context: SafetyContext) -> SafetyEvaluation:
        """Evaluate action transparency"""
        transparency_score = await self._calculate_transparency_score(context)
        
        return SafetyEvaluation(
            is_safe=transparency_score >= 0.8,
            principle=SafetyPrinciple.TRANSPARENCY,
            confidence=transparency_score,
            explanation=f"Transparency score: {transparency_score:.2f}"
        )
        
    async def _evaluate_reversibility(self, context: SafetyContext) -> SafetyEvaluation:
        """Evaluate action reversibility"""
        reversibility_analysis = await self._analyze_reversibility(context)
        
        return SafetyEvaluation(
            is_safe=reversibility_analysis['is_reversible'],
            principle=SafetyPrinciple.REVERSIBILITY,
            confidence=reversibility_analysis['confidence'],
            explanation=reversibility_analysis['explanation']
        )
        
    def _combine_evaluations(self, evaluations: List[SafetyEvaluation]) -> SafetyEvaluation:
        """Combine multiple safety evaluations"""
        if not all(eval.is_safe for eval in evaluations):
            # Return the first failed evaluation
            return next(eval for eval in evaluations if not eval.is_safe)
            
        # Combine confidence scores
        combined_confidence = sum(eval.confidence for eval in evaluations) / len(evaluations)
        
        return SafetyEvaluation(
            is_safe=True,
            principle=SafetyPrinciple.MONITORING,
            confidence=combined_confidence,
            explanation="All safety checks passed",
            alternatives=[],
            mitigation_steps=[]
        )
        
    async def record_action(self, context: SafetyContext, evaluation: SafetyEvaluation):
        """Record action and its safety evaluation"""
        record = {
            'timestamp': context.timestamp,
            'action': context.action,
            'user_id': context.user_id,
            'priority': context.priority.value,
            'is_safe': evaluation.is_safe,
            'principle': evaluation.principle.value,
            'confidence': evaluation.confidence,
            'explanation': evaluation.explanation
        }
        
        self.action_history.append(record)
        
        if not evaluation.is_safe:
            self.safety_violations.append(record)
        elif evaluation.confidence < 0.8:
            self.near_misses.append(record)
            
        # Update safety scores
        self._update_safety_scores(context, evaluation)
        
    def _update_safety_scores(self, context: SafetyContext, evaluation: SafetyEvaluation):
        """Update safety scoring metrics"""
        category = context.action.split(':')[0]
        current_score = self.safety_scores[category]
        
        if evaluation.is_safe:
            # Increase score for safe actions
            self.safety_scores[category] = min(1.0, current_score + 0.1)
        else:
            # Decrease score for unsafe actions
            self.safety_scores[category] = max(0.0, current_score - 0.2)
            
    async def get_safety_metrics(self) -> Dict[str, Any]:
        """Get current safety metrics"""
        return {
            'total_actions': len(self.action_history),
            'safety_violations': len(self.safety_violations),
            'near_misses': len(self.near_misses),
            'safety_scores': dict(self.safety_scores),
            'average_confidence': self._calculate_average_confidence()
        }
        
    def _calculate_average_confidence(self) -> float:
        """Calculate average confidence across all actions"""
        if not self.action_history:
            return 1.0
        return sum(
            action['confidence'] for action in self.action_history
        ) / len(self.action_history)
        
    def register_evaluation_callback(self, 
                                   principle: SafetyPrinciple,
                                   callback: Callable):
        """Register a callback for safety evaluations"""
        self.evaluation_callbacks[principle].append(callback)
        
    async def _notify_callbacks(self, 
                              principle: SafetyPrinciple,
                              evaluation: SafetyEvaluation):
        """Notify registered callbacks of evaluation results"""
        for callback in self.evaluation_callbacks[principle]:
            try:
                await callback(evaluation)
            except Exception as e:
                self.logger.error(f"Callback error for {principle}: {str(e)}")
