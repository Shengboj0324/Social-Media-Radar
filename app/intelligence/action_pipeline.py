"""Action pipeline orchestrator for Phase 3.

This module orchestrates the complete action workflow:
1. Action Ranking: Score and prioritize signals
2. Response Planning: Generate response drafts
3. Policy Checking: Validate drafts against policies
4. Queue Management: Persist and retrieve actions

Integrates all Phase 3 components into a single workflow.
"""

import logging
from typing import List, Dict, Optional
from uuid import UUID

from app.domain.normalized_models import NormalizedObservation
from app.domain.inference_models import SignalInference
from app.domain.action_models import ActionableSignal, ActionStatus
from app.intelligence.action_ranker import ActionRanker
from app.intelligence.response_planner import ResponsePlanner
from app.intelligence.policy_checker import PolicyChecker

logger = logging.getLogger(__name__)


class ActionPipeline:
    """Action pipeline orchestrator.
    
    Orchestrates the complete action workflow from signal inference
    to actionable signals with response drafts and policy validation.
    """
    
    def __init__(
        self,
        ranker: Optional[ActionRanker] = None,
        planner: Optional[ResponsePlanner] = None,
        policy_checker: Optional[PolicyChecker] = None,
    ):
        """Initialize action pipeline.
        
        Args:
            ranker: Action ranker (creates default if None)
            planner: Response planner (creates default if None)
            policy_checker: Policy checker (creates default if None)
        """
        self.ranker = ranker or ActionRanker()
        self.planner = planner or ResponsePlanner()
        self.policy_checker = policy_checker or PolicyChecker()
        
        logger.info("ActionPipeline initialized")
    
    async def process_inference(
        self,
        inference: SignalInference,
        observation: NormalizedObservation,
    ) -> Optional[ActionableSignal]:
        """Process a single signal inference into an actionable signal.
        
        Args:
            inference: Signal inference from Phase 2
            observation: Normalized observation
            
        Returns:
            ActionableSignal if worthy of action, None otherwise
        """
        # Stage 1: Rank action
        action = self.ranker.rank_action(inference, observation)
        
        if not action:
            logger.debug(f"Inference {inference.id} not worthy of action")
            return None
        
        logger.info(
            f"Ranked action {action.id}: "
            f"priority={action.priority.value}, "
            f"score={action.priority_score:.2f}"
        )
        
        # Stage 2: Plan response
        action = await self.planner.plan_response(action, observation, inference)
        
        logger.info(
            f"Generated {len(action.response_drafts)} drafts for action {action.id}"
        )
        
        # Stage 3: Policy check all drafts
        for draft in action.response_drafts:
            violations = self.policy_checker.check_draft(draft, action, observation)
            draft.policy_violations = violations
            
            # Update action-level policy violations
            for violation in violations:
                if violation not in action.policy_violations:
                    action.policy_violations.append(violation)
        
        # Stage 4: Determine if safe to auto-respond
        action.safe_to_auto_respond = self._is_safe_to_auto_respond(action)
        
        # Stage 5: Update status
        if action.safe_to_auto_respond:
            action.status = ActionStatus.APPROVED
        else:
            action.status = ActionStatus.PENDING_REVIEW
            action.requires_human_review = True
        
        logger.info(
            f"Action {action.id} ready: "
            f"status={action.status.value}, "
            f"safe_auto={action.safe_to_auto_respond}"
        )
        
        return action
    
    async def process_batch(
        self,
        inferences: List[SignalInference],
        observations: Dict[str, NormalizedObservation],
    ) -> List[ActionableSignal]:
        """Process a batch of signal inferences.
        
        Args:
            inferences: List of signal inferences
            observations: Dict mapping observation ID to observation
            
        Returns:
            List of actionable signals, sorted by priority
        """
        actions = []
        
        for inference in inferences:
            observation = observations.get(str(inference.normalized_observation_id))
            if not observation:
                logger.warning(
                    f"No observation found for inference {inference.id}"
                )
                continue
            
            action = await self.process_inference(inference, observation)
            if action:
                actions.append(action)
        
        # Sort by priority score (descending)
        actions.sort(key=lambda a: a.priority_score, reverse=True)
        
        logger.info(
            f"Processed {len(inferences)} inferences -> {len(actions)} actions"
        )
        
        return actions
    
    def _is_safe_to_auto_respond(self, action: ActionableSignal) -> bool:
        """Determine if action is safe for auto-response.
        
        Args:
            action: Actionable signal
            
        Returns:
            True if safe to auto-respond, False otherwise
        """
        # Never auto-respond to critical priority
        if action.priority.value == "critical":
            return False
        
        # Never auto-respond if there are blocking violations
        for violation in action.policy_violations:
            if violation.blocking or violation.severity == "critical":
                return False
        
        # Never auto-respond if no drafts passed policy check
        if not action.response_drafts:
            return False
        
        safe_drafts = [
            d for d in action.response_drafts
            if self.policy_checker.is_safe_to_execute(d)
        ]
        
        if not safe_drafts:
            return False
        
        # Only auto-respond for low-risk, high-confidence signals
        if action.risk_score > 0.5:
            return False
        
        if action.signal_confidence < 0.8:
            return False
        
        return True

