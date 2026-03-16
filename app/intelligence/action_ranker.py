"""Action ranking system for prioritizing actionable signals.

This module implements multi-dimensional action ranking:
- Priority scoring (overall priority)
- Opportunity scoring (business value potential)
- Urgency scoring (time sensitivity)
- Risk scoring (risk if not addressed)

Combines multiple signals to produce a calibrated priority score.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timezone

from app.domain.normalized_models import NormalizedObservation
from app.domain.inference_models import SignalInference, SignalType
from app.domain.action_models import ActionableSignal, ActionPriority, ActionStatus, ResponseChannel
from app.core.models import SourcePlatform

logger = logging.getLogger(__name__)


class ActionRanker:
    """Multi-dimensional action ranking system.

    Scores signals across 4 dimensions:
    1. Opportunity: Business value potential
    2. Urgency: Time sensitivity
    3. Risk: Risk if not addressed
    4. Priority: Overall priority (weighted combination)

    Dispatch tables (``_OPPORTUNITY_MAP``, ``_URGENCY_MAP``, ``_RISK_MAP``,
    ``_CHANNEL_MAP``) replace if/elif chains for O(1) lookup and zero-friction
    extensibility — add a new ``SignalType`` in one place only.
    """

    # ------------------------------------------------------------------
    # Dispatch tables — O(1) lookup, easy to extend
    # ------------------------------------------------------------------

    # Opportunity scores (base, before engagement boosts)
    _OPPORTUNITY_MAP: Dict[SignalType, float] = {
        SignalType.ALTERNATIVE_SEEKING: 0.9,
        SignalType.COMPETITOR_MENTION: 0.9,
        SignalType.EXPANSION_OPPORTUNITY: 0.9,
        SignalType.UPSELL_OPPORTUNITY: 0.9,
        SignalType.PARTNERSHIP_OPPORTUNITY: 0.9,
        SignalType.FEATURE_REQUEST: 0.7,
        SignalType.INTEGRATION_REQUEST: 0.7,
        SignalType.PRICE_SENSITIVITY: 0.7,
        SignalType.SUPPORT_REQUEST: 0.4,
        SignalType.BUG_REPORT: 0.4,
        SignalType.COMPLAINT: 0.4,
    }
    _OPPORTUNITY_DEFAULT = 0.5

    # Urgency scores (base, before freshness/velocity boosts)
    _URGENCY_MAP: Dict[SignalType, float] = {
        SignalType.CHURN_RISK: 0.9,
        SignalType.SECURITY_CONCERN: 0.9,
        SignalType.LEGAL_RISK: 0.9,
        SignalType.REPUTATION_RISK: 0.9,
        SignalType.COMPLAINT: 0.7,
        SignalType.BUG_REPORT: 0.7,
        SignalType.ALTERNATIVE_SEEKING: 0.7,
        SignalType.PRAISE: 0.3,
        SignalType.FEATURE_REQUEST: 0.3,
    }
    _URGENCY_DEFAULT = 0.5

    # Risk scores (base, before platform/virality boosts)
    _RISK_MAP: Dict[SignalType, float] = {
        SignalType.CHURN_RISK: 0.95,
        SignalType.SECURITY_CONCERN: 0.95,
        SignalType.LEGAL_RISK: 0.95,
        SignalType.REPUTATION_RISK: 0.95,
        SignalType.COMPLAINT: 0.6,
        SignalType.COMPETITOR_MENTION: 0.6,
        SignalType.ALTERNATIVE_SEEKING: 0.6,
        SignalType.PRAISE: 0.2,
        SignalType.FEATURE_REQUEST: 0.2,
        SignalType.SUPPORT_REQUEST: 0.2,
    }
    _RISK_DEFAULT = 0.3

    # Channel dispatch table
    _CHANNEL_MAP: Dict[SignalType, "ResponseChannel"] = {}  # populated after imports

    def __init__(
        self,
        opportunity_weight: float = 0.35,
        urgency_weight: float = 0.30,
        risk_weight: float = 0.35,
        min_confidence_threshold: float = 0.5,
    ):
        """Initialize action ranker.
        
        Args:
            opportunity_weight: Weight for opportunity score
            urgency_weight: Weight for urgency score
            risk_weight: Weight for risk score
            min_confidence_threshold: Minimum confidence to create action
        """
        self.opportunity_weight = opportunity_weight
        self.urgency_weight = urgency_weight
        self.risk_weight = risk_weight
        self.min_confidence_threshold = min_confidence_threshold
        
        # Validate weights sum to 1.0
        total_weight = opportunity_weight + urgency_weight + risk_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                f"Weights sum to {total_weight}, not 1.0. "
                f"Normalizing weights."
            )
            self.opportunity_weight /= total_weight
            self.urgency_weight /= total_weight
            self.risk_weight /= total_weight
        
        # Initialise dispatch table eagerly (idempotent)
        self._init_dispatch()

        logger.info(
            f"ActionRanker initialized: "
            f"opp={self.opportunity_weight:.2f}, "
            f"urg={self.urgency_weight:.2f}, "
            f"risk={self.risk_weight:.2f}"
        )
    
    def rank_action(
        self,
        inference: SignalInference,
        observation: NormalizedObservation,
    ) -> Optional[ActionableSignal]:
        """Rank a signal inference and create actionable signal if worthy.
        
        Args:
            inference: Signal inference from Phase 2
            observation: Normalized observation
            
        Returns:
            ActionableSignal if worthy of action, None otherwise
        """
        # Check if inference is abstained
        if inference.abstained:
            logger.debug(f"Skipping abstained inference {inference.id}")
            return None
        
        # Check if confidence meets threshold
        if not inference.top_prediction:
            logger.debug(f"No top prediction for inference {inference.id}")
            return None
        
        if inference.top_prediction.probability < self.min_confidence_threshold:
            logger.debug(
                f"Confidence {inference.top_prediction.probability:.2f} "
                f"below threshold {self.min_confidence_threshold}"
            )
            return None
        
        # Compute scores
        opportunity_score = self._compute_opportunity_score(inference, observation)
        urgency_score = self._compute_urgency_score(inference, observation)
        risk_score = self._compute_risk_score(inference, observation)
        
        # Compute overall priority score
        priority_score = (
            self.opportunity_weight * opportunity_score +
            self.urgency_weight * urgency_score +
            self.risk_weight * risk_score
        )
        
        # Determine priority level
        priority = self._determine_priority_level(priority_score)
        
        # Determine recommended channel
        recommended_channel = self._determine_channel(inference, observation)
        
        # Create actionable signal
        action = ActionableSignal(
            signal_inference_id=inference.id,
            normalized_observation_id=observation.id,
            user_id=observation.user_id,
            signal_type=inference.top_prediction.signal_type,
            signal_confidence=inference.top_prediction.probability,
            priority=priority,
            priority_score=priority_score,
            opportunity_score=opportunity_score,
            urgency_score=urgency_score,
            risk_score=risk_score,
            recommended_channel=recommended_channel,
            status=ActionStatus.NEW,
        )
        
        logger.info(
            f"Created action {action.id}: "
            f"type={action.signal_type.value}, "
            f"priority={priority.value}, "
            f"score={priority_score:.2f}"
        )
        
        return action
    
    def rank_batch(
        self,
        inferences: List[SignalInference],
        observations: Dict[str, NormalizedObservation],
    ) -> List[ActionableSignal]:
        """Rank a batch of inferences.
        
        Args:
            inferences: List of signal inferences
            observations: Dict mapping observation ID to observation
            
        Returns:
            List of actionable signals, sorted by priority score
        """
        actions = []
        
        for inference in inferences:
            observation = observations.get(str(inference.normalized_observation_id))
            if not observation:
                logger.warning(
                    f"No observation found for inference {inference.id}"
                )
                continue
            
            action = self.rank_action(inference, observation)
            if action:
                actions.append(action)
        
        # Sort by priority score (descending)
        actions.sort(key=lambda a: a.priority_score, reverse=True)
        
        logger.info(f"Ranked {len(actions)} actions from {len(inferences)} inferences")
        return actions

    def _compute_opportunity_score(
        self,
        inference: SignalInference,
        observation: NormalizedObservation,
    ) -> float:
        """Compute business opportunity score using the dispatch table.

        Args:
            inference: Calibrated signal inference.
            observation: Normalised observation.

        Returns:
            Opportunity score in [0.0, 1.0].
        """
        if not inference.top_prediction:
            return self._OPPORTUNITY_DEFAULT

        score = self._OPPORTUNITY_MAP.get(
            inference.top_prediction.signal_type, self._OPPORTUNITY_DEFAULT
        )

        # Engagement boosts (additive, clamped)
        if observation.engagement_velocity and observation.engagement_velocity > 10:
            score = min(1.0, score + 0.1)
        if observation.virality_score and observation.virality_score > 0.5:
            score = min(1.0, score + 0.1)

        return score

    def _compute_urgency_score(
        self,
        inference: SignalInference,
        observation: NormalizedObservation,
    ) -> float:
        """Compute time-sensitivity score using the dispatch table.

        Args:
            inference: Calibrated signal inference.
            observation: Normalised observation.

        Returns:
            Urgency score in [0.0, 1.0].
        """
        if not inference.top_prediction:
            return self._URGENCY_DEFAULT

        score = self._URGENCY_MAP.get(
            inference.top_prediction.signal_type, self._URGENCY_DEFAULT
        )

        # Freshness boost/penalty
        if observation.published_at:
            hours_old = (
                datetime.now(timezone.utc) - observation.published_at
            ).total_seconds() / 3600
            if hours_old < 1:
                score = min(1.0, score + 0.2)
            elif hours_old < 6:
                score = min(1.0, score + 0.1)
            elif hours_old > 48:
                score = max(0.0, score - 0.2)

        if observation.engagement_velocity and observation.engagement_velocity > 20:
            score = min(1.0, score + 0.1)

        return score

    def _compute_risk_score(
        self,
        inference: SignalInference,
        observation: NormalizedObservation,
    ) -> float:
        """Compute risk-if-unaddressed score using the dispatch table.

        Args:
            inference: Calibrated signal inference.
            observation: Normalised observation.

        Returns:
            Risk score in [0.0, 1.0].
        """
        if not inference.top_prediction:
            return self._RISK_DEFAULT

        score = self._RISK_MAP.get(
            inference.top_prediction.signal_type, self._RISK_DEFAULT
        )

        # Public platform visibility boost
        _PUBLIC_PLATFORMS = {
            SourcePlatform.REDDIT, SourcePlatform.YOUTUBE,
            SourcePlatform.TIKTOK, SourcePlatform.FACEBOOK, SourcePlatform.INSTAGRAM,
        }
        if observation.source_platform in _PUBLIC_PLATFORMS:
            score = min(1.0, score + 0.1)

        if observation.virality_score and observation.virality_score > 0.7:
            score = min(1.0, score + 0.15)

        return score

    def _determine_priority_level(self, priority_score: float) -> ActionPriority:
        """Determine priority level from score.

        Args:
            priority_score: Overall priority score

        Returns:
            ActionPriority enum
        """
        if priority_score >= 0.8:
            return ActionPriority.CRITICAL
        elif priority_score >= 0.6:
            return ActionPriority.HIGH
        elif priority_score >= 0.4:
            return ActionPriority.MEDIUM
        elif priority_score >= 0.2:
            return ActionPriority.LOW
        else:
            return ActionPriority.MONITOR

    # Channel dispatch table — defined at class body level after ResponseChannel is imported.
    # Maps every SignalType to its recommended ResponseChannel for O(1) lookup.
    _CHANNEL_DISPATCH: Dict[SignalType, "ResponseChannel"] = {}  # populated in _init_dispatch

    @classmethod
    def _init_dispatch(cls) -> None:
        """Populate the channel dispatch table (called once at first instantiation)."""
        if cls._CHANNEL_DISPATCH:
            return  # Already initialised
        cls._CHANNEL_DISPATCH = {
            # Public direct reply
            SignalType.ALTERNATIVE_SEEKING: ResponseChannel.DIRECT_REPLY,
            SignalType.COMPETITOR_MENTION: ResponseChannel.DIRECT_REPLY,
            SignalType.PRAISE: ResponseChannel.DIRECT_REPLY,
            SignalType.PRICE_SENSITIVITY: ResponseChannel.DIRECT_REPLY,
            # Sensitive — private DM
            SignalType.CHURN_RISK: ResponseChannel.DIRECT_MESSAGE,
            SignalType.COMPLAINT: ResponseChannel.DIRECT_MESSAGE,
            SignalType.SECURITY_CONCERN: ResponseChannel.DIRECT_MESSAGE,
            SignalType.LEGAL_RISK: ResponseChannel.DIRECT_MESSAGE,
            SignalType.REPUTATION_RISK: ResponseChannel.DIRECT_MESSAGE,
            # Business opportunity — email
            SignalType.PARTNERSHIP_OPPORTUNITY: ResponseChannel.EMAIL,
            SignalType.EXPANSION_OPPORTUNITY: ResponseChannel.EMAIL,
            SignalType.UPSELL_OPPORTUNITY: ResponseChannel.EMAIL,
            # Internal workflow
            SignalType.SUPPORT_REQUEST: ResponseChannel.INTERNAL_TICKET,
            SignalType.BUG_REPORT: ResponseChannel.INTERNAL_TICKET,
            SignalType.FEATURE_REQUEST: ResponseChannel.INTERNAL_TICKET,
            SignalType.INTEGRATION_REQUEST: ResponseChannel.INTERNAL_TICKET,
            # No response needed
            SignalType.UNCLEAR: ResponseChannel.NO_RESPONSE,
            SignalType.NOT_ACTIONABLE: ResponseChannel.NO_RESPONSE,
        }

    def _determine_channel(
        self,
        inference: SignalInference,
        observation: NormalizedObservation,
    ) -> ResponseChannel:
        """Determine recommended response channel via O(1) dispatch table lookup.

        Args:
            inference: Calibrated signal inference.
            observation: Normalised observation (not used currently; reserved for
                platform-specific channel overrides in future iterations).

        Returns:
            :class:`~app.domain.action_models.ResponseChannel` enum value.
        """
        if not inference.top_prediction:
            return ResponseChannel.NO_RESPONSE

        self._init_dispatch()
        return self._CHANNEL_DISPATCH.get(
            inference.top_prediction.signal_type,
            ResponseChannel.NO_RESPONSE,
        )

