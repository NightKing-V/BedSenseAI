"""
Clinical Policy Decision Engine for BedSense AI (§6.1 of Design Spec).
Evaluates decision tiers (NORMAL, MONITOR, ALERT) and caregiver suppression.
"""

import logging
from typing import Optional
from ..constants import (
    DECISION_ALERT,
    DECISION_MONITOR,
    DECISION_NORMAL,
    EVENT_BED_EXIT,
    EVENT_BED_RETURN,
    EVENT_MISSING,
)

logger = logging.getLogger("BedSensePolicy")


class PolicyEngine:
    """
    Evaluates clinical alert policies and caregiver suppression (§6.1 of Design Spec).
    """

    def __init__(
        self,
        edge_sit_monitor_sec: float = 10.0,
        unknown_monitor_sec: float = 10.0,
        out_of_bed_alert_sec: float = 10.0,
        suppress_with_caregiver: bool = True,
        caregiver_suppress_escalation: Optional[bool] = None,
    ):
        self.edge_sit_monitor_sec = edge_sit_monitor_sec
        self.unknown_monitor_sec = unknown_monitor_sec
        self.out_of_bed_alert_sec = out_of_bed_alert_sec
        if caregiver_suppress_escalation is not None:
            self.suppress_with_caregiver = caregiver_suppress_escalation
        else:
            self.suppress_with_caregiver = suppress_with_caregiver

    def evaluate_event(
        self,
        event_type: str,
        out_of_bed_sec: float = 0.0,
        has_caregiver: bool = False,
    ) -> str:
        """Determines decision tier: NORMAL | MONITOR | ALERT."""
        if event_type == EVENT_MISSING or out_of_bed_sec >= self.out_of_bed_alert_sec:
            decision = DECISION_ALERT
        elif event_type in (EVENT_BED_EXIT, EVENT_BED_RETURN):
            decision = DECISION_MONITOR
        else:
            decision = DECISION_NORMAL

        if has_caregiver and self.suppress_with_caregiver:
            decision = self.suppress_escalation(decision)

        return decision

    @staticmethod
    def suppress_escalation(decision: str) -> str:
        """Suppresses one escalation tier if caregiver is present."""
        if decision == DECISION_ALERT:
            return DECISION_MONITOR
        elif decision == DECISION_MONITOR:
            return DECISION_NORMAL
        return DECISION_NORMAL


__all__ = ["PolicyEngine"]
