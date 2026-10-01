"""
Clinical Policy Decision Engine for BedSense AI.
Evaluates decision tiers (NORMAL, MONITOR, ALERT).
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
    Evaluates clinical alert policies.
    """

    def __init__(
        self,
        edge_sit_monitor_sec: float = 10.0,
        unknown_monitor_sec: float = 10.0,
        out_of_bed_alert_sec: float = 10.0,
    ):
        self.edge_sit_monitor_sec = edge_sit_monitor_sec
        self.unknown_monitor_sec = unknown_monitor_sec
        self.out_of_bed_alert_sec = out_of_bed_alert_sec

    def evaluate_event(
        self,
        event_type: str,
        out_of_bed_sec: float = 0.0,
        **kwargs,
    ) -> str:
        """Determines decision tier: NORMAL | MONITOR | ALERT."""
        if event_type == EVENT_MISSING or out_of_bed_sec >= self.out_of_bed_alert_sec:
            return DECISION_ALERT
        elif event_type in (EVENT_BED_EXIT, EVENT_BED_RETURN):
            return DECISION_MONITOR
        else:
            return DECISION_NORMAL


__all__ = ["PolicyEngine"]
