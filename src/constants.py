"""
BedSense AI Constants (§3 of Design Spec).
"""

from typing import Dict, List, Set, Tuple

# 7 Canonical States
STATE_LYING_IN_BED = "LYING_IN_BED"
STATE_SITTING_ON_BED = "SITTING_ON_BED"
STATE_SITTING_OUTSIDE_BED = "SITTING_OUTSIDE_BED"
STATE_STANDING = "STANDING"
STATE_WALKING = "WALKING"
STATE_OUT_OF_BED = "OUT_OF_BED"
STATE_UNKNOWN = "UNKNOWN"

ALL_STATES: Set[str] = {
    STATE_LYING_IN_BED,
    STATE_SITTING_ON_BED,
    STATE_SITTING_OUTSIDE_BED,
    STATE_STANDING,
    STATE_WALKING,
    STATE_OUT_OF_BED,
    STATE_UNKNOWN,
}

# Events & Alert Decisions
EVENT_BED_EXIT = "bed_exit"
EVENT_BED_RETURN = "bed_return"

DECISION_NORMAL = "NORMAL"
DECISION_MONITOR = "MONITOR"
DECISION_ALERT = "ALERT"

# ---------------------------------------------------------------------------
# COCO 17 Keypoint Indices
# ---------------------------------------------------------------------------
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

KEYPOINT_NAMES: List[str] = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

SKELETON_PAIRS: List[Tuple[int, int]] = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
]

PERSON_CLASS_ID = 0

# BGR UI Colors
STATE_COLORS: Dict[str, Tuple[int, int, int]] = {
    STATE_LYING_IN_BED: (255, 105, 180),       # Hot pink
    STATE_SITTING_ON_BED: (0, 165, 255),       # Orange
    STATE_SITTING_OUTSIDE_BED: (255, 255, 0),   # Cyan
    STATE_STANDING: (0, 255, 0),               # Green
    STATE_WALKING: (0, 215, 255),              # Gold/Yellow
    STATE_OUT_OF_BED: (0, 69, 255),            # Red-Orange
    STATE_UNKNOWN: (128, 128, 128),            # Gray
}

