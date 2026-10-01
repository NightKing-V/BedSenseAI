"""
Logging configuration and custom formatting for BedSense AI.
Provides multi-tiered logging:
  - PROD / INFO (level 20): Production lifecycle — Starting, Calibration, Process steps, State transitions, Alerts, Exports, and Completion.
  - STATE (level 25): State-only transitions & clinical alerts.
  - AGENT (level 15): LangGraph reasoning loops, LLM structured prompts & tool traces.
  - DEBUG (level 10): Deep telemetry, per-frame keypoints & polygon overlap ratios.
"""

import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, Optional
import yaml

# ---------------------------------------------------------------------------
# Custom Log Level Numbers
# ---------------------------------------------------------------------------
AGENT_LEVEL_NUM = 15  # Agent reasoning & tool traces (Debug tier)
STATE_LEVEL_NUM = 25  # State transitions & clinical alerts (State-only tier)

if not hasattr(logging, "AGENT"):
    logging.addLevelName(AGENT_LEVEL_NUM, "AGENT")
if not hasattr(logging, "STATE"):
    logging.addLevelName(STATE_LEVEL_NUM, "STATE")


def log_agent(self, message, *args, **kws):
    """Log an agent reasoning / tool trace event."""
    if self.isEnabledFor(AGENT_LEVEL_NUM):
        self._log(AGENT_LEVEL_NUM, message, args, **kws)


def log_state(self, message, *args, **kws):
    """Log a clinical state transition / alert event."""
    if self.isEnabledFor(STATE_LEVEL_NUM):
        self._log(STATE_LEVEL_NUM, message, args, **kws)


def log_start(self, message, *args, **kws):
    """Log a pipeline startup lifecycle event."""
    if self.isEnabledFor(logging.INFO):
        if any(box_char in str(message) for box_char in ("╔", "╠", "╚", "║")):
            msg = message
        else:
            msg = message if str(message).startswith("[START]") else f"[START] {message}"
        self._log(logging.INFO, msg, args, **kws)


def log_calibrate(self, message, *args, **kws):
    """Log a calibration lifecycle event."""
    if self.isEnabledFor(logging.INFO):
        if any(box_char in str(message) for box_char in ("╔", "╠", "╚", "║")):
            msg = message
        else:
            msg = message if str(message).startswith("[CALIBRATE]") else f"[CALIBRATE] {message}"
        self._log(logging.INFO, msg, args, **kws)


def log_process(self, message, *args, **kws):
    """Log a processing step / progress lifecycle event."""
    if self.isEnabledFor(logging.INFO):
        if any(box_char in str(message) for box_char in ("╔", "╠", "╚", "║")):
            msg = message
        else:
            msg = message if str(message).startswith("[PROCESS]") else f"[PROCESS] {message}"
        self._log(logging.INFO, msg, args, **kws)


def log_export(self, message, *args, **kws):
    """Log an artifact export lifecycle event."""
    if self.isEnabledFor(logging.INFO):
        if any(box_char in str(message) for box_char in ("╔", "╠", "╚", "║")):
            msg = message
        else:
            msg = message if str(message).startswith("[EXPORT]") else f"[EXPORT] {message}"
        self._log(logging.INFO, msg, args, **kws)


def log_completed(self, message, *args, **kws):
    """Log a pipeline completion lifecycle event."""
    if self.isEnabledFor(logging.INFO):
        if any(box_char in str(message) for box_char in ("╔", "╠", "╚", "║")):
            msg = message
        else:
            msg = message if str(message).startswith("[COMPLETED]") else f"[COMPLETED] {message}"
        self._log(logging.INFO, msg, args, **kws)


# Attach helper methods to standard Logger class
logging.Logger.agent = log_agent
logging.Logger.state = log_state
logging.Logger.start = log_start
logging.Logger.calibrate = log_calibrate
logging.Logger.process = log_process
logging.Logger.export = log_export
logging.Logger.completed = log_completed


class BedSenseFormatter(logging.Formatter):
    """Custom formatter distinguishing Lifecycle (Start, Calibrate, Process, State, Export, Completed), Agent Reasoning, Operational Info, and Debug logs."""

    # ANSI colors for terminal
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    TAG_COLORS = {
        "START": ("\033[1;94m", "[START]"),          # Bold Blue
        "CALIBRATE": ("\033[1;96m", "[CALIBRATE]"),  # Bold Cyan
        "PROCESS": ("\033[1;96m", "[PROCESS]"),      # Bold Cyan
        "PROGRESS": ("\033[96m", "[PROGRESS]"),      # Cyan
        "STATE/EVENT": ("\033[1;92m", "[STATE/EVENT]"), # Bold Green
        "ALERT": ("\033[1;91m", "[ALERT]"),          # Bold Red
        "AGENT": ("\033[1;95m", "[AGENT]"),          # Bold Magenta
        "EXPORT": ("\033[1;95m", "[EXPORT]"),        # Bold Magenta
        "COMPLETED": ("\033[1;92m", "[COMPLETED]"),  # Bold Green
        "SUCCESS": ("\033[1;92m", "[SUCCESS]"),      # Bold Green
        "INFO": ("\033[96m", "[INFO]"),              # Cyan
        "WARNING": ("\033[1;93m", "[WARNING]"),      # Bold Yellow
        "ERROR": ("\033[1;91m", "[ERROR]"),          # Bold Red
        "DEBUG": ("\033[2m", "[DEBUG]"),            # Dim
    }

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()

        # If message contains raw multi-line box banners (e.g. ╔═══, ║, ╚═══), return without adding prefix
        if any(box_char in msg for box_char in ("╔", "╠", "╚", "║")):
            return msg

        # Check for explicit bracketed tags at the start of the message
        tag_match = re.match(r"^\[([A-Za-z0-9_/]+)\]\s*(.*)$", msg, re.DOTALL)
        if tag_match:
            raw_tag = tag_match.group(1).upper()
            body = tag_match.group(2)
            if raw_tag in self.TAG_COLORS:
                color_prefix, tag_str = self.TAG_COLORS[raw_tag]
                return f"{color_prefix}{tag_str}{self.RESET} {body}"

        # Otherwise format by log level
        if record.levelno == STATE_LEVEL_NUM:
            return f"{self.BOLD}{self.GREEN}[STATE/EVENT]{self.RESET} {msg}"
        elif record.levelno == AGENT_LEVEL_NUM:
            return f"{self.BOLD}{self.MAGENTA}[AGENT]{self.RESET} {msg}"
        elif record.levelno >= logging.ERROR:
            return f"{self.BOLD}{self.RED}[ERROR]{self.RESET} {msg}"
        elif record.levelno >= logging.WARNING:
            return f"{self.BOLD}{self.YELLOW}[WARNING]{self.RESET} {msg}"
        elif record.levelno == logging.INFO:
            return f"{self.CYAN}[INFO]{self.RESET} {msg}"
        elif record.levelno <= logging.DEBUG:
            return f"{self.DIM}[DEBUG]{self.RESET} {msg}"

        return super().format(record)


def load_config(config_path: Optional[str] = "configs/configurations.yaml") -> Dict[str, Any]:
    """Loads configuration dictionary from YAML config file."""
    if not config_path:
        return {}
    p = Path(config_path)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def setup_logger(log_level_str: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
    """Configure system-wide loggers with the specified level and BedSenseFormatter."""
    if not log_level_str:
        if config and "logging" in config and isinstance(config["logging"], dict):
            log_level_str = config["logging"].get("default_level", "PROD")
        else:
            log_level_str = "PROD"

    level_map = {
        "DEBUG": logging.DEBUG,
        "AGENT": AGENT_LEVEL_NUM,
        "INFO": logging.INFO,
        "PROD": logging.INFO,
        "PRODUCTION": logging.INFO,
        "STATE": STATE_LEVEL_NUM,
        "STATE_ONLY": STATE_LEVEL_NUM,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    level = level_map.get(str(log_level_str).upper(), logging.INFO)

    all_loggers = [
        "BedSense",
        "BedSenseVision",
        "BedSenseStreaming",
        "BedSenseState",
        "BedSenseEvidence",
        "BedSensePolicy",
        "BedSenseAgent",
        "BedSenseExporter",
        "BedSenseTemporal",
    ]

    for name in all_loggers:
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = False
        lg.setLevel(level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(BedSenseFormatter())
        lg.addHandler(handler)


# Pre-initialize root BedSense logger
logger = logging.getLogger("BedSense")
