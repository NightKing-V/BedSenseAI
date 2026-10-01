"""
Logging configuration and custom formatting for BedSense AI.
Provides multi-tiered logging:
  - PROD (level 25): Real-time state transitions, clinical alerts & final reports.
  - INFO (level 20): Operational video ingestion progress & status.
  - AGENT (level 15): LangGraph reasoning loops, LLM structured prompts & tool traces.
  - DEBUG (level 10): Deep telemetry, per-frame keypoints & polygon overlap ratios.
"""

import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional
import yaml

# ---------------------------------------------------------------------------
# Custom Log Level Numbers
# ---------------------------------------------------------------------------
AGENT_LEVEL_NUM = 15  # Agent reasoning & tool traces (Debug tier)
STATE_LEVEL_NUM = 25  # State transitions & clinical alerts (Production tier)

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


# Attach helper methods to standard Logger class
logging.Logger.agent = log_agent
logging.Logger.state = log_state


class BedSenseFormatter(logging.Formatter):
    """Custom formatter distinguishing State Transitions, Agent Reasoning, Operational Info, and Debug logs."""

    # ANSI colors for terminal
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == STATE_LEVEL_NUM:
            # High-visibility state transition / event formatting
            return f"{self.BOLD}{self.GREEN}[STATE/EVENT]{self.RESET} {record.getMessage()}"
        elif record.levelno == AGENT_LEVEL_NUM:
            # High-visibility agent reasoning / tool trace formatting
            return f"{self.BOLD}{self.MAGENTA}[AGENT]{self.RESET} {record.getMessage()}"
        elif record.levelno >= logging.WARNING:
            return f"{self.BOLD}{self.YELLOW}[WARNING]{self.RESET} {record.getMessage()}"
        elif record.levelno == logging.INFO:
            return f"{self.CYAN}[INFO]{self.RESET} {record.getMessage()}"
        elif record.levelno <= logging.DEBUG:
            return f"{self.DIM}[DEBUG]{self.RESET} {record.getMessage()}"
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
        "PROD": STATE_LEVEL_NUM,
        "PRODUCTION": STATE_LEVEL_NUM,
        "STATE": STATE_LEVEL_NUM,
        "STATE_ONLY": STATE_LEVEL_NUM,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    level = level_map.get(str(log_level_str).upper(), STATE_LEVEL_NUM)

    for name in [
        "BedSense",
        "BedSenseAgent",
        "BedSenseStreaming",
        "BedSenseState",
        "BedSensePolicy",
        "BedSenseVision",
        "BedSenseTemporal",
    ]:
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = False
        lg.setLevel(level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(BedSenseFormatter())
        lg.addHandler(handler)


# Pre-initialize root logger
logger = logging.getLogger("BedSense")
