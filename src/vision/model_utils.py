"""
Model Weight Resolution & Directory Management for BedSense AI.
Ensures YOLO and YOLO-Pose models are loaded from and downloaded into 'models/' or '/model'.
"""

import os
from pathlib import Path
from typing import Union
import logging

logger = logging.getLogger("BedSenseVision")


def resolve_model_path(model_path: Union[str, Path], default_dir: str = "models") -> str:
    """
    Resolves model weights path, checking:
    1. Direct path existence
    2. models/<name>
    3. model/<name>
    4. /model/<name>
    5. /workspace/models/<name>
    6. /workspace/model/<name>

    If the weights file does not exist locally yet, ensures `models/` directory
    exists and returns the `models/<name>` path so that Ultralytics YOLO downloads
    directly into the `models/` folder.
    """
    if not isinstance(model_path, (str, Path)):
        return model_path

    p = Path(model_path)

    # 1. Direct path exists
    if p.exists() and p.is_file():
        return str(p)

    filename = p.name

    # 2. Candidate paths in search priority
    candidates = [
        Path("models") / filename,
        Path("model") / filename,
        Path("/models") / filename,
        Path("/model") / filename,
        Path("/workspace/models") / filename,
        Path("/workspace/model") / filename,
    ]

    for cand in candidates:
        try:
            if cand.exists() and cand.is_file():
                return str(cand)
        except Exception:
            pass

    # 3. If not found, ensure default directory exists so downloads land in default_dir
    target_dir = Path(default_dir)
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    target_path = target_dir / filename
    return str(target_path)
