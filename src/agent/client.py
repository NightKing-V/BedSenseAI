"""
Ollama Client for local Text and VLM inference.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional
import httpx

AGENT_LEVEL_NUM = 15
if not hasattr(logging, "AGENT"):
    logging.addLevelName(AGENT_LEVEL_NUM, "AGENT")

if not hasattr(logging.Logger, "agent"):
    def log_agent(self, message, *args, **kws):
        if self.isEnabledFor(AGENT_LEVEL_NUM):
            self._log(AGENT_LEVEL_NUM, message, args, **kws)
    logging.Logger.agent = log_agent

logger = logging.getLogger("BedSenseAgent")


class OllamaClient:
    """HTTP client communicating with Ollama service for Text and VLM inference."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        text_model: str = "qwen2.5:3b",
        vlm_model: str = "qwen2.5vl:3b",
        timeout_sec: float = 30.0,
    ):
        if not base_url:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        self.base_url = base_url.rstrip("/")
        self.text_model = text_model
        self.vlm_model = vlm_model
        self.timeout_sec = timeout_sec

        # Availability status for LLM and VLM instances
        self.llm_available = False
        self.vlm_available = False

    def is_available(self, auto_pull: bool = True) -> bool:
        """Checks if the configured Ollama endpoint is reachable and models are ready, auto-pulling missing models if needed."""
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            if r.status_code == 200:
                tags_data = r.json()
                model_names = [m.get("name", "") for m in tags_data.get("models", [])]

                # Check LLM text model
                self.llm_available = any(self.text_model in name for name in model_names)
                if not self.llm_available:
                    if auto_pull:
                        logger.info(f"[START] Ollama LLM text model '{self.text_model}' not found in local cache. Automatically pulling from registry...")
                        if self.pull_model(self.text_model):
                            self.llm_available = True
                    else:
                        logger.error(f"[ERROR] Ollama LLM text model '{self.text_model}' not found in available models: {model_names}")

                # Check VLM vision model
                self.vlm_available = any(self.vlm_model in name for name in model_names)
                if not self.vlm_available:
                    if auto_pull:
                        logger.info(f"[START] Ollama VLM model '{self.vlm_model}' not found in local cache. Automatically pulling from registry...")
                        if self.pull_model(self.vlm_model):
                            self.vlm_available = True
                    else:
                        logger.error(f"[ERROR] Ollama VLM vision model '{self.vlm_model}' not found in available models: {model_names}")

                return True
            else:
                logger.error(f"[ERROR] Ollama endpoint {self.base_url} returned status code {r.status_code}")
                return False
        except Exception as e:
            logger.error(f"[ERROR] Failed to connect to Ollama service at {self.base_url}: {e}")
            return False

    def pull_model(self, model_name: str) -> bool:
        """Pulls a model from the Ollama registry via the HTTP API."""
        try:
            logger.info(f"[START] Pulling model '{model_name}' from Ollama API ({self.base_url})...")
            with httpx.Client(timeout=1800.0) as client:
                resp = client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model_name, "stream": False},
                )
                if resp.status_code == 200:
                    logger.info(f"[START] Model '{model_name}' successfully downloaded and loaded into Ollama.")
                    return True
                else:
                    logger.error(f"[ERROR] Failed to pull model '{model_name}': HTTP {resp.status_code} - {resp.text}")
                    return False
        except Exception as e:
            logger.error(f"[ERROR] Exception occurred while downloading model '{model_name}': {e}")
            return False

    def generate_json(self, prompt: str, system_prompt: str = "") -> Optional[Dict[str, Any]]:
        """Invokes Ollama text model with JSON schema constraint."""
        try:
            payload = {
                "model": self.text_model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.1,
                },
            }
            logger.agent(f"[LLM Query] Prompting text model '{self.text_model}' at {self.base_url}...")
            resp = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout_sec,
            )
            if resp.status_code == 200:
                body = resp.json()
                raw_response = body.get("response", "")
                parsed = json.loads(raw_response)
                logger.agent(f"[LLM Response] Received structured JSON: {parsed}")
                return parsed
            else:
                logger.error(f"[ERROR] Ollama LLM request to '{self.text_model}' returned HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"[ERROR] Ollama LLM generation failed: {e}")
        return None

    def query_vlm(
        self,
        prompt: str,
        image_base64_list: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Sends sampled frames to Qwen-VL model with constrained JSON prompt."""
        try:
            payload = {
                "model": self.vlm_model,
                "prompt": prompt,
                "images": image_base64_list,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.1,
                },
            }
            logger.agent(f"[VLM Query] Sending {len(image_base64_list)} frames to '{self.vlm_model}' at {self.base_url}...")
            resp = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout_sec,
            )
            if resp.status_code == 200:
                body = resp.json()
                raw_response = body.get("response", "")
                parsed = json.loads(raw_response)
                logger.agent(f"[VLM Response] Received visual inspection: {parsed}")
                return parsed
            else:
                logger.error(f"[ERROR] Ollama VLM request to '{self.vlm_model}' returned HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"[ERROR] Ollama VLM query failed: {e}")
        return None


__all__ = ["OllamaClient"]
