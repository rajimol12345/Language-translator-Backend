import os
import re
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TranslationEngine:
    def __init__(self, model: Optional[str] = None):
        # -----------------------------
        # Load API key
        # -----------------------------
        self.api_key = os.getenv("FEATHERLESS_API_KEY")
        if not self.api_key:
            raise RuntimeError("FEATHERLESS_API_KEY not found in environment variables")

        clean_key = self.api_key.strip().replace('"', "").replace("'", "")

        # -----------------------------
        # Model & Endpoint
        # -----------------------------
        self.model = model or "meta-llama/Meta-Llama-3-8B-Instruct"
        self.endpoint = "https://api.featherless.ai/v1/chat/completions"

        # -----------------------------
        # HTTP session with retries
        # -----------------------------
        self.session = requests.Session()
        retry_params = dict(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        try:
            retries = Retry(allowed_methods=frozenset({"POST"}), **retry_params)
        except TypeError:
            retries = Retry(method_whitelist=frozenset({"POST"}), **retry_params)

        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self.session.headers.update({
            "Authorization": f"Bearer {clean_key}",
            "Content-Type": "application/json",
        })

    # --------------------------------------------------
    # Clean API output
    # --------------------------------------------------
    def _clean_output(self, text: str) -> str:
        if not text:
            return ""

        # Remove backticks used for code markdown
        text = re.sub(r"```(?:\w+)?\n?", "", text)  # opening triple backticks
        text = text.replace("```", "")  # closing triple backticks
        text = text.strip("` \n")

        # Remove common LLM prefixes
        prefixes = [
            r"^here is the translation:?\s*",
            r"^translation:?\s*",
            r"^output:?\s*",
            r"^sure, here is the translation.*?:?\s*",
            r"^the translated text is:?\s*",
            r"^translated text:?\s*",
        ]
        for p in prefixes:
            text = re.sub(p, "", text, flags=re.IGNORECASE | re.MULTILINE)

        # Stop at explanatory notes
        stop_keywords = ("note:", "(note", "translator's note", "literally:", "explanation:")
        lines = []
        for line in text.splitlines():
            clean_line = line.strip().lower()
            if any(clean_line.startswith(kw) for kw in stop_keywords):
                break
            lines.append(line)

        cleaned = "\n".join(lines)
        return cleaned.strip().strip('"').strip("'")

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def translate(self, text: str, target_lang: str) -> str:
        if not text or not text.strip():
            return ""

        payload = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a professional translator.\n"
                        f"Target Language: {target_lang.capitalize()}\n"
                        "CRITICAL RULES:\n"
                        "1. Output ONLY the translated text.\n"
                        "2. Do NOT provide explanations or notes.\n"
                        "3. Preserve paragraph breaks and punctuation.\n"
                        "4. Maintain tone of the source."
                    )
                },
                {
                    "role": "user",
                    "content": f"Translate this text into {target_lang.capitalize()}:\n\n{text}"
                }
            ]
        }

        try:
            response = self.session.post(self.endpoint, json=payload, timeout=(10, 180))
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                raise RuntimeError(f"API Error: {data['error'].get('message', 'Unknown error')}")

            content = ""
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                content = choice.get("message", {}).get("content", "") or choice.get("text", "")

            if not content:
                logger.warning("API returned empty string")
                return ""

            return self._clean_output(content)

        except requests.exceptions.RequestException as e:
            logger.error(f"Network/API error: {e}")
            raise RuntimeError(f"Featherless API request failed: {e}")
        except Exception as e:
            logger.exception("Translation failed")
            raise RuntimeError(f"Unexpected error: {e}")


# -------------------- TEST --------------------
if __name__ == "__main__":
    try:
        engine = TranslationEngine()
        print("Hindi:", engine.translate("Hello world! How are you today?", "Hindi"))
        print("German:", engine.translate("The quick brown fox jumps over the lazy dog.", "German"))
    except Exception as e:
        print("Error:", e)
