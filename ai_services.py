import os
import logging
from typing import Optional

logging.basicConfig(
	level=os.getenv("LOG_LEVEL", "INFO"),
	format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ai_services")

# Optional OpenAI SDK (new v1 client)
try:
	from openai import OpenAI  # type: ignore
except Exception:
	OpenAI = None  # type: ignore


class AIServices:
	"""AI abstraction that uses OpenAI if configured, otherwise safe fallbacks.
	No persistence; only env vars.
	"""

	def __init__(self) -> None:
		self.openai_api_key = os.getenv("OPENAI_API_KEY")
		self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
		self.image_model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
		self.client = None
		if self.openai_api_key and OpenAI is not None:
			try:
				self.client = OpenAI(api_key=self.openai_api_key)
				logger.info("OpenAI client initialized")
			except Exception as exc:  # pragma: no cover
				logger.warning("OpenAI init failed: %s", exc)

	def is_available(self) -> bool:
		return self.client is not None

	def generate_text(self, prompt: str, system: Optional[str] = None) -> str:
		if self.client is None:
			return f"[AI unavailable] {prompt}"
		try:
			msgs = []
			if system:
				msgs.append({"role": "system", "content": system})
			msgs.append({"role": "user", "content": prompt})
			resp = self.client.chat.completions.create(
				model=self.openai_model,
				messages=msgs,
				max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "512")),
				temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
			)
			return (resp.choices[0].message.content or "").strip()
		except Exception as exc:  # pragma: no cover
			logger.warning("OpenAI text generation failed: %s", exc)
			return f"[AI error] {prompt}"

	def summarize(self, text: str) -> str:
		if self.client is None:
			parts = [p.strip() for p in text.replace("\n", ". ").split(".") if p.strip()]
			return ". ".join(parts[:3])[:800]
		return self.generate_text(
			prompt=f"Summarize the following text concisely:\n\n{text}",
			system="You are a concise summarizer.",
		)

	def translate(self, text: str, target_lang: str) -> str:
		if self.client is None:
			return f"[translate->{target_lang}] {text}"
		return self.generate_text(
			prompt=f"Translate to {target_lang}:\n\n{text}",
			system="You are a helpful translator.",
		)

	def generate_image(self, prompt: str) -> str:
		"""Return an image URL or a fallback message."""
		if self.client is None:
			return f"[image unavailable] {prompt}"
		try:
			resp = self.client.images.generate(
				model=self.image_model,
				prompt=prompt,
				size=os.getenv("OPENAI_IMAGE_SIZE", "1024x1024"),
			)
			if resp.data and getattr(resp.data[0], "url", None):
				return resp.data[0].url
			return "[image generated, but URL unavailable]"
		except Exception as exc:  # pragma: no cover
			logger.warning("OpenAI image generation failed: %s", exc)
			return f"[image error] {prompt}"