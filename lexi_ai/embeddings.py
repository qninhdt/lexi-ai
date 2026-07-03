"""Local sense-embedding via ``transformers`` (optional ``[embeddings]`` extra).

Why local, not the API: the configured chat proxy serves no embeddings model
(verified 403), so vectors are produced offline by a small sentence-transformer
(default ``all-MiniLM-L6-v2``, 384-dim, CPU-fast). No network, no key, no cost.

Design mirrors :class:`lexi_ai.generation.generator.Generator`:
  * the model is **injectable** — tests pass a fake ``encode`` and never import
    torch;
  * the real model is **lazy-loaded** on first use, so importing this module (and
    constructing a :class:`Lexicon`) is cheap even without the extra installed;
  * ``transformers`` is CPU-bound and blocking, so :meth:`embed` offloads to a
    worker thread via :func:`asyncio.to_thread`, like the sqlite Cambridge reader.

Vectors are mean-pooled over tokens (attention-masked) and L2-normalized, so
cosine similarity reduces to a dot product downstream. Embeddings are
**best-effort** everywhere they are used: if the extra is missing or the model
fails to load, callers skip embedding rather than fail generation.
"""

import asyncio
import threading
from collections.abc import Callable

from lexi_ai.config import Settings, get_settings

# An injected encoder: batch of texts -> list of vectors (already normalized).
# Takes a concrete list (``embed`` always materializes one before calling).
EncodeFn = Callable[[list[str]], list[list[float]]]


class EmbeddingUnavailable(RuntimeError):
    """Raised when the ``[embeddings]`` extra (torch/transformers) is not installed."""


class Embedder:
    """Turn text into normalized vectors with a local transformers model.

    Construct with an injected ``encode`` (tests) or let it lazy-build the real
    model from settings (production). ``model_name``/``dim`` describe whatever the
    encoder produces so persisted rows can be tagged and filtered by model.
    """

    def __init__(
        self,
        encode: EncodeFn | None = None,
        settings: Settings | None = None,
        model_name: str | None = None,
        dim: int | None = None,
    ):
        self._settings = settings or get_settings()
        self._encode = encode
        self._model_name = model_name or self._settings.embedding_model
        self._dim = dim
        # Guards the one-time lazy build of the real encoder. The build runs in a
        # worker thread (it downloads + loads ~200MB), so the lock is a threading
        # lock, not an asyncio one — mirrors CambridgeSource's index locks.
        self._build_lock = threading.Lock()

    @property
    def model_name(self) -> str:
        """Identifier stored on each embedded row (for model-drift filtering)."""
        return self._model_name

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts -> normalized float vectors.

        Offloads BOTH the one-time model build and the per-call encode to a worker
        thread, so the event loop is never blocked by the ~200MB model load (not
        just the encode). Empty input short-circuits. Raises
        :class:`EmbeddingUnavailable` if the extra is missing; other failures
        (model load, OOM, device) propagate and are handled best-effort by the
        api layer, which must never fail generation on an embedding error.
        """
        if not texts:
            return []
        return await asyncio.to_thread(self._encode_sync, list(texts))

    async def embed_one(self, text: str) -> list[float]:
        """Embed a single string (convenience for a query vector)."""
        vecs = await self.embed([text])
        return vecs[0]

    # --- lazy real-model construction (runs in the worker thread) ---------

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        """Resolve the encoder (building it once, lazily) then encode.

        Called only via :func:`asyncio.to_thread`, so the expensive one-time
        build happens off the event loop. Double-checked under a threading lock so
        two concurrent first-embeds don't both load the model.
        """
        encode = self._encode
        if encode is None:
            with self._build_lock:
                if self._encode is None:
                    self._encode = self._build_transformers_encoder()
                encode = self._encode
        return encode(texts)

    def _build_transformers_encoder(self) -> EncodeFn:
        """Build a mean-pool + L2-normalize encoder over a HF transformer.

        Imported lazily so the dependency is only required when a real embed
        actually happens. Weights download to the HF cache on first call.
        """
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import AutoModel, AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as exc:  # extra not installed
            raise EmbeddingUnavailable(
                "Sense embeddings need the '[embeddings]' extra. Install with: "
                "uv sync --extra embeddings"
            ) from exc

        settings = self._settings
        device = settings.embedding_device
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        model = AutoModel.from_pretrained(self._model_name)
        model.to(device)
        model.eval()
        self._dim = int(model.config.hidden_size)
        max_length = settings.embedding_max_length

        def encode(texts: list[str]) -> list[list[float]]:
            out: list[list[float]] = []
            batch = settings.embedding_batch_size
            for start in range(0, len(texts), batch):
                chunk = texts[start : start + batch]
                enc = tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    hidden = model(**enc).last_hidden_state  # (B, T, H)
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # (B, T, 1)
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / counts  # mean pool over real tokens
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                out.extend(pooled.cpu().tolist())
            return out

        return encode
