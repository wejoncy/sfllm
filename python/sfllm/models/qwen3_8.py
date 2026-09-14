"""Text-only inference entry for dense Qwen3.8 checkpoints.

Qwen3.8-27B retains the Qwen3.5 Hugging Face architecture name and uses the
same decoder and weight layout. Reuse that implementation through a distinct
entry class so future Qwen3.8-specific behavior has its own home.
The official checkpoint resolves to Qwen3_5ForConditionalGeneration as declared
in its config; configs declaring Qwen3_8ForConditionalGeneration use this entry.
"""

from sfllm.models.qwen3_5 import Qwen3_5ForConditionalGeneration


class Qwen3_8ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Dense Qwen3.8 text model, sharing Qwen3.5 execution and weight loading."""


EntryClass = Qwen3_8ForConditionalGeneration
