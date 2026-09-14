import asyncio
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import httpx
import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from sfllm.engine.sequence import AbortSequence, SequenceStatus
from sfllm.model_loader.model_config import ModelConfig, get_pool_index_layers
from sfllm.model_loader.model_loader import get_model_architecture
from sfllm.serving.app import create_app
from sfllm.serving.engine_server import EngineServer
from sfllm.serving.req_protocol import ChatRequest, GenerateReqInput
from sfllm.serving.tokenizer_manager import TokenizerManager


FIXTURE = Path(__file__).parent / "fixtures" / "qwen3_8"


@pytest.fixture
def manager():
    # Lossless byte encoding exercises the official template without downloading
    # the model's vocabulary or weights. Model inference is checked separately.
    vocab = {char: i for i, char in enumerate(pre_tokenizers.ByteLevel.alphabet())}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        chat_template=(FIXTURE / "chat_template.jinja").read_text(),
    )
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.tokenizer = tokenizer
    return manager


def submit_chat(**kwargs):
    request = ChatRequest(
        model="Qwen/Qwen3.8-27B",
        **{"messages": [{"role": "user", "content": "Hello"}], **kwargs},
    )
    server = EngineServer.__new__(EngineServer)
    server.running = True
    server.control_results = {}
    server.req_to_state = {}
    server.tokenizer_input_queue = Queue()
    asyncio.run(server.submit_request(GenerateReqInput.from_basemodel(request)))
    return server.tokenizer_input_queue.get_nowait()


def render(manager, **kwargs):
    sequence = submit_chat(**kwargs)
    return manager.tokenizer.decode(manager.encode_chat(sequence))


def test_official_config_uses_declared_architecture():
    config = ModelConfig(str(FIXTURE))
    model_class, architecture = get_model_architecture(config.hf_config)
    assert architecture == "Qwen3_5ForConditionalGeneration"
    assert model_class.__module__ == "sfllm.models.qwen3_5"
    assert config.hf_config.architectures == ["Qwen3_5ForConditionalGeneration"]
    assert config.dtype == torch.bfloat16
    text = config.hf_config.get_text_config()
    assert text.output_gate_type == "swish"
    assert get_pool_index_layers(text) == list(range(3, 64, 4))


@pytest.mark.parametrize("gate", [None, "silu", "swish"])
def test_gate_type_does_not_override_architecture(gate):
    config = ModelConfig(str(FIXTURE)).hf_config
    text = config.get_text_config()
    if gate is None:
        del text.output_gate_type
    else:
        text.output_gate_type = gate
    model_class, architecture = get_model_architecture(config)
    assert architecture == "Qwen3_5ForConditionalGeneration"
    assert model_class.__module__ == "sfllm.models.qwen3_5"


def test_explicit_qwen38_architecture_is_registered():
    config = ModelConfig(str(FIXTURE)).hf_config
    config.architectures = ["Qwen3_8ForConditionalGeneration"]
    model_class, architecture = get_model_architecture(config)
    assert architecture == "Qwen3_8ForConditionalGeneration"
    assert model_class.__module__ == "sfllm.models.qwen3_8"


def test_swish_does_not_redirect_other_architectures():
    config = SimpleNamespace(
        architectures=["Qwen3ForCausalLM"], output_gate_type="swish",
    )
    model_class, architecture = get_model_architecture(config)
    assert architecture == "Qwen3ForCausalLM"
    assert model_class.__module__ == "sfllm.models.qwen3"


@pytest.mark.parametrize("effort", [None, "xhigh", "medium", "low"])
def test_reasoning_effort_reaches_official_template(manager, effort):
    prompt = render(manager, reasoning_effort=effort)
    assert prompt.endswith("<|im_start|>assistant\n<think>\n")
    if effort in (None, "xhigh"):
        assert "Reasoning effort is set to xhigh." in prompt
    elif effort == "low":
        assert "Keep your thinking brief and focused" in prompt
    else:
        assert "Reasoning effort" not in prompt


def test_top_level_reasoning_effort_takes_precedence(manager):
    prompt = render(
        manager, reasoning_effort="low",
        chat_template_kwargs={"reasoning_effort": "xhigh"},
    )
    assert "Reasoning effort is set to low." in prompt
    assert "Reasoning effort is set to xhigh." not in prompt


def test_non_thinking_and_engine_tokenization_options(manager):
    prompt = render(manager, chat_template_kwargs={
        "enable_thinking": False, "tokenize": False,
        "return_dict": True, "return_tensors": "pt", "add_generation_prompt": False,
    })
    assert "Reasoning effort" not in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
@pytest.mark.parametrize("preserve", [None, True, False])
def test_historical_reasoning_is_preserved(manager, field, preserve):
    messages = [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4", field: "Adding two pairs gives four."},
        {"role": "user", "content": "And 3 + 3?"},
    ]
    kwargs = {} if preserve is None else {"preserve_thinking": preserve}
    prompt = render(manager, messages=messages, chat_template_kwargs=kwargs)
    assert ("Adding two pairs gives four." in prompt) is (preserve is not False)
    assert "4<|im_end|>" in prompt


def test_template_error_does_not_stop_tokenizer_worker(manager):
    invalid = submit_chat(reasoning_effort="invalid")
    valid = submit_chat(chat_template_kwargs={"enable_thinking": False})
    requests = iter((invalid, valid))

    def next_request():
        try:
            return next(requests)
        except StopIteration:
            manager.running = False
            return AbortSequence(-1)

    manager.load_tokenizer = lambda: None
    manager.eos_token_ids = frozenset({248044, 248046})
    manager.decode_states = {}
    manager.tokenizer_input_queue = SimpleNamespace(get=next_request)
    manager.tokenizer_output_queue = Queue()
    manager.inferengine_input_queue = Queue()
    TokenizerManager.tokenizer_event_run_loop(manager)

    error = manager.tokenizer_output_queue.get_nowait()[invalid.sequence_id]
    assert error["status"] == SequenceStatus.FAILED
    assert "Unexpected reasoning effort" in error["error"]
    assert manager.inferengine_input_queue.get_nowait() is valid
    assert valid.sampling_params.stop_token_ids == {248044, 248046}
    assert valid.tokens == manager.encode_chat(valid)


def test_template_error_returns_http_400():
    class Worker:
        async def submit_request(self, request):
            return 1

        async def get_response(self, request_id):
            yield {"error": "Unexpected reasoning effort invalid"}

    async def check():
        app = create_app(SimpleNamespace())
        app.state.inference_worker = Worker()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/v1/chat/completions", json={
                "model": "Qwen/Qwen3.8-27B",
                "messages": [{"role": "user", "content": "Hello"}],
                "reasoning_effort": "invalid",
            })
        assert response.status_code == 400
        assert response.json()["detail"] == "Unexpected reasoning effort invalid"

    asyncio.run(check())
