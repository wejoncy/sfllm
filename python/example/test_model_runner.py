from sfllm.engine.forward_params import ForwardBatch
from sfllm.model_loader.model_loader import TorchDefaultDtype, _load_check_point
from sfllm.models.modeling_qwen3 import Qwen3ForCausalLM
from sfllm.engine.model_runner import ModelRunner

if __name__ == "__main__":
    import transformers

    model_path = r"D:\\work\\Qwen3-0.6B"

    config = transformers.AutoConfig.from_pretrained(model_path)
    forward_batch = ForwardBatch(config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    with TorchDefaultDtype(config.dtype):
        model = Qwen3ForCausalLM(config).cuda()
        _load_check_point(model, model_path)
    model.eval()
    model_runner = ModelRunner(model)
    model_runner.capture_graph()
