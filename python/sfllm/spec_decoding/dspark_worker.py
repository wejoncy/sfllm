"""DSpark proposals on the shared block-draft verification and KV protocol."""

from sfllm.models.dspark import DSparkConfig
from sfllm.spec_decoding.dflash2_worker import DFlash2Worker


class DSparkWorker(DFlash2Worker):
    config_cls = DSparkConfig
    checkpoint_verify_extra = 1

    def __init__(self, server_args):
        top_k = server_args.speculative_dspark_topk
        if top_k != -1 and top_k <= 0:
            raise ValueError("DSpark top-k must be -1 (original DSpark) or a positive integer.")
        super().__init__(server_args)
        model = self.draft_model_runner.model
        if top_k > 0 and model.dflash_config.markov_head_type != "vanilla":
            raise ValueError("DSpark top-k proposals require a vanilla Markov head.")
        if top_k > int(model.config.vocab_size):
            raise ValueError("DSpark top-k exceeds the vocabulary size.")
        model.proposal_top_k = top_k
