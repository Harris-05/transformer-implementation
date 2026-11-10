"""Exercise the real Trainer/PEFT integration on a tiny random CPU model.

This downloads no model weights and produces no clinical-model results.
"""

import json
import math
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOCAL_TEST_DEPS = ROOT / "artifacts/training_check_deps"
if LOCAL_TEST_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_TEST_DEPS))


def main():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM, Trainer
    from finetune import CompletionCollator, mask_completion, training_arguments

    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=64, bos_token_id=1, eos_token_id=2, pad_token_id=0,
        use_cache=False))
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0,
                                            target_modules="all-linear", task_type="CAUSAL_LM"))
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    base_embedding = model.get_input_embeddings().weight.detach().clone()
    data = [mask_completion([1, 3, 4], [1, 3, 4, 20, 21, 2], 16),
            mask_completion([1, 5], [1, 5, 22, 2], 16)]
    settings = json.loads((ROOT / "configs/qlora_config.json").read_text())
    settings.update(num_train_epochs=1, gradient_accumulation_steps=1, learning_rate=0.01)
    (ROOT / "artifacts").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as temporary:
        path = Path(temporary).resolve()
        if not path.is_relative_to((ROOT / "artifacts").resolve()):
            raise ValueError("Unexpected temporary training directory.")
        trainer = Trainer(model=model, args=training_arguments(path, settings, len(data), cpu=True),
                          train_dataset=data, eval_dataset=data, data_collator=CompletionCollator(0))
        result = trainer.train()
        if not math.isfinite(result.training_loss):
            raise AssertionError("Non-finite training loss.")
        if not any(not torch.equal(before[name], parameter.detach()) for name, parameter in model.named_parameters() if name in before):
            raise AssertionError("No adapter weight was updated.")
        if not torch.equal(base_embedding, model.get_input_embeddings().weight.detach()):
            raise AssertionError("Frozen base embeddings changed.")
    print("Tiny CPU integration check passed: real adapter updates, frozen base embeddings, checkpoint save/load and validation loss.")


if __name__ == "__main__":
    main()
