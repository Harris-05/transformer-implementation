# Copy this file into ONE new Colab cell after the completed benchmark cell.
# Rechecks one failing case; does not train or repeat the full timing benchmark.

import gc
import json
from pathlib import Path
import sys

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def check_attention_variants(model, inputs):
    from benchmark_kv_cache import compare_paths

    results = {}
    for name in ("automatic_sdpa", "math_sdpa"):
        if name == "math_sdpa":
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                result = compare_paths(model, inputs, budget=16, eos_token_id=None)
        else:
            result = compare_paths(model, inputs, budget=16, eos_token_id=None)
        results[name] = result
        print(name, "passed:", result["passed"], flush=True)
        print("First differences:", result["first_divergence"], flush=True)
        for key in ("cached", "uncached"):
            tokens = result[key]["token_ids"]
            print(key, "token 6:", tokens[5] if len(tokens) > 5 else None, flush=True)
        print("HF token 6:", result["hf_token_ids"][5], flush=True)
        if result["logit_diagnostic"] is not None:
            print(json.dumps(result["logit_diagnostic"], indent=2), flush=True)
    return results


def main(project, training, original_run):
    sys.path.insert(0, str(project))
    from baseline import save_json
    from benchmark_kv_cache import load_model, prepare_cases

    if not torch.cuda.is_available():
        raise RuntimeError("Use the same Colab GPU runtime as the benchmark.")
    settings = json.loads((project / "configs/kv_cache_config.json").read_text())
    config, selected, validation_hash, cases, _ = prepare_cases(project, training, settings)
    case = next(c for c in cases if c["case_id"] == "p256-o16")
    original = json.loads((original_run / "run_manifest.json").read_text())
    expected = original["identity"]
    if (expected["adapter_sha256"] != selected["adapter_sha256"]
            or expected["validation_sha256"] != validation_hash
            or expected["model"] != config):
        raise ValueError("Original run and current adapter/data/model differ.")
    torch.manual_seed(settings["seed"])
    torch.cuda.manual_seed_all(settings["seed"])
    model = load_model(config, training, "sdpa")
    try:
        inputs = torch.tensor([case["token_ids"]], dtype=torch.long, device="cuda:0")
        results = check_attention_variants(model, inputs)
        payload = {
            "experiment": "focused_attention_backend_diagnostic",
            "case_id": case["case_id"], "input_token_ids": case["token_ids"],
            "original_run_signature": original["run_signature"],
            "model": config, "adapter_sha256": selected["adapter_sha256"],
            "torch_version": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "precision": "same 4-bit NF4 weights and FP16 compute as original",
            "purpose": "Change only SDPA backend; neither result replaces the original benchmark.",
            "variants": results,
        }
        output = original_run / "numerical_diagnostic.json"
        save_json(output, payload)
        print("Saved diagnostic:", output, flush=True)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # PROJECT, TRAINING and RUN are defined by your previous benchmark cell.
    main(Path(PROJECT), Path(TRAINING), Path(RUN))
