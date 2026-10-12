"""Optional Gemma 4 attention inspection. No model downloads at import time."""
import argparse
import json
from pathlib import Path
from .core import checkpoint_indices


def inspect(model_id, prompt, output, points=7):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype="auto", attn_implementation="eager"
    )
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        result = model(**inputs, output_attentions=True, use_cache=False, return_dict=True)
    if result.attentions is None:
        raise RuntimeError("Backend did not expose attentions; use compatible eager attention")
    indices = checkpoint_indices(len(result.attentions), points)
    rows = []
    for index in indices:
        att = result.attentions[index]
        if att is None:
            raise RuntimeError(f"Attention layer {index} not available")
        rows.append({"layer": index, "shape": list(att.shape), "mean": att.float().mean().item(),
                     "maximum": att.float().max().item()})
    payload = {"model": model_id, "points": points, "sequence_length": inputs["input_ids"].shape[-1],
               "checkpoints": rows, "warning": "Attention summaries are descriptive, not causal explanations"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument("--prompt", default="Which is larger: 3 or 7?")
    p.add_argument("--points", type=int, default=7)
    p.add_argument("--output", type=Path, default=Path("results/gemma-attention.json"))
    a = p.parse_args()
    inspect(a.model, a.prompt, a.output, a.points)


if __name__ == "__main__":
    main()
