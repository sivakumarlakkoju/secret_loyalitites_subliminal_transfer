import json
import os
import sys

from huggingface_hub import snapshot_download

MODELS = {
    "organism_1_5b": "Alamerton/12-mar-gen9-1.5b",
    "base_1_5b": "Qwen/Qwen2.5-1.5B-Instruct",
}

MODELS_DIR = "/workspace/subliminal-loyalty/models"


def download_all():
    paths = {}
    for name, repo_id in MODELS.items():
        local_dir = os.path.join(MODELS_DIR, name)
        print(f"Downloading {repo_id} -> {local_dir}", flush=True)
        path = snapshot_download(repo_id=repo_id, local_dir=local_dir)
        paths[name] = path
    return paths


def load_config(path):
    with open(os.path.join(path, "config.json")) as f:
        return json.load(f)


def verify_architecture_match(paths):
    organism_cfg = load_config(paths["organism_1_5b"])
    base_cfg = load_config(paths["base_1_5b"])

    keys_to_check = [
        "architectures",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "intermediate_size",
        "max_position_embeddings",
        "model_type",
    ]

    mismatches = []
    for key in keys_to_check:
        o_val = organism_cfg.get(key)
        b_val = base_cfg.get(key)
        status = "OK" if o_val == b_val else "MISMATCH"
        if o_val != b_val:
            mismatches.append(key)
        print(f"  {key:28s} organism={o_val!r:20} base={b_val!r:20} [{status}]")

    if mismatches:
        print(f"\nARCHITECTURE MISMATCH on: {mismatches}", file=sys.stderr)
        sys.exit(1)
    print("\nArchitecture match: OK (shared initialisation precondition holds)")


if __name__ == "__main__":
    paths = download_all()
    print("\n--- Verifying architecture match (organism vs base) ---")
    verify_architecture_match(paths)
    print("\nModel paths:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
