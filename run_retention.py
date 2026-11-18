"""Score base and weight-edited Qwen3.5-0.8B on MMLU-Pro with thinking off.

For each question the model sees the question and options, then the fixed
assistant prefix "The answer is (". We read the probability of each option
letter as the next token. No sampling, no generation, so no looping and no
random noise.

    python run_retention.py                 # the 1,000 questions the MIT group used
    python run_retention.py --limit 50      # quick smoke test
    python run_retention.py --all           # all 12,032 test questions (slow)
    python run_retention.py --device cpu    # if the GPU gives errors

Outputs: results/<variant>.jsonl, one line per question. Safe to stop and rerun.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

import edits as E

MODEL = "Qwen/Qwen3.5-0.8B"
REV = "2fc06364715b967f1860aea9cf38778875588b17"
HERE = os.path.dirname(os.path.abspath(__file__))
LETTERS = "ABCDEFGHIJ"
PREFIX = "The answer is ("


def load_model():
    import transformers
    err = None
    for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel"):
        try:
            try:
                m = getattr(transformers, cls).from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16)
            except TypeError:
                m = getattr(transformers, cls).from_pretrained(MODEL, revision=REV, torch_dtype=torch.bfloat16)
            print("loaded with", cls)
            return m.eval()
        except Exception as e:  # try the next loader class
            err = e
    raise err


def find(params, suf):
    hits = [n for n in params if n.endswith(suf) and "vision" not in n and "visual" not in n]
    assert len(hits) == 1, (suf, hits)
    return hits[0]


def find_top(params, leaf):
    """Embedding table / final norm: match by the end of the name, outside the numbered layers."""
    hits = [n for n in params if (n == leaf or n.endswith("." + leaf))
            and "layers." not in n and "vision" not in n and "visual" not in n]
    assert len(hits) == 1, (leaf, hits, list(params)[:5], list(params)[-5:])
    return hits[0]


def to_bits(t):
    return t.detach().cpu().contiguous().view(torch.int16).numpy().view(np.uint16).copy()


def from_bits(bits):
    return torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16)


def prompt(tok, q):
    opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(q["options"]))
    user = (f"The following is a multiple choice question about {q['category']}.\n\n"
            f"Question: {q['question']}\n\nOptions:\n{opts}\n\n"
            f"Reply with the letter of the correct option.")
    msgs = [{"role": "user", "content": user}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return text + PREFIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    ap.add_argument("--only", default="", help="comma-separated variant names")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    model = load_model()
    params = dict(model.named_parameters())

    # ---- build every edit on the exact bfloat16 values, before any dtype change
    cache = {}

    def get_bits(suf):
        if suf not in cache:
            cache[suf] = to_bits(params[find(params, suf)])
        return cache[suf]

    print("parameter names look like:", list(params)[:2], "...", list(params)[-2:])
    embed = E.bits_to_f32(to_bits(params[find_top(params, "embed_tokens.weight")]))
    gamma = E.bits_to_f32(to_bits(params[find_top(params, "norm.weight")]))
    first = lambda w: tok.encode(w, add_special_tokens=False)[0]
    all_edits, report = E.build_all(get_bits, os.path.join(HERE, "mit_edits"),
                                    embed_f32=embed, gamma_f32=gamma, first_token=first)
    print("\n".join(report))
    with open(os.path.join(a.out, "edit_report.txt"), "w") as f:
        f.write("\n".join(report) + "\n")
    del embed

    # ---- float32 holds every bfloat16 value exactly, and runs on MPS and CPU
    model = model.to(torch.float32).to(a.device)
    params = dict(model.named_parameters())
    base_copy = {s: params[find(params, s)].detach().cpu().clone() for s in cache}

    def set_variant(name):
        with torch.no_grad():
            for s, w in base_copy.items():
                params[find(params, s)].copy_(w.to(a.device))
            if name != "base":
                for s, bits in all_edits[name].items():
                    params[find(params, s)].copy_(from_bits(bits).to(torch.float32).to(a.device))

    # ---- data
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    if not a.all:
        keep = set(json.load(open(os.path.join(HERE, "mmlupro_1k_ids.json"))))
        ds = ds.filter(lambda r: r["question_id"] in keep)
        assert len(ds) == len(keep), (len(ds), len(keep))
    if a.limit:
        ds = ds.select(range(a.limit))
    letter_ids = []
    for L in LETTERS:
        ids = tok.encode(L, add_special_tokens=False)
        assert len(ids) == 1, (L, ids)
        letter_ids.append(ids[0])
    print(f"{len(ds)} questions, device {a.device}")
    if a.limit:
        print("--- example prompt ---\n" + prompt(tok, ds[0]) + "\n----------------------")

    variants = ["base"] + list(all_edits)
    if a.only:
        variants = [v for v in variants if v in a.only.split(",")]
    for v in variants:
        path = os.path.join(a.out, v + ".jsonl")
        done = sum(1 for _ in open(path)) if os.path.exists(path) else 0
        if done >= len(ds):
            print(v, "already done")
            continue
        set_variant(v)
        t0 = time.time()
        with open(path, "a") as f, torch.no_grad():
            for i in range(done, len(ds)):
                q = ds[i]
                n = len(q["options"])
                ids = tok(prompt(tok, q), return_tensors="pt").input_ids.to(a.device)
                logits = model(input_ids=ids).logits[0, -1].float().cpu()
                lp_all = torch.log_softmax(logits, -1)
                lp = lp_all[letter_ids[:n]]
                letter_mass = float(lp.exp().sum())           # share of probability on valid letters
                lpn = torch.log_softmax(lp, -1)               # renormalized over the valid letters
                pred = int(lpn.argmax())
                gold = int(q["answer_index"])
                f.write(json.dumps({"id": int(q["question_id"]), "cat": q["category"], "n": n, "gold": gold,
                                    "pred": pred, "correct": int(pred == gold),
                                    "lp_gold": round(float(lpn[gold]), 5),
                                    "letter_mass": round(letter_mass, 5)}) + "\n")
                if (i + 1) % 25 == 0 or i + 1 == len(ds):
                    f.flush()
                    print(f"{v}: {i + 1}/{len(ds)}  {(time.time() - t0) / (i + 1 - done):.2f}s per question")
        assert not math.isnan(letter_mass)
    set_variant("base")
    print("done. Now run: python analyze.py")


if __name__ == "__main__":
    main()
