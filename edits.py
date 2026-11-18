"""Build the edited tensors. Pure numpy, no torch.

All tensors are handled as bfloat16 bit patterns (uint16), because the published
edits are defined on the stored bfloat16 values.

Two sources of edits:
  MIT    (Takehi et al., "Sparks of Weight-Space Programming"): p10c, p25c, p50c.
         Rebuilt from their released edit files and checked against their sha256.
  WSP-S  (Hu et al., arXiv 2611.09382): rebuilt from the formulas in their
         Appendix C. Checked against the "Changed BF16" counts they report.
Controls: a random direction of matched norm in the column the best math edit
touches (should do nothing), and five deliberately harmful edits (should lower
accuracy, to show the test can detect damage).
"""
import hashlib
import json
import os

import numpy as np

P = "layers.{}.{}.weight"  # suffix of a tensor name inside the language model


def bits_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def f32_to_bits(x):
    """float32 -> bfloat16 bits, round to nearest even."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)


def sha(bits):
    return hashlib.sha256(np.ascontiguousarray(bits).tobytes()).hexdigest()


def suffix(name):
    """'model.language_model.layers.15.mlp.down_proj.weight' -> 'layers.15.mlp.down_proj.weight'"""
    return name[name.index("layers."):]


# ---------------------------------------------------------------- MIT edits
def mit_edit(edit_dir, get_bits):
    """get_bits(suffix) -> uint16 array of the base tensor. Returns {suffix: edited bits}."""
    man = json.load(open(os.path.join(edit_dir, "manifest.json")))
    z = np.load(os.path.join(edit_dir, "edit.npz"))
    out = {}
    for i, t in enumerate(man["tensors"]):
        s = suffix(t["name"])
        base = get_bits(s)
        assert list(base.shape) == t["shape"], (s, base.shape)
        assert sha(base) == t["base_sha256"], f"{s}: base tensor differs from the one the edit was made on"
        if t["form"] == "lowrank":
            w = bits_to_f32(base).astype(np.float64)
            a, b = z[f"t{i}_a"], z[f"t{i}_b"]
            for k in range(a.shape[1]):
                w = w + np.outer(a[:, k], b[k])
            e = f32_to_bits(w.astype(np.float32)).reshape(-1)
        else:
            e = base.copy().reshape(-1)
        e[z[f"t{i}_idx"]] = z[f"t{i}_val"]
        e = e.reshape(base.shape)
        assert sha(e) == t["edited_sha256"], f"{s}: rebuilt tensor does not match the evaluated checkpoint"
        out[s] = e
    return out


# ---------------------------------------------------------- WSP-Scope edits
def _cols(get_bits, s, cols, fn):
    w = bits_to_f32(get_bits(s)).copy()
    w[:, cols] = fn(w[:, cols])
    return f32_to_bits(w)


def head(h):  # 256 output-projection columns of full-attention head h
    return slice(256 * h, 256 * (h + 1))


O = lambda l: P.format(l, "self_attn.o_proj")
W = lambda l: P.format(l, "mlp.down_proj")


def ws_gpqa(get_bits, **_):
    return {O(15): _cols(get_bits, O(15), head(6), lambda x: 0 * x)}


def ws_mmlupro(get_bits, **_):
    return {O(15): _cols(get_bits, O(15), head(7), lambda x: 0 * x)}


def ws_medxpertqa(get_bits, **_):
    C = {13: [7], 15: [0], 17: [86, 119], 18: [29, 32], 19: [11, 171], 20: [89], 22: [20, 26, 89],
         23: [0, 40, 161, 162, 190, 266, 317, 350, 495]}
    return {W(l): _cols(get_bits, W(l), c, lambda x: 2 * x) for l, c in C.items()}


def ws_scibench(get_bits, **_):
    out = {O(11): _cols(get_bits, O(11), head(1), lambda x: 0.5 * x),
           O(15): _cols(get_bits, O(15), head(7), lambda x: 0.5 * x),
           O(23): _cols(get_bits, O(23), head(5), lambda x: 0.5 * x),
           W(16): _cols(get_bits, W(16), [998], lambda x: -x)}
    return out


def ws_scicode(get_bits, **_):
    return {W(16): _cols(get_bits, W(16), [998], lambda x: -2 * x),
            O(23): _cols(get_bits, O(23), head(5), lambda x: 0 * x)}


def _doubt_direction(embed_f32, gamma_f32, first_token):
    A = ["Wait", "However", "But", "Maybe", "Perhaps"]
    B = ["Therefore", "Thus", "So", "Now", "The", "First", "Next", "Let"]
    ea = np.mean([embed_f32[first_token(w)] for w in A], axis=0)
    eb = np.mean([embed_f32[first_token(w)] for w in B], axis=0)
    u = (ea - eb) * (1.0 + gamma_f32)
    return (u / np.linalg.norm(u)).astype(np.float32)


def ws_math_primary(get_bits, embed_f32=None, gamma_f32=None, first_token=None, **_):
    """W23[:, 507] -= u, u = doubt-minus-progress readout direction (their Eq. 3)."""
    u = _doubt_direction(embed_f32, gamma_f32, first_token)
    w = bits_to_f32(get_bits(W(23))).copy()
    w[:, 507] = w[:, 507] - u
    return {W(23): f32_to_bits(w)}


def ctrl_random(get_bits, embed_f32=None, gamma_f32=None, first_token=None, seed=0, **_):
    """Control: same column, same norm (1), random direction orthogonal to u."""
    u = _doubt_direction(embed_f32, gamma_f32, first_token).astype(np.float64)
    r = np.random.default_rng(seed).standard_normal(u.shape[0])
    r = r - (r @ u) * u
    r = (r / np.linalg.norm(r)).astype(np.float32)
    w = bits_to_f32(get_bits(W(23))).copy()
    w[:, 507] = w[:, 507] - r
    return {W(23): f32_to_bits(w)}


# ------------------------------------------------- positive (damage) controls
# Added 18 November 2026, after the main run and before their results were seen.
# Purpose: show that the test detects damage when damage exists.
def _p10c_tensors(mit_dir):
    man = json.load(open(os.path.join(mit_dir, "p10c", "manifest.json")))
    return [suffix(t["name"]) for t in man["tensors"]]


def damage_noise(level, seed=0):
    """Multiplicative Gaussian noise on the same 30 matrices the broadest published edit touches."""
    def fn(get_bits, mit_dir=None, **_):
        rng = np.random.default_rng(seed)
        out = {}
        for s in _p10c_tensors(mit_dir):
            w = bits_to_f32(get_bits(s))
            out[s] = f32_to_bits(w * (1.0 + level * rng.standard_normal(w.shape, dtype=np.float32)))
        return out
    return fn


def damage_zero_attn15(get_bits, **_):
    """All eight heads of layer 15 set to zero (the published edits zero one head)."""
    return {O(15): f32_to_bits(0 * bits_to_f32(get_bits(O(15))))}


DAMAGE_CONTROLS = {
    "ctrl_damage_noise05": damage_noise(0.05),
    "ctrl_damage_noise10": damage_noise(0.10),
    "ctrl_damage_noise20": damage_noise(0.20),
    "ctrl_damage_noise40": damage_noise(0.40),
    "ctrl_damage_zero_attn15": damage_zero_attn15,
    # second noise seed, to see how much the noise controls vary by seed
    "ctrl_damage_noise10_s1": damage_noise(0.10, seed=1),
    "ctrl_damage_noise20_s1": damage_noise(0.20, seed=1),
}


def rel_change(edit, get_bits):
    """Size of an edit: norm of the change divided by norm of the touched tensors."""
    num = den = 0.0
    for s, e in edit.items():
        w = bits_to_f32(get_bits(s)).astype(np.float64)
        num += float(((bits_to_f32(e) - w) ** 2).sum())
        den += float((w ** 2).sum())
    return (num / den) ** 0.5


# name -> (builder, changed-BF16 count reported by the authors, or None)
SPEC_EDITS = {
    "ws_math_primary": (ws_math_primary, 1023),
    "ws_gpqa": (ws_gpqa, 262144),
    "ws_mmlupro": (ws_mmlupro, 262144),
    "ws_medxpertqa": (ws_medxpertqa, 21504),
    "ws_scibench": (ws_scibench, 787456),
    "ws_scicode": (ws_scicode, 263168),
    "ctrl_random": (ctrl_random, None),
}
MIT_EDITS = ["mit_p10c", "mit_p25c", "mit_p50c"]


def build_all(get_bits, mit_dir, **ctx):
    """Returns ({edit name: {suffix: edited bits}}, [report lines])."""
    edits, report = {}, []
    for name in MIT_EDITS:
        edits[name] = mit_edit(os.path.join(mit_dir, name[4:]), get_bits)
        n = sum(int((e != get_bits(s)).sum()) for s, e in edits[name].items())
        report.append(f"{name}: {len(edits[name])} tensors, {n:,} values changed, sha256 OK")
    for name, (fn, expect) in SPEC_EDITS.items():
        edits[name] = fn(get_bits, **ctx)
        n = sum(int((e != get_bits(s)).sum()) for s, e in edits[name].items())
        ok = "" if expect is None else (" (matches paper)" if n == expect else f" (PAPER SAYS {expect:,}: CHECK)")
        report.append(f"{name}: {len(edits[name])} tensors, {n:,} values changed{ok}")
    for name, fn in DAMAGE_CONTROLS.items():
        edits[name] = fn(get_bits, mit_dir=mit_dir, **ctx)
        report.append(f"{name}: {len(edits[name])} tensors (damage control)")
    report.append("relative size of each edit (norm of change / norm of touched tensors):")
    for name, ed in edits.items():
        report.append(f"  {name}: {rel_change(ed, get_bits):.4f}")
    a, b = edits["ws_mmlupro"], edits["mit_p25c"]
    same = a.keys() == b.keys() and all((bits_to_f32(a[k]) == bits_to_f32(b[k])).all() for k in a)  # -0.0 == +0.0
    report.append("ws_mmlupro has the same values as mit_p25c: " + ("yes (two groups found the same edit)" if same else "no"))
    return edits, report
