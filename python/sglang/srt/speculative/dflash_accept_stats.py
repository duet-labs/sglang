"""Opt-in (SGLANG_DFLASH_ACCEPT_HIST=1) accept-position statistics for DFLASH chain/tree
verify: commit-length histogram, accepted candidate rank per depth, and draft-confidence
calibration (P(accept | conf bin, parent accepted)). Printed as one JSON line every
SGLANG_DFLASH_ACCEPT_HIST_EVERY steps on TP rank 0."""
import json
import os

import torch

NBINS = 20
MAXC = 32
_EVERY = int(os.environ.get("SGLANG_DFLASH_ACCEPT_HIST_EVERY", "100"))
_S: dict = {}


def _state(kind, N, device):
    st = _S.get(kind)
    if st is None:
        z = lambda *s: torch.zeros(s, dtype=torch.int64, device=device)
        st = dict(
            steps=0, rows=0, N=N,
            commit_hist=z(N + 2),
            depth_cand=z(N + 1, MAXC),
            cal_total=z(NBINS), cal_acc=z(NBINS),
            cal0_total=z(NBINS), cal0_acc=z(NBINS),
            cal_depth_total=z(N + 1, NBINS), cal_depth_acc=z(N + 1, NBINS),
            lm_total=z(NBINS), lm_acc=z(NBINS),
        )
        _S[kind] = st
    return st


def _emit(kind, st, rank0):
    st["steps"] += 1
    if not rank0 or st["steps"] % _EVERY:
        return
    out = {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in st.items()}
    out["kind"] = kind
    print("ACCEPT_HIST " + json.dumps(out), flush=True)


def record_chain(commit_lens, bs, rank0, block_size):
    st = _state("chain", int(block_size), commit_lens.device)
    cl = commit_lens[:bs].to(torch.int64).clamp(0, st["N"] + 1)
    st["commit_hist"] += torch.bincount(cl, minlength=st["N"] + 2)
    st["rows"] += int(bs)
    _emit("chain", st, rank0)


def record_tree(tree, accept_index, accept_token_num, bs, rank0):
    N = int(tree.num_nodes)
    dev = accept_index.device
    st = _state("tree", N, dev)
    atn = accept_token_num[:bs].to(torch.int64)
    st["commit_hist"] += torch.bincount((atn + 1).clamp(0, N + 1), minlength=N + 2)
    st["rows"] += int(bs)

    k = torch.arange(accept_index.shape[1], device=dev)
    valid = k[None, :] <= atn[:, None]
    idx = accept_index[:bs].to(torch.int64).clamp(min=0)
    acc = torch.zeros((bs, N), dtype=torch.bool, device=dev)
    acc.scatter_(1, idx, valid)
    acc[:, 0] = True

    parents = tree.parents[:bs].to(torch.int64)
    pidx = parents.clamp(min=0)
    depth = tree.depth[:bs].to(torch.int64)
    cand = tree.node_cand[:bs].to(torch.int64)
    lm = tree.log_mass[:bs].float()
    conf = torch.exp(lm - lm.gather(1, pidx)).clamp(0, 1)
    pacc = acc.gather(1, pidx)
    nonroot = parents >= 0
    live = nonroot & pacc  # the node was actually tested against the target

    d = depth.clamp(0, N)
    c = cand.clamp(0, MAXC - 1)
    sel = acc & nonroot
    st["depth_cand"].index_put_((d[sel], c[sel]), torch.ones_like(d[sel]), accumulate=True)

    b = (conf * NBINS).long().clamp(0, NBINS - 1)
    one = torch.ones_like(b)
    st["cal_total"].index_put_((b[live],), one[live], accumulate=True)
    st["cal_acc"].index_put_((b[live & acc],), one[live & acc], accumulate=True)
    l0 = live & (cand == 0)
    st["cal0_total"].index_put_((b[l0],), one[l0], accumulate=True)
    st["cal0_acc"].index_put_((b[l0 & acc],), one[l0 & acc], accumulate=True)
    st["cal_depth_total"].index_put_((d[live], b[live]), one[live], accumulate=True)
    st["cal_depth_acc"].index_put_((d[live & acc], b[live & acc]), one[live & acc], accumulate=True)
    bl = (torch.exp(lm).clamp(0, 1) * NBINS).long().clamp(0, NBINS - 1)
    st["lm_total"].index_put_((bl[nonroot],), one[nonroot], accumulate=True)
    st["lm_acc"].index_put_((bl[sel],), one[sel], accumulate=True)
    _emit("tree", st, rank0)
