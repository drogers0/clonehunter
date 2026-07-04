"""Re-export microsoft/codebert-base to ONNX with a FIXED sequence length (256)
and dynamic batch, using the TorchScript exporter (simpler graph than dynamo).

CoreML's MIL compiler fails (-14) on the dynamic-seq dynamo export. A fixed-seq
graph compiles to a static-shape ML Program. Mean-pooling with the attention
mask makes the pooled output identical regardless of pad length, so this does
not change embeddings vs the dynamic CPU export.
"""

import os

import torch
from transformers import AutoModel

REV = "3b0952feddeffad0063f274080e3c23d75e7eb39"
SEQ = 256

base = AutoModel.from_pretrained("microsoft/codebert-base", revision=REV)
base.eval()


class Wrapper(torch.nn.Module):
    """Explicit-kwarg forward returning only last_hidden_state (avoids the
    transformers positional-arg collision and drops pooler_output)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.m(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state


model = Wrapper(base)
model.eval()

bs = 2
ids = torch.ones(bs, SEQ, dtype=torch.long)
mask = torch.ones(bs, SEQ, dtype=torch.long)
tti = torch.zeros(bs, SEQ, dtype=torch.long)

outdir = os.path.expanduser("~/.cache/clonehunter/onnx/codebert-base-static")
os.makedirs(outdir, exist_ok=True)
out = os.path.join(outdir, "model.onnx")

with torch.no_grad():
    torch.onnx.export(
        model,
        (ids, mask, tti),
        out,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "token_type_ids": {0: "batch"},
            "last_hidden_state": {0: "batch"},
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )

print("exported:", out, os.path.getsize(out) // (1024 * 1024), "MB")
