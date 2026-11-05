"""Verify exact-select upstream PyTorch reference vs full-causal SDPA (not a speed benchmark)."""
import importlib.util
import json
import os
import pathlib
import sys
import torch
import torch.nn.functional as F


def load_reference():
    root = pathlib.Path(os.getenv('LSA_UPSTREAM', 'third_party/latent-sparse-attention')).resolve()
    if not (root / 'src/lsa/core/attention.py').is_file():
        raise FileNotFoundError('Missing upstream checkout; run scripts/checkout_upstream.sh')
    sys.path.insert(0, str(root / 'src'))
    from lsa.core.attention import topk_attention
    return topk_attention


def main():
    if not torch.cuda.is_available(): raise RuntimeError('GPU unavailable')
    reference = load_reference()
    rows = []
    for dtype,atol,rtol in [(torch.float32, 1e-4, 1e-4),(torch.float16, 5e-3, 5e-3),(torch.bfloat16, 2e-2, 2e-2)]:
        for t in [16, 64, 128, 256]:
            torch.manual_seed(20261105 + t)
            q = torch.randn(1,t,4,32,device='cuda',dtype=dtype)*0.2
            k = torch.randn(1,t,4,32,device='cuda',dtype=dtype)*0.2
            v = torch.randn(1,t,4,32,device='cuda',dtype=dtype)*0.2
            ids = torch.arange(t,device='cuda',dtype=torch.int32).view(1,1,t).expand(1,t,t).contiguous()
            actual = reference(q,k,v,ids,window=0,sink=0,scale=32**-0.5)
            expected = F.scaled_dot_product_attention(q.permute(0,2,1,3),k.permute(0,2,1,3),v.permute(0,2,1,3),is_causal=True).permute(0,2,1,3)
            err = float((actual.float()-expected.float()).abs().max().item())
            ok = bool(torch.allclose(actual.float(),expected.float(),atol=atol,rtol=rtol))
            rows.append({'dtype':str(dtype),'tokens':t,'max_abs_error':err,'allclose':ok})
            if not ok: print(json.dumps({'results':rows},indent=2)); raise AssertionError(f'Full-select parity failed dtype={dtype} t={t}')
    print(json.dumps({'test':'full-select upstream reference vs ROCm dense causal SDPA','results':rows,'status':'PASS'},indent=2))

if __name__ == '__main__': main()
