"""One simple Triton HIP kernel to verify JIT toolchain. NOT an LSA kernel."""
import json
import torch
import triton
import triton.language as tl

@triton.jit
def vector_add(A, B, OUT, N:tl.constexpr, BLOCK:tl.constexpr):
    idx=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    a=tl.load(A+idx,idx<N,other=0)
    b=tl.load(B+idx,idx<N,other=0)
    tl.store(OUT+idx,a+b,idx<N)

def main():
    if not torch.cuda.is_available():raise RuntimeError('ROCm device unavailable')
    a=torch.arange(65536,device='cuda',dtype=torch.float32)
    b=torch.ones_like(a); o=torch.empty_like(a)
    vector_add[(triton.cdiv(a.numel(),256),)](a,b,o,a.numel(),BLOCK=256)
    torch.cuda.synchronize()
    err=(o-(a+b)).abs().max().item()
    print(json.dumps({'test':'Triton HIP JIT smoke (NOT LSA)','max_abs_error':err,'triton_version':triton.__version__}))
    if err!=0:raise AssertionError(f'error {err}')

if __name__=='__main__':main()
