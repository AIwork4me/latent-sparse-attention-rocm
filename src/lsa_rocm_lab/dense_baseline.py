"""Real full-attention ROCm PyTorch SDPA GPU latency baseline; no LSA acceleration claim."""
import argparse
import json
import statistics
import time
import torch
import torch.nn.functional as F


def sync(): torch.cuda.synchronize()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--max-context',type=int,default=32768)
    ap.add_argument('--reps',type=int,default=10)
    args=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('ROCm GPU unavailable')
    rows=[]
    for dtype in [torch.float16,torch.bfloat16]:
        for n in [512,1024,2048,4096,8192,16384,32768]:
            if n>args.max_context:continue
            try:
                torch.manual_seed(20261105+n)
                q=torch.randn((1,8,n,64),device='cuda',dtype=dtype)
                k=torch.randn_like(q);v=torch.randn_like(q)
                def fn():return F.scaled_dot_product_attention(q,k,v,is_causal=True)
                for _ in range(3):x=fn();sync()
                ts=[]
                for _ in range(args.reps):
                    sync();start=time.perf_counter();x=fn();sync()
                    ts.append(1000*(time.perf_counter()-start))
                row={'method':'torch_sdpa_dense','phase':'prefill','dtype':str(dtype),'context':n,'heads':8,'dim':64,'median_ms':statistics.median(ts),'p95_ms':sorted(ts)[min(len(ts)-1,int(len(ts)*0.95))],'repetitions':args.reps,'backend':'PyTorch SDPA automatic dispatch'}
                rows.append(row);print(json.dumps(row),flush=True)
                del q,k,v,x
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError as e:
                print(json.dumps({'method':'torch_sdpa_dense','dtype':str(dtype),'context':n,'status':'OOM','error':str(e)[:1000]}),flush=True)
                torch.cuda.empty_cache()
    if not rows:raise RuntimeError('No baseline samples obtained')

if __name__=='__main__':main()
