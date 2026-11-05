"""Overnight isolated stages; explicit PASS/FAIL/BLOCKED/TIMEOUT and immutable records."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
import signal
from pathlib import Path
import shlex
import subprocess
import sys
import time

from lsa_rocm_lab.gate_manifest import describe, verdict_of

ROOT=Path(__file__).resolve().parents[2]

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def stage_command(name, quick):
    py=sys.executable
    if name=='preflight':return [py,'-m','lsa_rocm_lab.preflight']
    if name=='reference_correctness':return [py,'-m','lsa_rocm_lab.correctness']
    if name=='triton_hip_probe':return [py,'-m','lsa_rocm_lab.triton_probe']
    if name=='dense_prefill_baseline':return [py,'-m','lsa_rocm_lab.dense_baseline','--max-context',str(2048 if quick else 32768),'--reps',str(3 if quick else 10)]
    if name=='lsa_rocm_kernel':return shlex.split(os.environ.get('LSA_ROCM_KERNEL_CMD','')) or None
    if name=='lsa_rocm_end_to_end':return shlex.split(os.environ.get('LSA_ROCM_E2E_CMD','')) or None
    raise ValueError(name)

def validate_receipt(receipt, stage, upstream_sha):
    """Require real machine-readable evidence from an externally implemented adapter.

    Fail-closed since v0.2.1: a PASS kernel receipt must carry exactly the
    required correctness gates (``gate_manifest``), each PASS - a skipped or
    empty suite (status UNVERIFIED) is rejected here, whatever the adapter exited.
    """
    if not receipt.is_file():
        raise ValueError('Adapter returned zero but wrote no validation receipt')
    data=json.loads(receipt.read_text())
    if data.get('kind')!=stage:
        raise ValueError('Receipt kind does not match the stage')
    if data.get('status')=='UNVERIFIED':
        raise ValueError('UNVERIFIED receipt (correctness gates skipped): not claimable')
    if data.get('status')!='PASS' or data.get('correctness_pass') is not True:
        raise ValueError('Receipt lacks required PASS/correctness gates')
    if upstream_sha=='UNAVAILABLE' or data.get('upstream_sha')!=upstream_sha:
        raise ValueError('Missing/mismatched upstream source SHA')
    if not str(data.get('gpu_arch','')).startswith('gfx1151'):
        raise ValueError('Adapter result not tied to gfx1151')
    if stage=='lsa_rocm_kernel':
        verdict=verdict_of(data.get('gates') or [])
        if not verdict['ok']:
            raise ValueError('Gate manifest mismatch ('+describe(verdict)+')')
        cases=data.get('comparisons',[])
        if not cases:raise ValueError('Empty real kernel benchmark comparisons')
        for case in cases:
            if case.get('baseline_ms',0)<=0 or case.get('candidate_ms',0)<=0:
                raise ValueError('Nonpositive synchronized timing')
            if not case.get('context') or not case.get('dtype') or case.get('samples',0)<10:
                raise ValueError('Insufficient matched context/dtype/repetition evidence')
    else:
        if data.get('sglang_armed') is not True or not data.get('model_checkpoint_sha') or not data.get('metrics'):
            raise ValueError('End-to-end adapter requires armed server, exact model and metrics')


def repo_head_sha() -> str:
    """This repository's HEAD SHA (provenance of the code that produced a run)."""
    p=subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True,text=True)
    return p.stdout.strip() if p.returncode==0 else 'UNAVAILABLE'


MANDATORY_STAGES=('preflight','reference_correctness','triton_hip_probe','dense_prefill_baseline','lsa_rocm_kernel')


def campaign_exit_code(statuses) -> int:
    """Fail-closed campaign verdict - mandatory stages must not fail.

    Every always-on stage that FAILs or TIMEOUTs fails the campaign; the kernel
    stage additionally fails it whenever an adapter RAN and did not PASS (the
    v0.2.0 exit code ignored kernel FAIL entirely). lsa_rocm_end_to_end is
    optional evidence: BLOCKED without an adapter is a legitimate outcome, and a
    failing adapter surfaces in its own receipt - it never silently greens this.
    """
    by={s['stage']:s for s in statuses}
    for name in MANDATORY_STAGES:
        if name not in by:
            return 1
        s=by[name]
        if name!='lsa_rocm_kernel':
            if s['status']!='PASS':
                return 1
        else:
            if s.get('command') is not None and s['status'] in ('FAIL','TIMEOUT'):
                return 1
    return 0


def main():
    ap=argparse.ArgumentParser()
    m=ap.add_mutually_exclusive_group(required=True)
    m.add_argument('--quick',action='store_true');m.add_argument('--overnight',action='store_true')
    args=ap.parse_args()
    timestamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder=ROOT/'results'/timestamp;folder.mkdir(parents=True,exist_ok=False)
    upstream=ROOT/'third_party'/'latent-sparse-attention'
    sha='UNAVAILABLE'
    if (upstream/'.git').exists():
        p=subprocess.run(['git','-C',str(upstream),'rev-parse','HEAD'],capture_output=True,text=True)
        if p.returncode==0:sha=p.stdout.strip()
    head=repo_head_sha()
    meta={'run_id':timestamp,'started_utc':timestamp,'profile':'quick' if args.quick else 'overnight',
          'repo_head_sha':head,'upstream_sha':sha,'hostname':os.uname().nodename,
          'command':sys.argv,'env':{'LSA_ROCM_KERNEL_CMD':os.getenv('LSA_ROCM_KERNEL_CMD',''),
                                    'LSA_ROCM_E2E_CMD':os.getenv('LSA_ROCM_E2E_CMD','')},
          'user':os.environ.get('USER','')}
    (folder/'run_meta.json').write_text(json.dumps(meta,indent=2))
    names=['preflight','reference_correctness','triton_hip_probe','dense_prefill_baseline','lsa_rocm_kernel','lsa_rocm_end_to_end']
    tmax={'preflight':120,'reference_correctness':1800,'triton_hip_probe':900,'dense_prefill_baseline':7200,'lsa_rocm_kernel':7200,'lsa_rocm_end_to_end':7200}
    (folder/'receipts').mkdir()
    statuses=[];preflight_ok=False;correctness_ok=False
    for name in names:
        cmd=stage_command(name,args.quick)
        if name=='lsa_rocm_kernel' and not correctness_ok:cmd=None
        if name=='lsa_rocm_end_to_end' and not (correctness_ok and len(statuses)>=5 and statuses[4]['status']=='PASS'):cmd=None
        status={'stage':name,'command':cmd,'started_utc':datetime.now(timezone.utc).isoformat()}
        if cmd is None or (name!='preflight' and not preflight_ok):
            status['status']='BLOCKED';status['reason']='No verified ROCm LSA adapter, prerequisite correctness, or GPU preflight'
        else:
            try:
                started=time.monotonic()
                receipt=folder/'receipts'/(name+'.json')
                env={**os.environ,'LSA_UPSTREAM':str(upstream),'LSA_STAGE_RECEIPT':str(receipt),'LSA_UPSTREAM_SHA':sha}
                with open(folder/(name+'.log'),'w') as log:
                    proc=subprocess.Popen(cmd,cwd=ROOT,text=True,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                    try:
                        proc.wait(timeout=tmax[name])
                        status['exit_code']=proc.returncode
                        status['status']='PASS' if proc.returncode==0 else ('BLOCKED' if proc.returncode==78 else 'FAIL')
                        if status['status']=='PASS' and name.startswith('lsa_rocm_'):
                            validate_receipt(receipt,name,sha)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                        status['status']='TIMEOUT'
                status['elapsed_s']=round(time.monotonic()-started,3)
            except Exception as e:
                status['status']='FAIL';status['error']=repr(e)
        if name=='preflight':preflight_ok=status['status']=='PASS'
        if name=='reference_correctness':correctness_ok=status['status']=='PASS'
        statuses.append(status)
        with open(folder/'stages.jsonl','a') as f:f.write(json.dumps(status,ensure_ascii=False)+'\n')
        print(f"[{name}] {status['status']}",flush=True)
    summary={'run_dir':str(folder),'run_id':timestamp,'profile':meta['profile'],
             'repo_head_sha':head,'upstream_sha':sha,
             'statuses':{s['stage']:s['status'] for s in statuses},
             'claimable_lsa_speedup':statuses[-2]['status']=='PASS' and correctness_ok}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2))
    (folder/'REPORT.md').write_text('# LSA Radeon Run\n\nRun ID: `'+timestamp+'` · repo HEAD: `'+head+'` · upstream SHA: `'+sha+'`\n\n|Stage|Status|\n|---|---|\n'+''.join(f"|{s['stage']}|{s['status']}|\n" for s in statuses)+'\nNo speedup is published by the starter harness. Real GPU LSA port and baseline-matched timing required.\n')
    with open(folder/'SHA256SUMS','w') as out:
        for f in sorted(folder.rglob('*')):
            if f.is_file() and f.name!='SHA256SUMS':out.write(digest(f)+'  '+str(f.relative_to(folder))+'\n')
    print(json.dumps(summary,indent=2))
    return campaign_exit_code(statuses)

if __name__=='__main__':raise SystemExit(main())
