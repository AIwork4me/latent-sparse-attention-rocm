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

    Every always-on stage that is anything other than PASS fails the campaign.
    The kernel stage fails it whenever an adapter was **configured and
    attempted** and the outcome is anything other than a verified PASS -
    including adapter BLOCKED (exit 78), FAIL, TIMEOUT, a crash, a nonzero or
    missing exit, a malformed/missing receipt, or gates that did not verify
    (validate_receipt turns those into FAIL before this runs). A kernel adapter
    that was never configured is the only legitimate non-PASS outcome: BLOCKED
    with no command, never a speedup claim. The optional E2E stage follows the
    same rule when its adapter was explicitly configured and attempted (F11);
    an E2E adapter that was not requested stays a legitimate BLOCKED.
    """
    by={s['stage']:s for s in statuses}
    for name in MANDATORY_STAGES:
        if name not in by:
            return 1
        s=by[name]
        if name!='lsa_rocm_kernel':
            if s.get('status')!='PASS':
                return 1
        else:
            attempted=s.get('command') is not None
            if attempted and s.get('status')!='PASS':
                return 1
            if not attempted and s.get('status')!='BLOCKED':
                return 1  # a PASS record with no adapter run is fabricated evidence
    e2e=by.get('lsa_rocm_end_to_end')
    if e2e is not None and e2e.get('command') is not None and e2e.get('status')!='PASS':
        return 1
    return 0


def execute_stage(name,cmd,folder,upstream,sha,run_id,tmax,base_env=None):
    """Run one stage subprocess and classify it fail-closed (unit-testable).

    PASS requires exit 0 AND, for adapter stages, a receipt that survives
    validate_receipt. BLOCKED is only the adapter's explicit exit 78 or a stage
    that was not runnable (cmd None). FAIL covers nonzero exit, process crash,
    receipt-validation errors and pre-execution failures (e.g. the log file
    cannot be created). TIMEOUT reaps the whole process group.
    """
    status={'stage':name,'command':cmd,'started_utc':datetime.now(timezone.utc).isoformat()}
    if cmd is None:
        status['status']='BLOCKED'
        status['reason']='No verified ROCm LSA adapter, prerequisite correctness, or GPU preflight'
        return status
    try:
        started=time.monotonic()
        receipt=folder/'receipts'/(name+'.json')
        env={**(os.environ if base_env is None else base_env),'LSA_UPSTREAM':str(upstream),
             'LSA_STAGE_RECEIPT':str(receipt),'LSA_UPSTREAM_SHA':sha,'LSA_RUN_ID':run_id}
        with open(folder/(name+'.log'),'w') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,text=True,stdout=log,stderr=subprocess.STDOUT,
                                  env=env,start_new_session=True)
            try:
                proc.wait(timeout=tmax)
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
    return status


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
        kernel_pass=any(s['stage']=='lsa_rocm_kernel' and s['status']=='PASS' for s in statuses)
        if name=='lsa_rocm_end_to_end' and not (correctness_ok and kernel_pass):cmd=None
        if name!='preflight' and not preflight_ok:cmd=None
        status=execute_stage(name,cmd,folder,upstream,sha,timestamp,tmax[name])
        if name=='preflight':preflight_ok=status['status']=='PASS'
        if name=='reference_correctness':correctness_ok=status['status']=='PASS'
        statuses.append(status)
        with open(folder/'stages.jsonl','a') as f:f.write(json.dumps(status,ensure_ascii=False)+'\n')
        print(f"[{name}] {status['status']}",flush=True)
    kernel=next((s for s in statuses if s['stage']=='lsa_rocm_kernel'),None)
    summary={'run_dir':str(folder),'run_id':timestamp,'profile':meta['profile'],
             'repo_head_sha':head,'upstream_sha':sha,
             'statuses':{s['stage']:s['status'] for s in statuses},
             'claimable_lsa_speedup':bool(kernel) and kernel.get('status')=='PASS' and correctness_ok}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2))
    (folder/'REPORT.md').write_text('# LSA Radeon Run\n\nRun ID: `'+timestamp+'` · repo HEAD: `'+head+'` · upstream SHA: `'+sha+'`\n\n|Stage|Status|\n|---|---|\n'+''.join(f"|{s['stage']}|{s['status']}|\n" for s in statuses)+'\nNo speedup is published by the starter harness. Real GPU LSA port and baseline-matched timing required.\n')
    with open(folder/'SHA256SUMS','w') as out:
        for f in sorted(folder.rglob('*')):
            if f.is_file() and f.name!='SHA256SUMS':out.write(digest(f)+'  '+str(f.relative_to(folder))+'\n')
    print(json.dumps(summary,indent=2))
    return campaign_exit_code(statuses)

if __name__=='__main__':raise SystemExit(main())
