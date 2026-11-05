#!/usr/bin/env python3
"""Track upstream HEAD and releases; optionally open a deduplicated GitHub issue.

Default is read-only. With --issue runs `gh issue create` using GH_TOKEN/GH_REPO.
The reviewed baseline is deliberately moved only after human/ROCm review.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
REPO='Prefrontal-AI/latent-sparse-attention'
HEAD_URL=f'https://api.github.com/repos/{REPO}/commits/main'
REL_URL=f'https://api.github.com/repos/{REPO}/releases?per_page=10'

def api(url):
    headers={'Accept':'application/vnd.github+json','User-Agent':'lsa-rocm-upstream-tracker'}
    if os.getenv('GH_TOKEN'):headers['Authorization']='Bearer '+os.environ['GH_TOKEN']
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=20) as rsp:
        return json.load(rsp)

def run(argv):
    return subprocess.run(argv,text=True,capture_output=True,check=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--issue',action='store_true');args=p.parse_args()
    baseline=json.loads((ROOT/'tracking/UPSTREAM_BASELINE.json').read_text())
    commit=api(HEAD_URL);releases=api(REL_URL)
    sha=commit['sha'];reviewed=baseline['reviewed_head_sha']
    info={
        'checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'upstream_sha':sha,'reviewed_sha':reviewed,'changed':sha!=reviewed,
        'subject':commit['commit']['message'].split('\n')[0],
        'commit_url':commit['html_url'],
        'releases':[{'tag_name':x['tag_name'],'published_at':x.get('published_at'),'url':x['html_url']} for x in releases[:10]],
    }
    out=ROOT/'results'/'upstream-latest.json';out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2))
    if not args.issue or not info['changed']:return
    repo=os.getenv('GH_REPO','')
    if not repo or not os.getenv('GH_TOKEN'):
        raise RuntimeError('--issue requires GH_REPO and GH_TOKEN')
    short=sha[:12]; title=f'[Upstream LSA] Review {short} for ROCm gfx1151'
    # Detect an existing issue for this upstream SHA to avoid daily duplicates
    old=run(['gh','issue','list','-R',repo,'--state','all','--search',short+' in:title','--json','title','--limit','100'])
    if old.returncode!=0:raise RuntimeError('gh issue list failed: '+old.stderr)
    if any(short in x.get('title','') for x in json.loads(old.stdout)):
        print('Issue already exists; no duplicate created');return
    compare=f'https://github.com/{REPO}/compare/{reviewed}...{sha}'
    body=(f'Automated upstream change notice (not an automated GPU compatibility claim).\n\n'
          f'- Last reviewed SHA: `{reviewed}`\n- New HEAD: `{sha}`\n'
          f'- Changes: {compare}\n- HEAD: {commit["html_url"]}\n\n'
          'Review code/algorithm changes, evaluate risks to Triton/HIP, add regression tests, '
          'and only update `tracking/UPSTREAM_BASELINE.json` after completing ROCm review and producing evidence. '
          'Never auto-merge upstream changes.\n')
    created=run(['gh','issue','create','-R',repo,'--title',title,'--body',body])
    if created.returncode!=0:raise RuntimeError('gh issue create failed: '+created.stderr)
    print('Opened issue: '+created.stdout.strip())

if __name__=='__main__':main()
