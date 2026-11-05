import json
import os
import platform
import subprocess
import sys
import torch

def cmd(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=12, check=False)
        return {'returncode': p.returncode, 'stdout': p.stdout[-3000:], 'stderr': p.stderr[-500:]}
    except Exception as e:
        return {'error': repr(e)}

def main():
    data = {
        'python': sys.version,
        'platform': platform.platform(),
        'uname': platform.uname()._asdict(),
        'torch_version': torch.__version__,
        'torch_hip': torch.version.hip,
        'cuda_is_available': torch.cuda.is_available(),
        'rocm_smi': cmd(['rocm-smi', '--showproductname']),
        'rocminfo': cmd(['rocminfo']),
        'pip_freeze': cmd([sys.executable, '-m', 'pip', 'freeze']),
        'visible_devices': os.getenv('HIP_VISIBLE_DEVICES', ''),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        data['device_name'] = props.name
        data['gcnArchName'] = getattr(props, 'gcnArchName', '')
        data['total_memory_bytes'] = props.total_memory
        x = torch.randn(512, 512, device='cuda', dtype=torch.float16)
        torch.cuda.synchronize()
        y = x @ x
        torch.cuda.synchronize()
        data['gpu_matmul_finite'] = bool(torch.isfinite(y).all().item())
        del x, y
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError('ROCm GPU not accessible from PyTorch')
    if not torch.__version__.startswith('2.14.0+rocm10.1.0'):
        raise RuntimeError('Expected PyTorch 2.14.0+rocm10.1.0')
    if not data.get('gcnArchName', '').startswith('gfx1151'):
        raise RuntimeError('Expected gfx1151, got ' + repr(data.get('gcnArchName')))

if __name__ == '__main__': main()
