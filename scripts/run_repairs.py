"""Two predeclared local variants, serial GPU use, durable per-run logs."""
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
for name in ('repair1_jpeg80','repair2_reserve10'):
    print('starting',name,flush=True)
    with (ROOT/f'results/{name}_run.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-u','-m','scripts.run_mvp','--config',f'configs/{name}.json'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('finished',name,flush=True)
