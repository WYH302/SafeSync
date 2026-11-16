"""Copy compatible local cached distributions into this task's isolated env.

This fallback avoids network installs and never changes an existing user env.
Requirements are exported after import verification; normal fresh installs may
use those versions instead. Cache selection checks CPython 3.11 Windows tags.
"""
from pathlib import Path
import json
import os
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]
CACHE=Path(os.environ['LOCALAPPDATA'])/'uv/cache/archive-v0'
TARGET=ROOT/'.venv/Lib/site-packages'
WANTED={'numpy','pillow','requests','charset-normalizer','certifi','idna','urllib3','networkx','sympy','mpmath','jinja2','markupsafe','fsspec','filelock','typing-extensions','setuptools'}


def main():
    found={}
    for metadata in CACHE.glob('*/*.dist-info'):
        text=(metadata/'METADATA').read_text(encoding='utf-8')
        name=re.search(r'^Name: (.+)$',text,re.M).group(1).lower().replace('_','-')
        if name not in WANTED:
            continue
        wheel=(metadata/'WHEEL').read_text()
        if not ('-none-any' in wheel or 'cp311-cp311-win_amd64' in wheel or any(f'cp{x}-abi3-win_amd64' in wheel for x in ('37','38','39','310','311'))):
            continue
        version=re.search(r'^Version: (.+)$',text,re.M).group(1)
        # First compatible cache entry is sufficient; avoid costly online resolution.
        found.setdefault(name,(metadata.parent,version))
    missing=WANTED-set(found)
    if missing:
        raise RuntimeError(f'no compatible cache for {missing}')
    copied=[]
    for name,(source,version) in sorted(found.items()):
        shutil.copytree(source,TARGET,dirs_exist_ok=True)
        copied.append({'name':name,'version':version,'cache_source':str(source)})
        print(name,version,flush=True)
    (ROOT/'results/environment').mkdir(parents=True,exist_ok=True)
    (ROOT/'results/environment/cache_bootstrap.json').write_text(json.dumps(copied,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
