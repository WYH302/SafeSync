"""Compare native grayscale IDs with the mirror's native colored labels."""
import io
import json
import struct
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.fetch_second import RangeFile, URL
from safesync.data import PALETTE

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'data/second_pilot'


def main():
    manifest=json.loads((folder/'manifest.json').read_text())
    tail=(folder/'zip_tail.bin').read_bytes()
    fields=struct.unpack_from('<4s4H2IH',tail,tail.rfind(b'PK\x05\x06'))
    start=fields[6]
    remote=RangeFile(URL,manifest['archive_bytes'])
    remote.cache=[(start,(folder/'zip_directory.bin').read_bytes())]
    result=[]
    mapping={}
    with zipfile.ZipFile(remote) as archive:
        audit=json.loads((ROOT/'results/data_audit/label_audit.json').read_text())
        chosen=set(audit['sample_rows'])
        present=set()
        for pair in manifest['pairs']:
            labels=set(np.unique(np.array(Image.open(ROOT/pair['files']['label_after']['path']))).tolist())
            if labels-present:
                chosen.add(pair['id'])
                present|=labels
        for pair in manifest['pairs']:
            if pair['id'] not in chosen:
                continue
            member=pair['files']['label_after']['archive_member'].replace('/GT_T2/','/GT_T2_COLORED/')
            path=folder/'source_colors'/Path(member).name
            if not path.exists():
                info=archive.getinfo(member)
                block=remote.get(info.header_offset,info.header_offset+30+len(member.encode())+len(info.extra)+info.compress_size+100)
                remote.cache.append((info.header_offset,block))
                content=archive.read(info)
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(content)
                remote.cache=remote.cache[:1]
            colors=np.array(Image.open(path).convert('RGB'))
            grey=np.array(Image.open(ROOT/pair['files']['label_after']['path']))
            agreement=float(np.mean(np.all(colors==PALETTE[grey],axis=-1)))
            for label in np.unique(grey):
                values=np.unique(colors[grey==label],axis=0).tolist()
                mapping.setdefault(int(label),[])
                for value in values:
                    if value not in mapping[int(label)]:
                        mapping[int(label)].append(value)
            result.append({'pair_id':pair['id'],'pixel_palette_agreement':agreement})
            print(pair['id'],agreement,flush=True)
    out={'samples':result,'observed_id_to_source_rgb':mapping,'all_palette_pixels_agree':all(x['pixel_palette_agreement']==1. for x in result)}
    (ROOT/'results/data_audit/source_color_check.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    if not out['all_palette_pixels_agree']:
        raise RuntimeError('palette mismatch: stop class interpretation')


if __name__=='__main__':
    main()
