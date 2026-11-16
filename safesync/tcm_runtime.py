"""Isolated official TCM network with the retained official native rANS backend."""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys

from .entropy_bridge import install_local_bridge,_namespace


def initialize(root:Path):
    """Load exact dependency operators without unrelated package initializers."""
    dependency = root/"resources/modern_codec/.deps"
    sys.path.insert(0,str(dependency))
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    bridge,_ = install_local_bridge(root/"resources/learned_codec")
    ans = sys.modules["compressai.ans"]

    class BufferedEncoder:
        def __init__(self):
            self.symbols,self.indexes = [],[]
            self.tables = None
            self.table_key = None

        def encode_with_indexes(self,symbols,indexes,cdfs,lengths,offsets):
            key = bridge.table(cdfs,lengths,offsets)
            if self.table_key is not None and key != self.table_key:
                raise ValueError("This scoped adapter requires the fixed TCM Gaussian CDF")
            self.tables,self.table_key = (cdfs,lengths,offsets),key
            self.symbols.extend(symbols)
            self.indexes.extend(indexes)

        def flush(self):
            if self.tables is None:
                raise ValueError("Buffered encoder has no symbols")
            return bridge.encode(self.symbols,self.indexes,*self.tables)

    class PrefixDecoder:
        def __init__(self):
            self.stream = None
            self.indexes = []
            self.table_key = None

        def set_stream(self,stream):
            self.stream = bytes(stream)
            self.indexes = []
            self.table_key = None

        def decode_stream(self,indexes,cdfs,lengths,offsets):
            key = bridge.table(cdfs,lengths,offsets)
            if self.stream is None or self.table_key is not None and key != self.table_key:
                raise ValueError("TCM decoder stream or fixed Gaussian CDF differs")
            self.table_key = key
            previous = len(self.indexes)
            self.indexes.extend(indexes)
            # Native decode restarts from the same stream and replays its exact
            # index prefix. Arithmetic operations stay in the official backend.
            values = bridge.decode(self.stream,self.indexes,cdfs,lengths,offsets)
            return values[previous:]

        def decode_with_indexes(self,stream,indexes,cdfs,lengths,offsets):
            return bridge.decode(stream,indexes,cdfs,lengths,offsets)

    ans.BufferedRansEncoder,ans.RansDecoder = BufferedEncoder,PrefixDecoder
    models = sys.modules["compressai.models"]
    models.CompressionModel = importlib.import_module("compressai.models.base").CompressionModel
    _namespace("timm",dependency/"timm")
    _namespace("timm.models",dependency/"timm/models")
    layers = _namespace("timm.models.layers",dependency/"timm/models/layers")
    layers.DropPath = importlib.import_module("timm.models.layers.drop").DropPath
    layers.trunc_normal_ = importlib.import_module("timm.models.layers.weight_init").trunc_normal_
    path = root/"resources/modern_codec/tcm_source/models/tcm.py"
    spec = importlib.util.spec_from_file_location("safesync_official_tcm",path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return bridge,module.TCM


def load_model(root:Path,model_class,checkpoint:Path,channels:int):
    import torch
    from compressai.zoo.pretrained import load_pretrained
    stored = torch.load(checkpoint,map_location="cpu",weights_only=True)
    state = stored["state_dict"] if "state_dict" in stored else stored
    state = load_pretrained(state)
    model = model_class(config=[2,2,2,2,2,2],head_dim=[8,16,32,32,16,8],drop_path_rate=0.,N=channels,M=320)
    model.load_state_dict(state)
    model.eval().requires_grad_(False)
    model.update(force=True)
    return model.to("cuda")
