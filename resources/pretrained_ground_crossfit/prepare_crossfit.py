"""Metadata/public-checkpoint preparation only: no images, forward passes or GPU.

Model selection is the author's deterministic AOI fold, never task scores.
All outputs stay in the new crossfit resources directory.
"""
from __future__ import annotations
import argparse
import ast
import concurrent.futures
import hashlib
import json
import random
import re
import shutil
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import requests

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]
ORIGINAL = ROOT / "resources/pretrained_ground"
SOURCE = ORIGINAL / "official_source"
COMMIT = "ff1241f01e391248a9d6111a60c97973fc5a9643"
BUCKET = "https://spacenet-dataset.s3.amazonaws.com/"
PREFIX = "spacenet-model-weights/spacenet-7/4-motokimura/weights/"
MULTIPART = 8 * 1024 * 1024
SOURCE_NAMES = [
    "4-motokimura/code/scripts/train/train_cnns.sh", "4-motokimura/code/scripts/train/preprocess.sh",
    "4-motokimura/code/tools/split_dataset.py", "4-motokimura/code/tools/train_spacenet7_model.py",
    "4-motokimura/code/configs/unet_timm-efficientnet-b3_scale-3.0_v_02.yml",
    "4-motokimura/code/configs/unet_timm-efficientnet-b3_scale-4.0_v_01.yml",
    "4-motokimura/code/spacenet7_model/configs/defaults.py",
    "4-motokimura/code/spacenet7_model/models/__init__.py",
    "4-motokimura/code/spacenet7_model/datasets/__init__.py",
    "4-motokimura/weights/README.txt", "LICENSE",
]


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def catalog():
    identity = json.loads((ORIGINAL / "source_identity.json").read_text("utf-8"))
    assert identity["commit"] == COMMIT
    expected = {entry["path"]: entry["sha256"] for entry in identity["source_files"]}
    files = {}
    for name in SOURCE_NAMES:
        assert sha(SOURCE / name) == expected[name], f"Cached pinned source changed: {name}"
        files[name] = expected[name]
        target = FOLDER / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / name, target)
    train = (SOURCE / SOURCE_NAMES[0]).read_text("utf-8")
    configs = dict(re.findall(r"(?m)^(CONFIG_\w+)='([^']+)'", train))
    result = []
    for block in re.split(r"(?m)^##\s+", train)[1:]:
        exp = re.search(r"EXP_ID\s+(\d+)", block)
        fold = re.search(r"INPUT.TRAIN_VAL_SPLIT_ID\s+(\d+)", block)
        config = re.search(r"--config\s+\$\{([^}]+)\}", block)
        if not (exp and fold and config):
            continue
        config_name = "4-motokimura/code/" + PurePosixPath(configs[config[1]]).relative_to("/work").as_posix()
        text = (SOURCE / config_name).read_text("utf-8")
        losses = re.search(r"LOSSES:\s*(\[[^\]]+\])", text)
        result.append(dict(experiment=f"exp_{int(exp[1]):04d}", fold=int(fold[1]), config=config_name,
            architecture=re.search(r"ARCHITECTURE:\s*(\w+)", text)[1],
            backbone=re.search(r"BACKBONE:\s*'([^']+)'", text)[1],
            input_scale=float(re.search(r"SIZE_SCALE:\s*([\d.]+)", text)[1]),
            losses=ast.literal_eval(losses[1]) if losses else ["dice", "bce"],
            input_channels=3, output_channels=3, activation="sigmoid",
            decoder_channels=[256,128,64,32,16], decoder_attention=None))
    assert len(result) == 10
    return result, files


def freeze_fold_selection(models, sources):
    listing_path = ROOT / "data/spacenet7_pilot/source/listing_train_aois.json"
    listing = json.loads(listing_path.read_text("utf-8"))
    ids = sorted(prefix.rstrip("/").rsplit("/", 1)[-1] for prefix in listing["prefixes"])
    assert len(ids) == len(set(ids)) == 60
    shuffled = list(ids)
    random.Random(777).shuffle(shuffled)
    folds = {aoi: index % 5 for index, aoi in enumerate(shuffled)}
    development_path = ROOT / "data/spacenet7_pilot/development_manifest_v1.json"
    confirmation_path = ROOT / "data/spacenet7_pilot/confirmation_manifest_v1.json"
    development = json.loads(development_path.read_text("utf-8"))
    confirmation = json.loads(confirmation_path.read_text("utf-8"))
    targets = []
    for role, aois in (("calibration", development["aois"][:2]), ("confirmation", confirmation["aois"])):
        for index, area in enumerate(aois):
            fold = folds[area["aoi_id"]]
            targets.append(dict(role=role, role_index=index, aoi_id=area["aoi_id"],
                author_validation_fold=fold, selected_experiment=f"exp_{200+fold:04d}",
                excluded_from_author_gradient_training=True,
                used_for_author_validation_and_best_epoch_selection=True, observations=len(area["months"])))
    required = sorted({target["selected_experiment"] for target in targets})
    selected = [entry for entry in models if entry["experiment"] in required]
    assert len(selected) == len(required)
    assert all(entry["input_scale"] == 3 and entry["architecture"] == "unet"
               and entry["backbone"] == "timm-efficientnet-b3" and entry["losses"] == ["dice", "focal"] for entry in selected)
    payload = dict(schema="sn7-author-fold-crossfit-preparation-v1", official_commit=COMMIT,
        selection_rule="sort 60 official train directory IDs; Python Random(777).shuffle; validation fold k is shuffled[k::5]; model exp_020k excludes this fold from gradient training",
        task_results_used_for_selection=False, source_files_sha256=sources,
        train_directory_listing_sha256=sha(listing_path), development_manifest_sha256=sha(development_path),
        confirmation_manifest_sha256=sha(confirmation_path), sorted_AOI_ids=ids, shuffled_AOI_ids=shuffled,
        author_folds={str(fold):shuffled[fold::5] for fold in range(5)}, targets=targets,
        required_experiments=required, selected_model_configs=selected,
        checkpoint_selection_caveat="Public model_best was selected by IoU on its validation fold. This is gradient-training exclusion, not an untouched independent test checkpoint; original training logs/lists are not embedded in the checkpoint.")
    path = FOLDER / "fold_selection.json"
    if path.exists():
        assert json.loads(path.read_text("utf-8")) == payload, "Refusing changed fold/model selection"
    else:
        save(path, payload)
    save(FOLDER / "model_catalog.json", dict(official_commit=COMMIT, experiments=models,
        note="0200..0204: scale3 dice+focal; 0500..0504: scale4 dice+bce. Only same 020x family is selected."))
    return payload


def availability(models):
    path = FOLDER / "public_availability.json"
    if path.exists():
        return json.loads(path.read_text("utf-8"))
    response = requests.get(BUCKET, params={"list-type":"2", "prefix":PREFIX}, timeout=(15,45))
    response.raise_for_status()
    xml = response.content
    (FOLDER / "public_weight_listing.xml").write_bytes(xml)
    ns = {"s":"http://s3.amazonaws.com/doc/2006-03-01/"}
    tree = ET.fromstring(xml)
    objects = {}
    for node in tree.findall("s:Contents", ns):
        key = node.findtext("s:Key", namespaces=ns)
        objects[key] = dict(key=key, bytes=int(node.findtext("s:Size", namespaces=ns)),
            etag=node.findtext("s:ETag", namespaces=ns).strip('"'),
            last_modified=node.findtext("s:LastModified", namespaces=ns))

    def head(model):
        key = PREFIX + model["experiment"] + "/model_best.pth"
        assert key in objects, f"Missing public author model: {key}"
        for attempt in range(5):
            try:
                response = requests.head(BUCKET + key, timeout=(15,30))
                response.raise_for_status()
                result = dict(objects[key], experiment=model["experiment"], url=BUCKET+key,
                    head_status=response.status_code, content_length=int(response.headers["Content-Length"]),
                    head_etag=response.headers.get("ETag", "").strip('"'),
                    accept_ranges=response.headers.get("Accept-Ranges"))
                assert result["content_length"] == result["bytes"] and result["head_etag"] == result["etag"]
                return result
            except (requests.RequestException, KeyError):
                if attempt == 4:
                    raise
                time.sleep(2*(attempt+1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        heads = list(pool.map(head, models))
    result = dict(checked_utc=utc(), public_bucket=BUCKET, prefix=PREFIX,
        anonymous_read_only=True, credentials_or_cookies_used=False, source_listing_sha256=hashlib.sha256(xml).hexdigest(),
        objects=heads)
    save(path, result)
    return result


def multipart_etag(path):
    chunks = []
    with path.open("rb") as stream:
        for body in iter(lambda:stream.read(MULTIPART), b""):
            chunks.append(hashlib.md5(body).digest())
    return hashlib.md5(b"".join(chunks)).hexdigest() + "-" + str(len(chunks))


def download_one(item):
    target = FOLDER / "weights" / f"model_best_{item['experiment']}.pth"
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        part = target.with_suffix(".pth.part")
        while True:
            start = part.stat().st_size if part.exists() else 0
            if start > item["bytes"]:
                raise ValueError("Oversized partial weight; refusing silent replacement")
            if start == item["bytes"]:
                part.replace(target)
                break
            end = min(item["bytes"]-1, start+1024*1024-1)
            for attempt in range(6):
                try:
                    headers = {"Accept-Encoding":"identity", "If-Match":'"'+item["etag"]+'"',
                               "Range":f"bytes={start}-{end}"}
                    with requests.get(item["url"], headers=headers, stream=True, timeout=(15,45)) as response:
                        response.raise_for_status()
                        if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{item['bytes']}":
                            raise ValueError("Incorrect bounded weight range; refusing full response")
                        body = response.content
                    if len(body) != end-start+1:
                        raise ValueError("Truncated public weight range")
                    with part.open("ab") as stream:
                        stream.write(body)
                    break
                except (requests.RequestException, ValueError) as error:
                    print(item["experiment"], f"range {start}-{end}", "retry", attempt+1, type(error).__name__, flush=True)
                    if attempt == 5:
                        raise
                    time.sleep(min(10, 2*(attempt+1)))
    assert target.stat().st_size == item["bytes"]
    matched = multipart_etag(target) == item["etag"]
    assert matched, "Public multipart ETag mismatch at the author's established 8 MiB part size"
    return dict(item, path=target.relative_to(ROOT).as_posix(), sha256=sha(target),
                official_multipart_etag_verified=matched, multipart_part_bytes=MULTIPART)


def inspect_cpu():
    import torch
    receipts = json.loads((FOLDER / "download_receipts.json").read_text("utf-8"))
    assert receipts["complete"]
    paths = [("exp_0200", ORIGINAL / "model_best_exp_0200.pth")]
    paths += [(entry["experiment"], ROOT / entry["path"]) for entry in receipts["weights"]]
    values = []
    signature = None
    for experiment, path in paths:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        tensors = [(key.removeprefix("module."), list(value.shape), str(value.dtype)) for key,value in sorted(saved.items())]
        assert all(value.device.type == "cpu" for value in saved.values())
        current = hashlib.sha256(json.dumps(tensors, separators=(",", ":")).encode()).hexdigest()
        if signature is None:
            signature = current
        assert current == signature, "Fold checkpoint tensor architecture differs"
        values.append(dict(experiment=experiment, weight_sha256=sha(path), tensor_count=len(tensors),
                           tensor_shape_dtype_signature=current, devices=["cpu"], first_tensors=tensors[:3], last_tensors=tensors[-3:]))
        print("CPU tensor metadata", experiment, len(tensors), current[:16], flush=True)
        del saved
    report = dict(torch=torch.__version__, checkpoint_shapes_all_match_exp0200=True, models=values,
                  cuda_initialized=torch.cuda.is_initialized(), forward_passes=0, images_or_labels_read=False)
    assert not report["cuda_initialized"]
    save(FOLDER / "cpu_tensor_metadata.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--download", action="store_true")
    mode.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if args.inspect_only:
        inspect_cpu(); return
    models, sources = catalog()
    selection = freeze_fold_selection(models, sources)
    public = availability(models)
    required = [entry for entry in public["objects"] if entry["experiment"] in selection["required_experiments"]]
    total = sum(entry["bytes"] for entry in required)
    assert total <= 5 * 54_000_000
    summary = dict(selection_sha256=sha(FOLDER / "fold_selection.json"), required_experiments=selection["required_experiments"],
        required_object_bytes=total, public_HEADs_passed=len(public["objects"]), GPU_inference_started=False)
    save(FOLDER / "preparation_plan.json", summary)
    print(json.dumps(summary), flush=True)
    if args.plan_only:
        return
    weights = []
    save(FOLDER / "download_receipts.json", dict(complete=False, expected_weights=len(required), weights=weights))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        required.sort(key=lambda item: not (FOLDER/"weights"/f"model_best_{item['experiment']}.pth").exists())
        futures = [pool.submit(download_one, item) for item in required]
        for future in concurrent.futures.as_completed(futures):
            entry = future.result()
            weights.append(entry)
            save(FOLDER / "download_receipts.json", dict(complete=len(weights)==len(required),
                expected_weights=len(required), total_expected_bytes=total, weights=sorted(weights,key=lambda x:x["experiment"])))
            print("verified", entry["experiment"], entry["bytes"], entry["sha256"], flush=True)


if __name__ == "__main__":
    main()
