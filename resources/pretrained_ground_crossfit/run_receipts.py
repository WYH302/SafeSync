"""Fixed author-fold OOF evaluation of already committed actual RGB receipts.

This process never imports learned-codec inference. It replays the frozen
layered/image protocols and consumes independently decoded LC09 PNG receipts.
All probabilities belong to this new weight/preprocessing/threshold/RGB cache.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_layered_receipts as common
from scripts import evaluate_learned_receipts as lic
from safesync.crossfit_ground import CrossfitGroundEvaluator
import numpy as np
from PIL import Image

RES = ROOT / "resources/pretrained_ground_crossfit"
OUT = RES / "receipts_v1"
MANIFEST = ROOT / "data/spacenet7_pilot/confirmation_manifest_v1.json"
MAIN = ROOT / "results/layered/confirmation_v1"
FULL = ROOT / "results/full_cache_controls/gate30"
LIC = ROOT / "results/learned_codec/confirmation_q1_8_v1"
BUDGETS = [32768, 65536, 131072]
METHODS = list(common.METHODS)
FULL_METHOD = "full_cache_gate30"
LIC_METHOD = "learned_hyperprior_q1_8"


def checked_freeze(path, value):
    if path.exists() and common.read(path) != value:
        raise ValueError("frozen crossfit input identity changed: " + str(path))
    if not path.exists():
        common.save_json(path, value)


def read_source(run, manifest, index, selected):
    """Require a complete immutable producer, including its source snapshots."""
    spec, progress = [common.read(run / name) for name in ["run_spec.json", "progress.json"]]
    if progress["status"] != "complete" or progress["expected"] != progress["records"]:
        raise ValueError("producer is incomplete")
    for key, name in [("spec_sha256", "run_spec.json"), ("records_sha256", "records.csv")]:
        if progress[key] != common.sha(run / name):
            raise ValueError("producer completion SHA changed")
    if spec["manifest_sha256"] != common.sha(MANIFEST) or spec["area_indices"] != [index]:
        raise ValueError("producer cohort or area differs")
    if spec["budgets"] != BUDGETS or not set(selected).issubset(spec["methods"]):
        raise ValueError("producer lacks the fixed method/budget recipe")
    for name, expected in spec["source_sha256"].items():
        if common.sha(ROOT / name) != expected or common.sha(run / "source_snapshot" / name.replace("/", "__")) != expected:
            raise ValueError("frozen transmission source differs: " + name)
    all_rows = list(csv.DictReader((run / "records.csv").open(encoding="utf-8", newline="")))
    if len(all_rows) != progress["expected"]:
        raise ValueError("producer row count differs")
    aid, months = manifest["aois"][index]["aoi_id"], manifest["aois"][index]["months"][1:]
    rows = [row for row in all_rows if row["method"] in selected]
    for method in selected:
        for cap in BUDGETS:
            subset = sorted([r for r in rows if r["method"] == method and int(r["budget"]) == cap], key=lambda r: int(r["step"]))
            expected = [(m["month"], i) for i, m in enumerate(months, 1)]
            if [(r["month"], int(r["step"])) for r in subset] != expected or any(r["aoi"] != aid for r in subset):
                raise ValueError("producer sequence reordered, omitted, or duplicated")
    return spec, rows


def input_plan():
    manifest = common.read(MANIFEST)
    selection = common.read(RES / "fold_selection.json")
    calibration = common.read(RES / "calibration.json")
    confirm = [r for r in selection["targets"] if r["role"] == "confirmation"]
    if [r["aoi_id"] for r in confirm] != [a["aoi_id"] for a in manifest["aois"]]:
        raise ValueError("author-fold ordering differs from the frozen confirmation manifest")
    if calibration["confirmation_scores_used"] or calibration["weights_refitted"]:
        raise ValueError("OOF threshold was not restricted to the original two calibration AOIs")
    sources, plans = {}, []
    for index, aoi in enumerate(manifest["aois"]):
        main_spec, main_rows = read_source(MAIN / f"area_{index}", manifest, index, METHODS)
        full_spec, full_rows = read_source(FULL / f"area_{index}", manifest, index, ["full_cache_mse"])
        lic_spec, receipt, lic_rows, lic_hashes = lic.load_source(LIC, aoi, index)
        sources.update(lic_hashes)
        for base in [MAIN / f"area_{index}", FULL / f"area_{index}"]:
            for name in ["run_spec.json", "progress.json", "records.csv"]:
                path = base / name
                sources[common.relative(path)] = common.sha(path)
        plans.append((main_spec, main_rows, full_spec, full_rows, lic_spec, receipt, lic_rows))
    paths = [Path(__file__), ROOT / "safesync/crossfit_ground.py", ROOT / "safesync/pretrained_ground.py",
        ROOT / "scripts/calibrate_crossfit_ground.py", ROOT / "scripts/evaluate_layered_receipts.py",
        ROOT / "scripts/evaluate_pretrained_receipts.py", ROOT / "scripts/evaluate_learned_receipts.py", MANIFEST,
        RES / "fold_selection.json", RES / "calibration.json", RES / "calibration_run_spec.json",
        RES / "download_receipts.json", RES / "cpu_tensor_metadata.json",
        ROOT / "resources/pretrained_ground/metadata.json", ROOT / "resources/pretrained_ground/calibration.json"]
    sources.update({common.relative(path): common.sha(path) for path in paths})
    updates = sum(len(a["months"]) - 1 for a in manifest["aois"])
    spec = {"schema": "author-fold-OOF-actual-receiver-tasks-v1", "part_of_original_blind_confirmation": False,
        "scope": "author validation-fold best-epoch weights; expected gradient-training exclusion; not untouched test",
        "aoi_ids": [a["aoi_id"] for a in manifest["aois"]], "fixed_fold_bindings": confirm,
        "methods": METHODS + [FULL_METHOD, LIC_METHOD], "budgets": BUDGETS,
        "expected_updates": updates, "expected_main_outputs": updates * len(METHODS) * len(BUDGETS),
        "expected_gate30_outputs": updates * len(BUDGETS), "expected_LIC_outputs": updates * len(BUDGETS),
        "expected_outputs": updates * (len(METHODS) + 2) * len(BUDGETS),
        "thresholds": {"building": .5, "event": calibration["change_threshold"]},
        "threshold_calibration_AOIs": calibration["calibration_aoi_ids"],
        "weight_sha256_by_experiment": calibration["weight_sha256_by_experiment"],
        "source_hashes": sources, "source_current_RGB_or_features_opened": False,
        "input_boundary": "native actual receiver RGB only; labels offline after receipt reconstruction",
        "cache_identity": "new OOF model/weight+preprocessing+calibration+threshold+native RGB bytes and shape",
        "old_exp0200_probabilities_reused": False, "no_training_or_transport_or_parameter_changes": True,
        "paired_bootstrap": {"resamples": 10000, "seed": 20261029, "confidence": 1 - .05/3,
            "unit": "8 paired AOIs, sum month counts before F1", "comparisons": ["Gaussian - WebP", "Gaussian - learned q1..8"],
            "correction_scope": "three budgets separately for each task/comparator; no across-comparator global-significance claim"}}
    return manifest, plans, spec


def freeze(spec):
    OUT.mkdir(parents=True, exist_ok=True)
    checked_freeze(OUT / "run_spec.json", spec)
    snap = OUT / "source_snapshot"
    snap.mkdir(exist_ok=True)
    for name, expected in spec["source_hashes"].items():
        source, target = ROOT / name, snap / name.replace("/", "__")
        if not target.exists():
            shutil.copyfile(source, target)
        if common.sha(target) != expected:
            raise ValueError("OOF frozen source snapshot SHA differs")
    marker = OUT / "frozen_before_confirmation_inference.json"
    if not marker.exists():
        common.save_json(marker, {"status": "frozen", "frozen_utc": common.now(),
            "spec_sha256": common.sha(OUT / "run_spec.json"), "expected_outputs": spec["expected_outputs"]})


def probability(evaluator, rgb, out, model_hash, threshold_hash, origin):
    """Read only this new cache: never search any exp0200 prediction folder."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    rgb_sha = common.sha_bytes(rgb.tobytes())
    key = common.canonical_hash({"actual_native_RGB": rgb_sha, "shape": list(rgb.shape),
        "OOF_model_identity": model_hash, "threshold_hash": threshold_hash})
    path = out / "probabilities" / (key + ".npz")
    cached, timing = path.exists(), None
    if cached:
        with np.load(path, allow_pickle=False) as archive:
            if any(str(archive[k]) != value for k, value in {"input_rgb_sha256":rgb_sha,
                "model_hash":model_hash, "threshold_hash":threshold_hash}.items()):
                raise ValueError("isolated OOF probability cache identity changed")
            result = archive["probability"].copy()
    else:
        result = evaluator.predict(Image.fromarray(rgb))
        timing = evaluator.last_timing
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, probability=result, input_rgb_sha256=rgb_sha,
            model_hash=model_hash, threshold_hash=threshold_hash, origin=json.dumps(origin))
        temporary.replace(path)
    if result.shape != rgb.shape[:2] or not np.isfinite(result).all() or result.min() < 0 or result.max() > 1:
        raise ValueError("OOF probability shape/range differs from native receiver input")
    return result, {"prediction_input_RGB_sha256":rgb_sha, "prediction_native_shape":list(rgb.shape),
        "probability_path":common.relative(path), "probability_file_sha256":common.sha(path),
        "probability_array_sha256":common.sha_bytes(result.tobytes()), "probability_cache_hit":cached,
        "prediction_inference_timing":timing, "cache_origin":"new isolated OOF actual receiver cache only"}


def layered_frames(aoi, source_spec, source_rows, method, cap):
    aid, first = aoi["aoi_id"], aoi["months"][0]
    sequence = sorted([r for r in source_rows if r["method"] == method and int(r["budget"]) == cap], key=lambda r:int(r["step"]))
    bootstrap_path = ROOT / sequence[0]["bootstrap_path"]
    bootstrap = bootstrap_path.read_bytes()
    w, h = first["dimensions"]
    receiver = common.ReplayedSequence(bootstrap, (h,w), hashlib.sha256(aid.encode()).digest()[:16], source_spec["method_configs"].get(method))
    if receiver.initial_ack != bootstrap_path.with_suffix(".ack").read_bytes():
        raise ValueError("replayed OOF bootstrap ACK differs from source")
    initial = {"bootstrap_path":common.relative(bootstrap_path), "bootstrap_packet_sha256":common.sha_bytes(bootstrap),
        "bootstrap_down_bytes":len(bootstrap), "bootstrap_ack_bytes":len(receiver.initial_ack),
        "bootstrap_ack_sha256":common.sha_bytes(receiver.initial_ack)}
    yield receiver.image()[:h,:w].copy(), initial
    down, up = len(bootstrap), len(receiver.initial_ack)
    for source in sequence:
        path, step = ROOT / source["packet_path"], int(source["step"])
        packet, actual_ack = path.read_bytes(), path.with_suffix(".ack").read_bytes()
        if common.sha_bytes(packet) != source["packet_sha256"] or common.sha_bytes(actual_ack) != source["ack_sha256"]:
            raise ValueError("actual source packet/ACK bytes changed")
        rgb, ack, state = receiver.apply(packet, step)
        rgb_sha = common.sha_bytes(rgb.tobytes())
        if ack != actual_ack or rgb_sha != source["receiver_rgb_sha256"]:
            raise ValueError("OOF replay differs from actual receiver RGB/ACK")
        down, up = down + len(packet), up + len(ack)
        measured = {"down_bytes":len(packet), "up_bytes":len(ack), "reference_up_bytes":0,
            "bootstrap_down_bytes":len(bootstrap), "bootstrap_up_bytes":len(receiver.initial_ack),
            "cumulative_down_bytes":down, "cumulative_up_bytes":up, "sender_summary_bytes":state["receiver_cache_bytes"]}
        if len(packet) > cap or any(value != int(source[k]) for k,value in measured.items()):
            raise ValueError("OOF replay actual byte/state-cache accounting changed")
        yield rgb[:h,:w].copy(), {"step":step, "month":source["month"], "packet_path":common.relative(path),
            "packet_sha256":common.sha_bytes(packet), "ack_sha256":common.sha_bytes(ack),
            "receiver_RGB_sha256_verified":rgb_sha, "down_bytes":len(packet), "ack_bytes":len(ack),
            "cumulative_down_bytes":down, "cumulative_up_bytes":up, "source_encoding_ms":float(source["compute_ms"]),
            "protocol":"LB01/LA01" if receiver.layered else "SP03/IF06", **state}


def learned_frames(aoi, receipt, rows, cap):
    sequence = sorted([row for row in rows if row["cap"] == cap], key=lambda r:r["step"])
    state, native, initial = lic.initialization(LIC, aoi, receipt, sequence[0])
    yield native, initial
    down, up = initial["bootstrap_down_bytes"], initial["bootstrap_ack_bytes"]
    for source, month in zip(sequence, aoi["months"][1:]):
        state, native, wire = lic.receiver_png_state(LIC, source, state, month)
        down, up = down + source["down_bytes"], up + source["up_bytes"]
        yield native, {**wire, "step":source["step"], "month":month["month"], "protocol":"SP03/LC09",
            "receiver_RGB_sha256_verified":source["receiver_rgb_sha256"], "down_bytes":source["down_bytes"],
            "ack_bytes":source["up_bytes"], "cumulative_down_bytes":down, "cumulative_up_bytes":up,
            "receiver_cache_sha256":None, "receiver_cache_bytes":0, "receiver_history_digest":state.digest().hex(),
            "receiver_archive_RGB_sha256":None, "receiver_coarse_RGB_sha256":None, "applied_ROI_count":None,
            "source_encoding_ms":source["sender_search_ms_shared"], "sender_search_time_shared_across_caps":True}


def measure_frame(evaluator, native, wire, initial, reference, reference_key, out, model_hash, threshold_hash, method, cap, aoi, gt, spec_sha):
    building, event, valid, gt_hashes = gt
    identity = common.row_identity(aoi["aoi_id"], method, cap, wire["step"])
    key = common.canonical_hash({"identity":identity, "packet":wire["packet_sha256"],
        "actual_native_RGB":common.sha_bytes(native.tobytes()), "GT":gt_hashes,
        "model":model_hash, "threshold":threshold_hash, "frozen_spec":spec_sha})
    path = out / "records" / (key + ".json")
    if path.exists():
        row = common.read(path)
        if common.sha(ROOT / row["probability_path"]) != row["probability_file_sha256"]:
            raise ValueError("saved new OOF probability file changed")
    else:
        prediction, receipt = probability(evaluator, native, out, model_hash, threshold_hash,
            {"actual_packet":wire["packet_path"], "packet_sha256":wire["packet_sha256"]})
        event_score = np.maximum(prediction-reference, 0)
        row = {**identity, "month":wire["month"], **wire,
            "author_validation_fold":evaluator.binding["author_validation_fold"], "model_experiment":evaluator.experiment,
            "weight_sha256":evaluator.metadata["weight_sha256"], "model_hash":model_hash,
            "threshold_hash":threshold_hash, "gt_sha256":gt_hashes, "reference_receipt_key":reference_key,
            "reference_up_bytes":0, "bootstrap_down_bytes":initial["bootstrap_down_bytes"],
            "bootstrap_ack_bytes":initial["bootstrap_ack_bytes"], "part_of_original_blind_confirmation":False,
            **receipt,
            **common.flat_counts("building", common.counts(prediction >= evaluator.threshold, building, valid)),
            **common.flat_counts("event", common.counts(event_score >= evaluator.change_threshold, event, valid)),
            **common.flat_counts("initial_only_building", common.counts(reference >= evaluator.threshold, building, valid)),
            **common.flat_counts("initial_only_event", common.counts(np.zeros_like(event,bool), event, valid))}
        common.save_json(path, row)
    if any(row[k] != value for k,value in identity.items()):
        raise ValueError("OOF cached metric has a different method/budget/sequence identity")
    return row


def evaluate_area(index, aoi, plan, spec, stop_after):
    out = OUT / f"area_{index}"
    expected = (len(aoi["months"])-1) * (len(METHODS)+2) * len(BUDGETS)
    if (out / "progress.json").exists():
        done = common.read(out / "progress.json")
        if done["status"] == "complete":
            if done["expected"] != expected or done["records_sha256"] != common.sha(out / "records.csv"):
                raise ValueError("completed OOF AOI changed")
            return common.read(out / "summary.json")["outputs"], False
    evaluator = CrossfitGroundEvaluator(ROOT, aoi["aoi_id"])
    lic.require_no_codec_imports()
    lic.assert_ground_flags(lic.numerical_flags(evaluator.torch))
    spec_sha = common.sha(OUT / "run_spec.json")
    model_hash = common.canonical_hash(json.loads(evaluator.identity))
    threshold_hash = common.canonical_hash(spec["thresholds"])
    checked_freeze(out / "run_spec.json", {"parent_spec_sha256":spec_sha, "area_index":index,
        "aoi_id":aoi["aoi_id"], "model_hash":model_hash, "threshold_hash":threshold_hash,
        "model_identity":json.loads(evaluator.identity), "expected_outputs":expected,
        "numerical_flags":lic.numerical_flags(evaluator.torch), "model_metadata":evaluator.metadata})
    main_spec, main_rows, full_spec, full_rows, _, receipt, lic_rows = plan
    results, references, initials, gt_cache = [], {}, {}, {}
    first, aid = aoi["months"][0], aoi["aoi_id"]
    month_lookup = {month["month"]:month for month in aoi["months"]}
    sequences = [(method, cap, layered_frames(aoi, main_spec, main_rows, method, cap)) for method in METHODS for cap in BUDGETS]
    sequences += [(FULL_METHOD, cap, layered_frames(aoi, full_spec, full_rows, "full_cache_mse", cap)) for cap in BUDGETS]
    sequences += [(LIC_METHOD, cap, learned_frames(aoi, receipt, lic_rows, cap)) for cap in BUDGETS]
    try:
        for method, cap, frames in sequences:
            initial_rgb, initial = next(frames)
            reference_key = common.canonical_hash({"native_initial_RGB":common.sha_bytes(initial_rgb.tobytes()),
                "shape":list(initial_rgb.shape), "model":model_hash, "threshold":threshold_hash})
            if reference_key not in references:
                references[reference_key], prediction_receipt = probability(evaluator, initial_rgb, out, model_hash,
                    threshold_hash, {"bootstrap":initial["bootstrap_path"]})
            else:
                _, prediction_receipt = probability(evaluator, initial_rgb, out, model_hash, threshold_hash,
                    {"bootstrap":initial["bootstrap_path"]})
            initials[f"{method}_{cap}"] = {**initial, **prediction_receipt, "reference_receipt_key":reference_key}
            common.save_json(out / "initial_receipts.json", initials)
            for native, wire in frames:
                month_name = wire["month"]
                if month_name not in gt_cache:
                    event_path = MAIN / f"area_{index}/gt" / aid / month_name / "new_track.png"
                    gt_cache[month_name] = common.offline_masks(month_lookup[month_name], first, aid, event_path, True)
                row = measure_frame(evaluator, native, wire, initial, references[reference_key], reference_key,
                    out, model_hash, threshold_hash, method, cap, aoi, gt_cache[month_name], spec_sha)
                results.append(row)
                common.save_json(out / "progress.json", {"status":"in_progress", "completed":len(results),
                    "expected":expected, "method":method, "budget":cap, "month":month_name,
                    "all_receiver_RGB_ACK_match":True, "updated_utc":common.now()})
                if stop_after and len(results) >= stop_after:
                    common.write_csv(out / "records.csv", results)
                    common.save_json(out / "summary.partial.json", common.aggregate(results))
                    return len(results), True
            common.write_csv(out / "records.csv", results)
            common.save_json(out / "summary.partial.json", common.aggregate(results))
            print(f"OOF area={index} {evaluator.experiment} {method} cap={cap} rows={len(results)}/{expected}", flush=True)
        identities = {(r["aoi"],r["method"],r["budget"],r["step"]) for r in results}
        if len(results) != expected or len(identities) != expected:
            raise ValueError("OOF AOI cohort incomplete or duplicated")
        lic.require_no_codec_imports()
        lic.assert_ground_flags(lic.numerical_flags(evaluator.torch))
        summary = {**common.aggregate(results), "status":"complete", "outputs":len(results),
            "parent_spec_sha256":spec_sha, "model_hash":model_hash, "threshold_hash":threshold_hash,
            "model_experiment":evaluator.experiment, "weight_sha256":evaluator.metadata["weight_sha256"],
            "author_validation_fold":evaluator.binding["author_validation_fold"], "scope":spec["scope"],
            "all_actual_RGB_ACK_and_state_histories_match":True, "source_current_RGB_or_features_opened":False,
            "old_exp0200_probabilities_reused":False, "numerical_flags_unchanged":True}
        common.save_json(out / "summary.json", summary)
        common.save_json(out / "progress.json", {"status":"complete", "completed":len(results), "expected":expected,
            "records_sha256":common.sha(out / "records.csv"), "summary_sha256":common.sha(out / "summary.json"),
            "spec_sha256":common.sha(out / "run_spec.json"), "updated_utc":common.now()})
        return len(results), False
    finally:
        evaluator.close()


def combined_summary(manifest, spec):
    per_aoi, sources, completed = [], {}, []
    for index, aoi in enumerate(manifest["aois"]):
        folder = OUT / f"area_{index}"
        marker = folder / "progress.json"
        if not marker.exists() or common.read(marker)["status"] != "complete":
            continue
        progress = common.read(marker)
        for key,name in [("records_sha256","records.csv"),("summary_sha256","summary.json"),("spec_sha256","run_spec.json")]:
            if progress[key] != common.sha(folder / name):
                raise ValueError("OOF completed area artifact changed")
            sources[common.relative(folder/name)] = progress[key]
        area = common.read(folder / "summary.json")
        if area["parent_spec_sha256"] != common.sha(OUT / "run_spec.json"):
            raise ValueError("OOF completed area refers to another frozen recipe")
        per_aoi.extend(area["per_aoi_seed"])
        completed.append(index)
    output = {"status":"complete" if len(completed) == len(manifest["aois"]) else "in_progress",
        "completed_areas":completed, "expected_areas":len(manifest["aois"]),
        "outputs":sum(g["frames"] for g in per_aoi), "expected_outputs":spec["expected_outputs"],
        "scope":spec["scope"], "thresholds":spec["thresholds"], "per_aoi_seed":per_aoi,
        "statistical_unit":"sum month counts within AOI, then equal AOI weight; one fixed seed per primary method",
        "source_spec_sha256":common.sha(OUT / "run_spec.json"), "area_artifact_sha256":sources}
    macro = []
    for method in spec["methods"]:
        for cap in BUDGETS:
            members = [g for g in per_aoi if (g["method"],g["budget"]) == (method,cap)]
            if not members:
                continue
            result = {"method":method, "budget":cap, "AOIs":len(members)}
            for task in common.PREFIXES:
                result[task] = {metric:common.mean_defined([g[task][metric] for g in members]) for metric in common.METRICS}
            for name in ["mean_update_bidirectional_bytes", "total_bytes_including_bootstrap", "bootstrap_bidirectional_bytes"]:
                result[name] = float(np.mean([g[name] for g in members]))
            macro.append(result)
    output["macro_aoi"] = macro
    common.save_json(OUT / "summary.json", output)
    common.write_csv(OUT / "macro.csv", [{"method":r["method"],"budget":r["budget"],"AOIs":r["AOIs"],
        **{f"{t}_{m}":r[t][m] for t in common.PREFIXES for m in common.METRICS},
        **{k:r[k] for k in ["mean_update_bidirectional_bytes","total_bytes_including_bootstrap","bootstrap_bidirectional_bytes"]}} for r in macro])
    if output["status"] == "complete":
        if output["outputs"] != spec["expected_outputs"]:
            raise ValueError("complete OOF summary has a wrong cohort count")
        contrasts = paired(per_aoi, spec)
        common.save_json(OUT / "paired_comparisons.json", contrasts)
        common.write_csv(OUT / "paired_comparisons.csv", [{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in contrasts])
        common.save_json(OUT / "complete.json", {"status":"complete", "outputs":output["outputs"],
            "source_spec_sha256":common.sha(OUT / "run_spec.json"), "summary_sha256":common.sha(OUT / "summary.json"),
            "paired_comparisons_sha256":common.sha(OUT / "paired_comparisons.json"), "completed_utc":common.now()})
    return output


def paired(groups, spec):
    lookup = {(g["method"],g["budget"],g["aoi"]):g for g in groups}
    ids = spec["aoi_ids"]
    rng = np.random.default_rng(spec["paired_bootstrap"]["seed"])
    resamples = rng.integers(0,len(ids),(10000,len(ids)))
    tail = (1-spec["paired_bootstrap"]["confidence"])/2
    contrasts = []
    for comparator in ["full_webp",LIC_METHOD]:
        for cap in BUDGETS:
            for task in ["building","event"]:
                pairs = []
                for aid in ids:
                    left,right = lookup[("gaussian_20261029",cap,aid)],lookup[(comparator,cap,aid)]
                    for support in ["valid_pixels","positive_pixels"]:
                        if left[task][support] != right[task][support]:
                            raise ValueError("paired OOF ground support changed across actual streams")
                    if left[task]["positive_pixels"] > 0:
                        pairs.append({"aoi":aid,"gaussian_f1":left[task]["f1"],"comparator_f1":right[task]["f1"],
                            "valid_pixels":left[task]["valid_pixels"],"positive_pixels":left[task]["positive_pixels"]})
                values = np.asarray([p["gaussian_f1"]-p["comparator_f1"] for p in pairs])
                if len(pairs) != len(ids):
                    # A positive-support subset would need an explicitly sized paired bootstrap.
                    raise ValueError("fixed confirmation positive-support AOIs differ from its predeclared all-eight cohort")
                samples = values[resamples].mean(axis=1)
                low,high = np.quantile(samples,[tail,1-tail])
                contrasts.append({"method":"gaussian_20261029","comparator":comparator,"budget":cap,"task":task,
                    "paired_AOIs":len(pairs),"difference_f1":float(values.mean()),"ci_low":float(low),"ci_high":float(high),
                    "confidence":spec["paired_bootstrap"]["confidence"],"bootstrap_resamples":10000,"bootstrap_seed":20261029,
                    "scope":"OOF validation-best additional analysis; three-budget correction per task/comparator",
                    "pairs":pairs})
    return contrasts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-inputs",action="store_true")
    parser.add_argument("--stop-after",type=int,default=0,help="interface sanity prefix; recipe still declares the complete cohort")
    parser.add_argument("--areas",type=int,nargs="*",default=list(range(8)))
    args = parser.parse_args()
    started = time.perf_counter()
    lic.require_no_codec_imports()
    manifest,plans,spec = input_plan()
    print(json.dumps({"expected_outputs":spec["expected_outputs"],"main":spec["expected_main_outputs"],
        "gate30":spec["expected_gate30_outputs"],"LIC":spec["expected_LIC_outputs"],"thresholds":spec["thresholds"]}),flush=True)
    if args.check_inputs:
        return
    freeze(spec)
    for index in args.areas:
        if index not in range(8):
            raise ValueError("AOI outside the fixed eight")
        count,stopped = evaluate_area(index,manifest["aois"][index],plans[index],spec,args.stop_after)
        status = combined_summary(manifest,spec)
        print(json.dumps({"area":index,"outputs":count,"status":status["status"],"completed_areas":status["completed_areas"],
            "invocation_wall_seconds":time.perf_counter()-started}),flush=True)
        if stopped:
            break


if __name__ == "__main__":
    main()
