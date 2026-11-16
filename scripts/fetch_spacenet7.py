"""Fetch a frozen, geographic SpaceNet 7 pilot from the official public bucket.

Only public HTTP GETs are used. Re-running verifies and reuses original files.
AOI ordering is frozen before opening imagery or labels and is prefix stable.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data/spacenet7_pilot"
sys.path.insert(0, str(DEFAULT_OUT / ".deps"))

import numpy as np
import requests
from PIL import Image, __version__ as pillow_version

BUCKETS = ["https://spacenet-dataset.s3.amazonaws.com/",
           "https://s3.amazonaws.com/spacenet-dataset/"]
PREFIX = "spacenet/SN7_buildings/train/"
XML_NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
PREPARATION_VERSION = 2


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def digest_file(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def list_bucket(prefix, cache_path, delimiter=None):
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    result = {"prefix": prefix, "retrieved_utc": utc_now(), "objects": [], "prefixes": []}
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": 1000}
        if delimiter:
            params["delimiter"] = delimiter
        if token:
            params["continuation-token"] = token
        for attempt in range(8):
            try:
                response = requests.get(BUCKETS[attempt % 2], params=params, timeout=(15, 45))
                response.raise_for_status()
                tree = ET.fromstring(response.content)
                break
            except (requests.RequestException, ET.ParseError):
                if attempt == 7:
                    raise
                time.sleep(min(8, attempt + 1))
        for node in tree.findall("s:Contents", XML_NS):
            result["objects"].append({
                "key": node.findtext("s:Key", namespaces=XML_NS),
                "bytes": int(node.findtext("s:Size", namespaces=XML_NS)),
                "etag": node.findtext("s:ETag", namespaces=XML_NS).strip('"'),
                "last_modified": node.findtext("s:LastModified", namespaces=XML_NS)})
        result["prefixes"].extend(n.text for n in tree.findall("s:CommonPrefixes/s:Prefix", XML_NS))
        if tree.findtext("s:IsTruncated", namespaces=XML_NS) != "true":
            break
        token = tree.findtext("s:NextContinuationToken", namespaces=XML_NS)
    write_json(cache_path, result)
    return result


def tile_center(aoi):
    """Geographic proxy from the documented-in-ID Web Mercator tile numbers."""
    x, y, z = map(int, aoi.rsplit("_", 3)[1:])
    scale = 2 ** z
    lon = (x + 0.5) / scale * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 0.5) / scale))))
    return lon, lat


def angular_distance(a, b):
    lon1, lat1 = map(math.radians, a)
    lon2, lat2 = map(math.radians, b)
    value = math.sin((lat1 - lat2) / 2) ** 2
    value += math.cos(lat1) * math.cos(lat2) * math.sin((lon1 - lon2) / 2) ** 2
    return 2 * math.asin(math.sqrt(min(1.0, value)))


def freeze_selection(out):
    path = out / "selection.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    listing = list_bucket(PREFIX, out / "source/listing_train_aois.json", delimiter="/")
    candidates = sorted(p.rstrip("/").split("/")[-1] for p in listing["prefixes"])
    ordered = [candidates[0]]
    while len(ordered) < len(candidates):
        remaining = [a for a in candidates if a not in ordered]
        chosen = min(remaining, key=lambda a: (
            -min(angular_distance(tile_center(a), tile_center(b)) for b in ordered), a))
        ordered.append(chosen)
    result = {
        "created_utc": utc_now(), "candidate_aoi_ids": candidates, "ordered_aoi_ids": ordered,
        "rule": "Lexicographic first seed, then greedy maximum minimum great-circle distance; ties lexicographic. Uses AOI-ID tile centers only, before image/label inspection. Expanding AOI count preserves this prefix.",
        "tile_center_lon_lat": {a: tile_center(a) for a in ordered},
        "source_listing_sha256": digest_file(out / "source/listing_train_aois.json")}
    write_json(path, result)
    return result


def native_path(out, key):
    relative = PurePosixPath(key).relative_to(PREFIX)
    if ".." in relative.parts:
        raise ValueError("unsafe bucket key")
    return out / "native" / Path(*relative.parts)


def fetch_object(out, obj):
    target = native_path(out, obj["key"])
    target.parent.mkdir(parents=True, exist_ok=True)
    etag_is_md5 = bool(re.fullmatch(r"[a-f0-9]{32}", obj["etag"]))
    if target.exists() and target.stat().st_size == obj["bytes"]:
        if not etag_is_md5 or digest_file(target, "md5") == obj["etag"]:
            return target
    part = target.with_suffix(target.suffix + ".part")
    for attempt in range(10):
        try:
            start = part.stat().st_size if part.exists() else 0
            if start == obj["bytes"]:
                if etag_is_md5 and digest_file(part, "md5") != obj["etag"]:
                    part.write_bytes(b"")
                    raise ValueError("MD5 mismatch; retrying object from byte zero")
                part.replace(target)
                return target
            if start > obj["bytes"]:
                part.write_bytes(b"")
                start = 0
            headers = {"Accept-Encoding": "identity"}
            if start:
                headers["Range"] = f"bytes={start}-"
            url = BUCKETS[attempt % 2] + obj["key"]
            with requests.get(url, headers=headers, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                append = bool(start and response.status_code == 206)
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                    raise ValueError("wrong HTTP range")
                with part.open("ab" if append else "wb") as stream:
                    for chunk in response.iter_content(256 * 1024):
                        stream.write(chunk)
            if part.stat().st_size != obj["bytes"]:
                raise ValueError("truncated object")
            if etag_is_md5 and digest_file(part, "md5") != obj["etag"]:
                part.write_bytes(b"")
                raise ValueError("MD5 mismatch")
            part.replace(target)
            return target
        except (requests.RequestException, ValueError) as error:
            print(f"retry {attempt+1}: {target.name}: {type(error).__name__}: {error}", flush=True)
            if attempt == 9:
                raise
            time.sleep(min(10, attempt + 1))
    raise RuntimeError("unreachable")


def month_jobs(out, selection, aoi_count, month_limit):
    jobs = []
    for index, aoi in enumerate(selection["ordered_aoi_ids"][:aoi_count]):
        listing = list_bucket(PREFIX + aoi + "/", out / f"source/listing_{aoi}.json")
        by_key = {o["key"]: o for o in listing["objects"]}
        images = sorted(o for o in by_key if "/images/" in o and o.endswith(".tif"))
        if month_limit:
            images = images[:month_limit]
        for image_key in images:
            stem = PurePosixPath(image_key).stem
            base = PREFIX + aoi + "/"
            keys = {"image": image_key,
                    "building_labels": base + "labels_match/" + stem + "_Buildings.geojson",
                    "udm_labels": base + "labels/" + stem + "_UDM.geojson"}
            missing = [key for key in keys.values() if key not in by_key]
            if missing:
                raise ValueError(f"Missing official labels for {stem}: {missing}")
            match = re.search(r"global_monthly_(\d{4})_(\d{2})_", stem)
            jobs.append({"aoi_id": aoi, "aoi_order": index, "month": "-".join(match.groups()),
                         "objects": {kind: by_key[key] for kind, key in keys.items()}})
    # First two real months from each geography become available early.
    return sorted(jobs, key=lambda j: (j["month"], j["aoi_order"]))


def geojson_geometries(path, target_crs):
    from rasterio.crs import CRS
    from rasterio.warp import transform_geom
    data = json.loads(path.read_text(encoding="utf-8"))
    source_name = data.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
    source_crs = CRS.from_string(source_name)
    geometries = []
    ids = []
    for feature in data["features"]:
        geom = feature.get("geometry")
        if not geom or not geom.get("coordinates"):
            continue
        if source_crs != target_crs:
            geom = transform_geom(source_crs, target_crs, geom, precision=-1)
        geometries.append(geom)
        properties = feature.get("properties") or {}
        ids.append(properties.get("id", properties.get("Id", feature.get("id"))))
    return geometries, ids, source_crs.to_string()


def prepare_month(out, job):
    import rasterio
    from rasterio.features import rasterize
    from rasterio.warp import transform_bounds
    paths = {kind: fetch_object(out, obj) for kind, obj in job["objects"].items()}
    with rasterio.open(paths["image"]) as source:
        native = source.read()
        if source.count != 4 or native.dtype != np.uint8:
            raise ValueError(f"Unexpected imagery bands/dtype: {source.count}/{native.dtype}")
        if [x.name for x in source.colorinterp[:3]] != ["red", "green", "blue"]:
            raise ValueError("RGB interpretation is not explicit")
        crs, transform = source.crs, source.transform
        shape = (source.height, source.width)
        native_valid = np.all(source.read_masks([1, 2, 3]) > 0, axis=0) & (native[3] > 0)
        building_geoms, track_ids, label_crs = geojson_geometries(paths["building_labels"], crs)
        udm_geoms, _, udm_crs = geojson_geometries(paths["udm_labels"], crs)
        building = rasterize([(g, 1) for g in building_geoms], out_shape=shape,
                             transform=transform, fill=0, all_touched=False, dtype="uint8") if building_geoms else np.zeros(shape, np.uint8)
        # Conservative exclusion: every pixel touched by official unusable polygons.
        udm = rasterize([(g, 1) for g in udm_geoms], out_shape=shape, transform=transform,
                        fill=0, all_touched=True, dtype="uint8") if udm_geoms else np.zeros(shape, np.uint8)
        valid = native_valid & (udm == 0)
        geo = {"crs": crs.to_string(), "native_transform": list(transform)[:6],
               "bounds": list(source.bounds), "bounds_lon_lat": list(transform_bounds(crs, "EPSG:4326", *source.bounds)),
               "dimensions": [source.width, source.height], "dtype": str(native.dtype),
               "channels": [x.name for x in source.colorinterp], "nodata": source.nodata,
               "label_crs": label_crs, "udm_crs": udm_crs}
    folder = out / "prepared" / job["aoi_id"] / job["month"]
    folder.mkdir(parents=True, exist_ok=True)
    outputs = {"rgb_path": folder / "rgb.png", "building_mask_path": folder / "building.png",
               "valid_mask_path": folder / "valid.png", "udm_mask_path": folder / "udm.png",
               "native_array_path": folder / "rgba.npy", "building_track_ids_path": folder / "track_ids.json"}
    Image.fromarray(np.moveaxis(native[:3], 0, -1)).save(outputs["rgb_path"])
    Image.fromarray(building).save(outputs["building_mask_path"])
    Image.fromarray(valid.astype("uint8")).save(outputs["valid_mask_path"])
    Image.fromarray(udm).save(outputs["udm_mask_path"])
    np.save(outputs["native_array_path"], np.moveaxis(native, 0, -1), allow_pickle=False)
    write_json(outputs["building_track_ids_path"], {"ids_in_feature_order": track_ids,
               "source_property": "Id", "source": paths["building_labels"].relative_to(ROOT).as_posix()})
    objects = {}
    for kind, obj in job["objects"].items():
        objects[kind] = {**obj, "url": BUCKETS[0] + obj["key"],
                         "path": paths[kind].relative_to(ROOT).as_posix(), "sha256": digest_file(paths[kind])}
    result = {"month": job["month"], "preparation_version": PREPARATION_VERSION, **geo,
              **{key: value.relative_to(ROOT).as_posix() for key, value in outputs.items()},
              "native_image_path": objects["image"]["path"],
              "label_geojson_path": objects["building_labels"]["path"],
              "udm_geojson_path": objects["udm_labels"]["path"], "source_objects": objects,
              "derived_sha256": {key: digest_file(value) for key, value in outputs.items()},
              "valid_fraction": float(valid.mean()), "udm_fraction": float(udm.mean()),
              "building_fraction_valid": float(building[valid].mean()) if valid.any() else None,
              "building_count": len(building_geoms), "unique_track_id_count": len(set(track_ids)),
              "missing_track_id_count": sum(x is None for x in track_ids),
              "status": "ready" if valid.any() else "all_unusable"}
    write_json(folder / "record.json", result)
    return job["aoi_id"], result


def manifest(out, selection, results, jobs, finished=False):
    aois = []
    for aoi in selection["ordered_aoi_ids"]:
        months = sorted(results.get(aoi, []), key=lambda m: m["month"])
        if not months:
            continue
        first = months[0]
        aois.append({"aoi_id": aoi, "official_split": "train", "split": "development",
                     "crs": first["crs"], "native_transform": first["native_transform"],
                     "bounds_lon_lat": first["bounds_lon_lat"], "months": months,
                     "usable_month_count": sum(m["status"] == "ready" for m in months)})
    value = {"schema_version": "sn7-pilot-v1", "updated_utc": utc_now(),
             "status": "complete" if finished else "partial", "paths_relative_to": "workspace_root",
             "source": {"bucket": "s3://spacenet-dataset/spacenet/SN7_buildings/",
                        "official_url": "https://spacenet.ai/sn7-challenge/",
                        "license": "CC-BY-SA-4.0", "imagery": "Planet Dove monthly RGB-alpha uint8 GeoTIFF",
                        "paper": "https://arxiv.org/abs/2102.04420"},
             "selection_path": (out / "selection.json").relative_to(ROOT).as_posix(),
             "selection": {"rule": selection["rule"], "ordered_aoi_ids": selection["ordered_aoi_ids"]},
             "mask_semantics": {"building": "uint8 1=building, 0=unlabelled/background; evaluate only where valid=1",
                                "valid": "uint8 1=RGB raster mask valid AND alpha>0 AND no official UDM polygon touches pixel",
                                "udm": "uint8 1=official unusable-data polygon touches pixel; 0=not UDM",
                                "warning": "Cloud-hidden buildings are unobservable, never training/evaluation background. Building IDs are AOI-local. No registration correction was introduced."},
             "planned_month_count": len(jobs), "completed_month_count": sum(len(x) for x in results.values()),
             "ready_month_count": sum(m["status"] == "ready" for x in results.values() for m in x),
             "source_object_bytes": sum(obj["bytes"] for j in jobs for obj in j["objects"].values()),
             "aois": aois}
    write_json(out / "manifest.partial.json", value)
    if finished:
        write_json(out / "manifest.json", value)
    return value


def validate_dataset(out):
    """Check the saved dataset against its original rasters and source records."""
    import rasterio
    path = out / "manifest.json"
    if not path.exists():
        path = out / "manifest.partial.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    errors, summaries = [], []
    actual_source_bytes = 0
    for aoi in data["aois"]:
        grids, source_hashes, dates = set(), [], []
        udm_months, missing_ids, ready_count = 0, 0, 0
        for month in aoi["months"]:
            context = f"{aoi['aoi_id']}/{month['month']}"
            dates.append(month["month"])
            grids.add(tuple(month["native_transform"]) + tuple(month["dimensions"]))
            for obj in month["source_objects"].values():
                source = ROOT / obj["path"]
                actual_source_bytes += source.stat().st_size
                if source.stat().st_size != obj["bytes"] or digest_file(source) != obj["sha256"]:
                    errors.append(context + ": corrupt source " + obj["path"])
            for key, expected in month["derived_sha256"].items():
                if digest_file(ROOT / month[key]) != expected:
                    errors.append(context + ": corrupt derivative " + key)
            rgba = np.load(ROOT / month["native_array_path"], allow_pickle=False)
            rgb = np.asarray(Image.open(ROOT / month["rgb_path"]))
            building = np.asarray(Image.open(ROOT / month["building_mask_path"]))
            valid = np.asarray(Image.open(ROOT / month["valid_mask_path"]))
            udm = np.asarray(Image.open(ROOT / month["udm_mask_path"]))
            with rasterio.open(ROOT / month["native_image_path"]) as source:
                original = np.moveaxis(source.read(), 0, -1)
                expected_valid = np.all(source.read_masks([1, 2, 3]) > 0, axis=0)
                expected_valid &= (original[:, :, 3] > 0) & (udm == 0)
            if not np.array_equal(rgba, original) or not np.array_equal(rgb, original[:, :, :3]):
                errors.append(context + ": prepared image differs from native bands")
            expected_shape = tuple(month["dimensions"][::-1])
            if rgb.shape != (*expected_shape, 3) or any(x.shape != expected_shape for x in [building, valid, udm]):
                errors.append(context + ": spatial shape mismatch")
            if any(not np.isin(x, [0, 1]).all() for x in [building, valid, udm]):
                errors.append(context + ": non-binary mask")
            if not np.array_equal(valid > 0, expected_valid):
                errors.append(context + ": invalid pixels incorrectly marked usable")
            if (valid[udm > 0] > 0).any():
                errors.append(context + ": UDM pixels present in valid mask")
            track = json.loads((ROOT / month["building_track_ids_path"]).read_text(encoding="utf-8"))
            ids = track["ids_in_feature_order"]
            if len(ids) != month["building_count"] or len(set(ids)) != len(ids):
                errors.append(context + ": inconsistent or duplicate tracking IDs")
            missing_ids += sum(x is None for x in ids)
            udm_months += int(udm.any())
            ready_count += int(valid.any())
            source_hashes.append(month["source_objects"]["image"]["sha256"])
        if dates != sorted(set(dates)):
            errors.append(aoi["aoi_id"] + ": repeated/nonchronological month IDs")
        if ready_count < 12:
            errors.append(aoi["aoi_id"] + ": fewer than 12 usable months")
        if missing_ids:
            errors.append(aoi["aoi_id"] + ": missing building tracking IDs")
        summaries.append({"aoi_id": aoi["aoi_id"], "month_count": len(dates),
                          "ready_month_count": ready_count, "udm_month_count": udm_months,
                          "min_valid_fraction": min(m["valid_fraction"] for m in aoi["months"]),
                          "native_grid_count": len(grids), "missing_track_id_count": missing_ids,
                          "distinct_native_image_hashes": len(set(source_hashes)), "month_ids": dates})
    report = {"checked_utc": utc_now(), "manifest_status": data["status"],
              "manifest_sha256": digest_file(path), "passed": not errors, "errors": errors,
              "observation_count": sum(a["month_count"] for a in summaries),
              "actual_source_bytes": actual_source_bytes, "aois": summaries,
              "checks": ["source/derivative SHA-256 and source lengths", "PNG and NPY equal original GeoTIFF bands",
                         "native spatial shape and mask binary values", "alpha/raster/UDM exclusion",
                         "chronological unique month IDs", "tracking ID presence/uniqueness", "at least 12 usable months/AOI"]}
    write_json(out / "validation.json", report)
    if errors:
        raise ValueError(json.dumps(report, ensure_ascii=False))
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aois", type=int, default=4)
    parser.add_argument("--months-per-aoi", type=int, default=0, help="0=all chronological official months")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-download-bytes", type=int, default=2_000_000_000)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.aois < 1 or args.months_per_aoi < 0 or args.workers < 1:
        parser.error("invalid AOI/month/worker count")
    out = DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    if args.validate_only:
        validate_dataset(out)
        return
    selection = freeze_selection(out)
    if args.aois > len(selection["ordered_aoi_ids"]):
        parser.error("requested more AOIs than official training set")
    jobs = month_jobs(out, selection, args.aois, args.months_per_aoi)
    total = sum(obj["bytes"] for job in jobs for obj in job["objects"].values())
    print(json.dumps({"selected": selection["ordered_aoi_ids"][:args.aois],
                      "planned_months": len(jobs), "download_bytes": total}), flush=True)
    if total > args.max_download_bytes:
        raise SystemExit("Planned source objects exceed explicit byte cap; no image downloaded")
    if args.plan_only:
        return
    try:
        import rasterio
    except ImportError as error:
        raise SystemExit("Install rasterio dependencies into data/spacenet7_pilot/.deps; see docs/SN7_DATA_PROVENANCE.md") from error
    results = {}
    # Recover completed months, checking every source file and recorded derivative.
    pending = []
    for job in jobs:
        record_path = out / "prepared" / job["aoi_id"] / job["month"] / "record.json"
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            valid = record.get("preparation_version") == PREPARATION_VERSION
            valid &= all((ROOT / o["path"]).exists() and digest_file(ROOT / o["path"]) == o["sha256"]
                        for o in record["source_objects"].values())
            valid &= all((ROOT / record[k]).exists() and digest_file(ROOT / record[k]) == checksum
                         for k, checksum in record["derived_sha256"].items())
            if valid:
                results.setdefault(job["aoi_id"], []).append(record)
                continue
        pending.append(job)
    manifest(out, selection, results, jobs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(prepare_month, out, job) for job in pending]
        for future in concurrent.futures.as_completed(futures):
            aoi, record = future.result()
            results.setdefault(aoi, []).append(record)
            manifest(out, selection, results, jobs)
            print(f"ready {aoi} {record['month']} valid={record['valid_fraction']:.4f} buildings={record['building_count']}", flush=True)
    value = manifest(out, selection, results, jobs, finished=True)
    write_json(out / "environment.json", {"python": sys.version, "rasterio": rasterio.__version__,
               "gdal": rasterio.__gdal_version__, "numpy": np.__version__, "requests": requests.__version__,
               "Pillow": pillow_version,
               "isolated_dependencies": {k: package_version(k) for k in ["rasterio", "affine", "attrs", "click", "click-plugins", "cligj", "certifi", "pyparsing"]},
               "argv": sys.argv, "completed_utc": utc_now()})
    print(f"complete: {value['ready_month_count']} months, {len(value['aois'])} AOIs", flush=True)


if __name__ == "__main__":
    main()
