"""Generate supplementary tables from saved counts and profiles (no inference)."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/supplement"
GEN = OUT / "generated"
INPUTS = {}
OUTPUTS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(relative):
    path = ROOT / relative
    INPUTS[relative] = sha(path)
    return path


def read_json(relative):
    return json.loads(source(relative).read_text(encoding="utf-8"))


def read_csv(relative):
    with source(relative).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save(name, value):
    path = GEN / name
    path.write_text(value, encoding="utf-8")
    OUTPUTS[path.relative_to(ROOT).as_posix()] = sha(path)


def tex(value):
    return str(value).replace("_", r"\_").replace("%", r"\%")


def table(name, caption, headers, rows, note, alignment=None):
    alignment = alignment or "l" + "r" * (len(headers) - 1)
    content = [
        r"\begin{table*}[t]", r"\centering\footnotesize",
        r"\caption{" + caption + "}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{" + alignment + "}", r"\toprule",
        " & ".join(headers) + r" \\", r"\midrule",
    ]
    content += [" & ".join(map(str, row)) + r" \\" for row in rows]
    content += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
                r"\begin{minipage}{0.97\textwidth}\footnotesize " + note +
                r"\end{minipage}", r"\end{table*}", ""]
    save(name + ".tex", "\n".join(content))


def cohort():
    folds = read_json("resources/pretrained_ground_crossfit/fold_selection.json")
    fold = {a["aoi_id"]: a["author_validation_fold"] for a in folds["targets"]}
    rows = []
    for manifest in ("development", "confirmation"):
        data = read_json(f"data/spacenet7_pilot/{manifest}_manifest_v1.json")
        for i, a in enumerate(data["aois"]):
            role = ("Cal." if i < 2 else "Dev.") if manifest == "development" else "Conf."
            months = a["months"]
            w, h = months[0]["dimensions"]
            rows.append([role, tex(a["aoi_id"]), len(months), len(months)-1,
                         f"{w}" + r"$\times$" + str(h), fold.get(a["aoi_id"], "--")])
    table("cohort", "Fixed areas, observed months, and author validation folds.",
          ["Role", "Official AOI identifier", "Images", "Updates", "Native grid", "OOF fold"],
          rows, "OOF folds identify validation-best public checkpoints, not an untouched test. "
          "Development evaluation areas do not require an OOF assignment in this sensitivity study. "
          "Padding to 64-pixel blocks is excluded from offline quality support.",
          "llrrrr")


def quality_tables():
    data = read_csv("results/layered_quality/confirmation_v1/per_seed_aoi.csv")
    groups = defaultdict(list)
    for row in data:
        if row["method"].startswith(("gaussian_", "random_")):
            groups[(row["method"], int(row["budget"]))].append(row)
    rows = []
    for (method, budget), values in sorted(groups.items()):
        if len(values) != 8:
            raise ValueError("Seed table needs all eight AOIs")
        mean = lambda key: statistics.mean(float(v[key]) for v in values)
        family, seed = method.rsplit("_", 1)
        rows.append([family.title(), seed, budget//1024,
                     f"{mean('update_total_bytes_mean')/1024:.2f}",
                     *[f"{100*mean('new_track_psnr'+str(t)+'_coverage'):.2f}" for t in (25, 30, 35)],
                     f"{mean('valid_psnr_db'):.2f}"])
    table("seed_sensitivity", "Every preselected Gaussian and random seed on the eight confirmation areas.",
          ["Policy", "Seed", "Cap KiB", "Actual KiB", r"$U_{25}$ \%", r"$U_{30}$ \%", r"$U_{35}$ \%", "PSNR dB"],
          rows, "All quantities are equal-area means after pooling monthly counts within each area. "
          "Actual bytes include update downlink and the 60-byte ACK; initialization is reported separately. "
          "Only Gaussian 20261029 and random 1729 have task inference in the original main assay.")

    macro = read_csv("results/layered_quality/confirmation_v1/macro_single_seed.csv")
    labels = {"gaussian_20261029": "Gaussian", "dct_bound": "DCT",
              "full_cache_mse": "Full history (25 dB)", "full_webp": "WebP"}
    rows = []
    for row in sorted(macro, key=lambda r: (int(r["budget"]), r["method"])):
        if row["method"] not in labels:
            continue
        values = [float(row[k]) / 1024 for k in
                  ("down_protocol_metadata_bytes_mean", "down_coarse_image_bytes_mean",
                   "down_image_payload_bytes_mean", "update_total_bytes_mean", "sequence_bytes_per_update")]
        if abs(sum(values[:3]) + 60/1024 - values[3]) > 1e-8:
            raise ValueError("Byte components do not sum to actual cost")
        rows.append([labels[row["method"]], int(row["budget"])//1024,
                     *[f"{v:.3f}" for v in values]])
    table("byte_components", "Actual communication components and bootstrap amortization.",
          ["Method", "Cap", "Metadata", "Coarse RGB", "Detail/full RGB", "Update + ACK", "With bootstrap"],
          rows, "All byte quantities are KiB per update, averaged equally across areas. "
          "Cap limits downlink only; ACK is 60 bytes. Metadata includes indices, status, header and checksum. "
          "The last column divides each area's complete sequence cost by its actual number of updates, "
          "then averages areas. It must not be confused with the update-only x-axis of the main task plot.")


def cpu_tables():
    methods = {
        "gaussian_20261029": "Gaussian",
        "dct_bound": "DCT",
        "full_cache_mse": "Full history (25 dB)",
        "full_jpeg420": "JPEG 4:2:0",
        "full_jpeg2000": "JPEG2000",
        "full_webp": "WebP",
    }
    single = read_json("results/sender_profile/layered_v1/selected_codec_calls.json")
    call = {x["method"]: statistics.mean(r["encode_call_ms"] for r in x["records"])
            for x in single["methods"]}
    rows = []
    profile = {}
    for method, label in methods.items():
        data = read_json(f"results/sender_profile/layered_v1/{method}.json")
        profile[method] = data
        s = data["statistics"]
        rows.append([label, 12,
                     f"{s['input_to_packet_ms']['mean']:.1f}",
                     f"{s['input_to_packet_ms']['p95']:.1f}",
                     f"{s['receiver_ms']['mean']:.1f}",
                     f"{call[method]:.1f}" if method in call else "--"])
    lic = read_json("results/sender_profile/lic_cpu_v1/summary.json")
    if lic["status"] != "complete" or lic["frames_completed"] != 2 or lic["CUDA_initialized"]:
        raise ValueError("Incomplete or non-CPU learned-codec profile")
    search_mean = statistics.mean(r["full_search_excluding_checkpoint_IO_ms"] for r in lic["records"])
    same_two = statistics.mean(r["input_to_packet_ms"] for r in profile["gaussian_20261029"]["records"][:2])
    table("cpu_cost", "Measured prototype costs on a Ryzen 7 5700X with four CPU threads (ms).",
          ["Method", "Frames", "Input--packet", "p95", "Receiver", "One encode"],
          rows, "The twelve frames are fixed consecutive updates of original calibration AOI 0, at the 32-KiB cap. "
          "Input--packet includes PNG loading, preprocessing, route/parameter search, local decoding when needed, "
          "and serialization. Receiver is measured separately. One encode excludes search, preprocessing, MSE "
          "comparison, and protocol; it is not a substitute for the full input--packet time.")
    selected_rows = []
    for cap in (32768, 65536, 131072):
        selections = [next(x for x in r["selected_single_calls"] if x["cap"] == cap) for r in lic["records"]]
        selected_rows.append([cap//1024,
            f"{statistics.mean(x['selected_preprocess_ms'] for x in selections):.1f}",
            f"{statistics.mean(x['single_compress_ms_including_local_RPC'] for x in selections):.1f}",
            f"{statistics.mean(x['single_decompress_ms_including_local_RPC_and_native_restoration'] for x in selections):.1f}"])
    table("lic_cpu", "CPU-only hyperprior on the same first two calibration frames (ms).",
          ["Cap KiB", "Selected preprocessing", "One compress", "One decode/restore"],
          selected_rows,
          f"The complete 40-candidate search averages {search_mean/1000:.2f} s, excluding artifact checkpoint I/O; "
          f"the Gaussian input--packet mean on these same two frames is {same_two:.1f} ms. "
          "One-call figures use warm selected settings and include the local official-entropy RPC. "
          "They exclude the decision search. The two paths use Python 3.11/NumPy 2.4.6 and "
          "Python 3.12/NumPy 2.4.3 respectively; no flight-platform or optimized-library claim is made.")
    # Keep one input point in the supplement; all underlying raw timings stay in the package.
    with (GEN / "cpu_cost.tex").open("a", encoding="utf-8") as stream:
        stream.write("\n" + r"\input{generated/lic_cpu}" + "\n")
    OUTPUTS[(GEN/"cpu_cost.tex").relative_to(ROOT).as_posix()] = sha(GEN/"cpu_cost.tex")
    save("cost_numbers.json", json.dumps({
        "lic_cpu_search_mean_ms": search_mean,
        "gaussian_same_two_input_packet_ms": same_two,
        "lic_windows_peak_rss_bytes": lic["Windows_process_RSS_monitored_peak_bytes"],
        "lic_cpu_frames": 2, "profile_frames": 12,
    }, indent=2))


def compressed_archive():
    summary = read_json("results/sender_profile/compressed_archive_v1/summary.json")
    if (summary["status"] != "complete" or summary["update_snapshots"] != 534
            or not summary["all_decoded_archives_exact"]
            or not summary["all_update_archive_hashes_match_original_task_receipts"]):
        raise ValueError("Compressed historical-archive measurement incomplete")
    values = read_csv("results/sender_profile/compressed_archive_v1/per_cap.csv")
    rows = []
    for r in values:
        png = float(r["sum_AOI_mean_PNG_bytes"])
        compact = int(r["Gaussian_eight_summaries_plus_one_matrix_bytes"])
        rows.append([int(r["budget"])//1024, f"{png/2**20:.3f}",
                     f"{int(r['sum_raw_archive_bytes'])/2**20:.3f}",
                     f"{compact/2**20:.3f}", f"{png/compact:.3f}"])
    table("compressed_archive", "Lossless storage of the exact full-history gate30 archives.",
          ["Cap KiB", "PNG archives MiB", "Raw RGB MiB", "Summaries + matrix MiB", "PNG / compact"],
          rows, "PNG quantities sum the eight AOIs' individual temporal mean archive lengths over actual updates. "
          "They are not simultaneous peaks. Every archive is encoded using fixed PNG compression level 6 "
          "and decoded byte-for-byte before its size is accepted. No rendering, routing or task result changes. "
          "Protocol metadata, current inputs and temporary buffers are additional; the full-history prototype's "
          "existing 98,624 summary bytes across eight areas are listed separately in the raw report.")


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    cohort()
    quality_tables()
    cpu_tables()
    compressed_archive()
    for origin in [
        "paper/figures/task_uncertainty.pdf",
        "results/layered_quality/confirmation_v1/figures/rate_quality_amortized.pdf",
        "results/layered_quality/confirmation_v1/figures/per_aoi_difference.pdf",
    ]:
        path = source(origin)
        target = OUT / "figures" / path.name
        shutil.copyfile(path, target)
        OUTPUTS[target.relative_to(ROOT).as_posix()] = sha(target)
    for relative, digest in INPUTS.items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError("Source changed during table generation: " + relative)
    manifest = {"status": "complete", "scope": "cohort, quality seeds, bytes and CPU profile assets",
                "source_sha256": INPUTS, "output_sha256": OUTPUTS,
                "generator_sha256": sha(__file__)}
    (GEN / "asset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"inputs": len(INPUTS), "outputs": len(OUTPUTS), "status": "complete"}))


if __name__ == "__main__":
    main()
