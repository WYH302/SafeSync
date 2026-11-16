"""Freeze the residual method, sampling and joint tests before new pixels."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results/task_evidence/residual_confirmation_v1"
DATA = ROOT/"data/second_residual_test_v1"
RESERVATION = ROOT/"data/second_residual_reservation_v1.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    reservation = json.loads(RESERVATION.read_text("utf-8"))
    old = json.loads((ROOT/"results/task_evidence/fresh_context_v1/method_lock.json").read_text("utf-8"))
    names = ("scripts/freeze_residual_confirmation.py","scripts/reserve_second_residual_test.py",
        "scripts/fetch_residual_confirmation.py","scripts/fetch_task_context_confirmation.py",
        "scripts/prepare_residual_confirmation.py","scripts/run_residual_confirmation.py",
        "scripts/evaluate_residual_confirmation.py","scripts/analyze_residual_confirmation.py",
        "scripts/run_residual_evidence_development.py","scripts/run_task_context_confirmation.py",
        "scripts/run_task_evidence_selector.py","scripts/run_task_evidence_development.py",
        "scripts/run_locked_second_test.py","scripts/run_second_feature_mix.py","scripts/run_learned_codec.py",
        "scripts/evaluate_task_context_confirmation.py","scripts/analyze_task_context_confirmation.py",
        "scripts/fetch_second_scaled.py","scripts/fetch_locked_second_test.py","safesync/residual_evidence.py",
        "safesync/task_evidence_wire.py","safesync/learned_codec.py","safesync/entropy_bridge.py",
        "safesync/features.py","safesync/scoring.py","safesync/data.py","safesync/pixel_wire.py","safesync/full_image_codecs.py",
        "resources/learned_codec/entropy_server.py")
    spec = dict(schema="learned-base-residual-confirmation-v1",pair_ids=[p["id"] for p in reservation["reserved_pairs"]],
        reservation_sha256=sha(RESERVATION),source_sha256={name:sha(ROOT/name) for name in names},
        galleries=old["galleries"],prototypes=old["prototypes"],model_weight_sha256=old["model_weight_sha256"],
        later_ground_head_sha256=old["later_ground_head_sha256"],public_hyperprior_registry_sha256=old["public_hyperprior_registry_sha256"],
        helper_spec_sha256=sha(ROOT/"results/second_pixels/fixed_v1/run_spec.json"),
        development_analysis_sha256=sha(ROOT/"results/task_evidence/residual_analysis_v1/analysis.json"),
        smoke_verification_sha256=sha(ROOT/"results/task_evidence/residual_confirmation_smoke_v1/verification.json"),
        caps=[4096,8192,16384,32768],heldouts=[5,3,4],seeds=[1729,2718,31415],
        task="Exact known-prototype current16x16 array, shorter reconstructed-reference XOR/absolute zlib; all codecs get this same service",
        backbone=old["encoder_config"],novelty_k=10,mixture=[.75,.25],novelty="Current tenth-known-neighbor distance, no temporal increment",
        base_fraction=.5,atlas="Absolute/residual native WebP70, fixed64-pixel cells, signed residual scale2",
        selector="rank-weighted max(1 - native cell RGB MSE/0.001,0), compare against the full-cap hyperprior incumbent, use incumbent on ties",
        hyperprior="Public bmshj2018-hyperprior MSE,8 qualities x5 scales; actual rANS candidates, minimum sender MSE",
        variants=["hyperprior","absolute_change","absolute_current","residual_change","residual_current","residual_random (3 seeds)","full_webp"],
        controls=["labels_only","old_only","full_current"],expected_records=29952,
        optimization="Pure WebP-atlas byte cache within one source pair, keyed by complete source image bytes/shape, ordered indices, codec, quality, block. Encoding decisions and output bytes retain the development definition.",
        primary=dict(heldout=5,budget=8192,method="residual_current",
            coverage_comparators=["residual_change","absolute_current","hyperprior","full_webp"],
            later_comparators=["hyperprior","full_webp"],later_class_f1_margin=-.02,
            decision="All4 coverage95% lower bounds>0, both later-use95% lower bounds>-0.02, and current task states identical to full-current controls"),
        secondary="All3 classes x4 caps retained; descriptive99.583333% intervals percontrast family",
        scope="New256 pair-ID-disjoint official-test inputs. Method/criteria freeze before any of their pixels. The earlier32-pair result is exploratory.",
        exclusions=reservation["excluded_pair_ids"])
    spec = json.loads(json.dumps(spec))
    lock_path = OUT/"method_lock.json"
    if lock_path.exists():
        saved = json.loads(lock_path.read_text("utf-8"))
        if {k:v for k,v in saved.items() if k != "locked_utc"} != spec:
            raise ValueError("Refusing to change the new residual confirmation")
    else:
        if DATA.exists() and any(DATA.rglob("*.png")):
            raise ValueError("Attempted residual confirmation lock after pixel access")
        OUT.mkdir(parents=True,exist_ok=True)
        spec["locked_utc"] = datetime.now(timezone.utc).isoformat()
        lock_path.write_text(json.dumps(spec,indent=2)+"\n","utf-8")
        for name in names:
            path = OUT/"source_snapshot"/name.replace("/","__")
            path.parent.mkdir(exist_ok=True)
            path.write_bytes((ROOT/name).read_bytes())
    print(json.dumps(dict(status="locked_before_new_pixels",pairs=256,expected=29952,lock_sha256=sha(lock_path))),flush=True)


if __name__ == "__main__":
    main()
