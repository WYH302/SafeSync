"""Check compiled paper provenance and PDF mechanics; no scientific re-evaluation.

Requires pypdf. Compile logs are optional in the compact review archive; use
--require-logs for the local final build. Author placeholders are intentional.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_hashes(base, entries):
    for relative, expected in entries.items():
        path = base / relative
        if not path.is_file() or sha(path) != expected:
            raise ValueError("Missing or changed bound input: " + str(path))
    return len(entries)


def embedded_fonts(reader):
    fonts = {}
    for page in reader.pages:
        resources = page.get("/Resources", {}).get_object()
        for ref in resources.get("/Font", {}).get_object().values():
            font = ref.get_object()
            descendants = font.get("/DescendantFonts", [font])
            for descendant in descendants:
                item = descendant.get_object()
                name = str(item.get("/BaseFont", "unnamed"))
                descriptor = item.get("/FontDescriptor")
                embedded = item.get("/Subtype") == "/Type3"
                if descriptor:
                    embedded |= any(k in descriptor.get_object() for k in ("/FontFile", "/FontFile2", "/FontFile3"))
                fonts[name] = bool(embedded)
    if not fonts or not all(fonts.values()):
        raise ValueError("Missing or unembedded PDF fonts: " + repr(fonts))
    return fonts


def check_document(base, expected_pages, require_logs):
    manifest_path = base / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if manifest["status"] != "success" or manifest["returncode"] != 0:
        raise ValueError("Unsuccessful compile: " + str(base))
    count = verify_hashes(base, manifest["source_sha256"])
    pdf = base / manifest["output"]
    if sha(pdf) != manifest["pdf_sha256"] or pdf.stat().st_size != manifest["pdf_bytes"]:
        raise ValueError("PDF differs from successful build: " + str(pdf))
    reader = PdfReader(pdf)
    if len(reader.pages) != expected_pages:
        raise ValueError("Unexpected page count: " + str(pdf))
    page_text = [page.extract_text() or "" for page in reader.pages]
    if any(not text.strip() or "\ufffd" in text for text in page_text):
        raise ValueError("Empty page text or replacement glyph: " + str(pdf))
    logs = []
    for name in ("compiled_main.log", "build.log"):
        path = base / name
        if not path.exists():
            if require_logs:
                raise ValueError("Required compile log missing: " + str(path))
            continue
        text = path.read_text("utf-8", errors="replace")
        # latexmk stdout contains expected unresolved references from early passes.
        # Only the final TeX pass is authoritative for resolved references/layout.
        errors = re.findall(r"(?im)^.*(?:Overfull|undefined references|undefined citations|Citation .* undefined|Reference .* undefined|multiply[- ]defined|LaTeX Error|^! ).*$", text) if name == "compiled_main.log" else []
        if errors:
            raise ValueError("Final compile diagnostics: " + repr(errors))
        logs.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
                     "diagnostics_scope": "final TeX pass" if name == "compiled_main.log" else "all-pass transcript; provenance only",
                     "blocking_diagnostics": []})
    return {"pdf": pdf.relative_to(ROOT).as_posix(), "sha256": sha(pdf), "pages": len(reader.pages),
            "bytes": pdf.stat().st_size, "bound_sources_verified": count,
            "build_manifest_sha256": sha(manifest_path), "fonts_embedded": embedded_fonts(reader),
            "compile_logs": logs, "page_text_characters": [len(t) for t in page_text]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-logs", action="store_true")
    parser.add_argument("--main-pages", type=int, default=12,
                        help="Expected pages in the current main manuscript")
    parser.add_argument("--supplement-pages", type=int, default=18,
                        help="Expected pages in the current supplementary manuscript")
    parser.add_argument("--output", type=Path, default=ROOT / "paper/technical_checks.json")
    args = parser.parse_args()
    documents = [check_document(ROOT / "paper", args.main_pages, args.require_logs),
                 check_document(ROOT / "paper/supplement", args.supplement_pages, args.require_logs)]
    assets = []
    for relative, generator in (
        ("paper/generated/oof_asset_manifest.json", "scripts/build_oof_paper_assets.py"),
        ("paper/supplement/generated/asset_manifest.json", "scripts/build_supplement_assets.py"),
        ("paper/supplement/generated/task_context_manifest.json", "scripts/build_task_context_assets.py"),
        ("paper/supplement/generated/task_followup_manifest.json", "scripts/build_task_evidence_followup_assets.py"),
        ("paper/supplement/generated/residual_confirmation_manifest.json", "scripts/build_residual_confirmation_assets.py"),
        ("paper/supplement/generated/tcm_second_manifest.json", "scripts/build_tcm_assets.py"),
        ("paper/generated/decision_assets_manifest.json", "scripts/build_decision_figures_and_cost_table.py"),
    ):
        path = ROOT / relative
        manifest = json.loads(path.read_text("utf-8"))
        if manifest["status"] != "complete" or sha(ROOT / generator) != manifest["generator_sha256"]:
            raise ValueError("Changed or incomplete asset generator: " + relative)
        assets.append({"manifest": relative, "sha256": sha(path),
                       "sources": verify_hashes(ROOT, manifest["source_sha256"]),
                       "outputs": verify_hashes(ROOT, manifest["output_sha256"])})
    source = (ROOT / "paper/main.tex").read_text("utf-8")
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", source, re.S).group(1)
    word_count = len(abstract.split())
    if not 150 <= word_count <= 250:
        raise ValueError("Abstract outside the selected 150--250 word range")
    report = {"status": "PASS", "checked_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "compiled artifact provenance, embedded fonts, page text, final build diagnostics and generated-asset hashes",
              "checker_sha256": sha(Path(__file__)), "documents": documents, "assets": assets,
              "abstract_words_whitespace": word_count,
              "intentional_pending": ["real author details", "human author review and declarations", "external submission"],
              "limits": "This artifact check does not establish scientific novelty, reviewer acceptance, or completion of human author review."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(json.dumps({"status": report["status"], "pages": [d["pages"] for d in documents],
                      "abstract_words": word_count, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
