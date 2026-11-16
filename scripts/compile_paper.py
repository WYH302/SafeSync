"""Build the local manuscript through an ASCII staging path on Windows.

The installed Perl/latexmk cannot resolve this project's Unicode working path.
Sources remain canonical in paper/; the stage is a reproducible local copy.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FALLBACK_LATEXMK = Path(r"D:\texlive\2026\bin\windows\latexmk.exe")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT / "paper")
    parser.add_argument("--output", default="manuscript.pdf",
                        help="PDF name relative to the project, or an absolute path")
    parser.add_argument("--latexmk", type=Path,
                        help="Existing latexmk executable; otherwise PATH, then local TeX Live")
    args = parser.parse_args()
    paper = args.project.resolve()
    latexmk = args.latexmk or Path(shutil.which("latexmk") or FALLBACK_LATEXMK)
    output = (paper / args.output).resolve()
    if not latexmk.is_file():
        raise FileNotFoundError("Install TeX Live/latexmk or specify --latexmk; no automatic install is performed")
    if not (paper / "main.tex").is_file():
        raise FileNotFoundError(f"Missing main.tex in project: {paper}")
    stage = Path(tempfile.mkdtemp(prefix="safesync-paper-"))
    files = {}
    excluded = {"build", "builds", "render", "supplement", "bibliography_sources"}
    for path in paper.rglob("*"):
        rel = path.relative_to(paper)
        if (path.is_file() and path.suffix in {".tex", ".bib", ".cls", ".sty", ".bst", ".png", ".pdf"}
                and not any(part in excluded for part in rel.parts[:-1])
                and path.resolve() != output
                and path.name not in {"manuscript.pdf", "supplementary.pdf"}
                and not path.name.startswith("task_table_preview")):
            target = stage / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            files[rel.as_posix()] = sha(path)
    env = os.environ.copy()
    for key in ("LC_ALL", "LANG", "LC_CTYPE", "LC_NUMERIC", "LC_COLLATE"):
        env.pop(key, None)
    env["PATH"] = str(latexmk.parent) + os.pathsep + env.get("PATH", "")
    command = [str(latexmk), "-norc", "-pdf", "-interaction=nonstopmode",
               "-halt-on-error", "-outdir=build", "-no-shell-escape", "main.tex"]
    completed = subprocess.run(command, cwd=stage, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=360, check=False)
    log = completed.stdout.decode("utf-8", errors="replace")
    (paper / "build.log").write_text(log, encoding="utf-8")
    manifest = dict(status="failed", stage=str(stage), command=command,
                    source_sha256=files, returncode=completed.returncode,
                    built_utc=datetime.now(timezone.utc).isoformat(), output=output.name)
    pdf = stage / "build/main.pdf"
    if completed.returncode == 0 and pdf.is_file():
        for rel, expected in files.items():
            if sha(paper / rel) != expected:
                raise RuntimeError("Paper source changed during build")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        shutil.copyfile(pdf, temporary)
        temporary.replace(output)
        manifest.update(status="success", pdf_sha256=sha(pdf), pdf_bytes=pdf.stat().st_size)
        for suffix in ("log", "bbl"):
            extra = stage / f"build/main.{suffix}"
            if extra.exists():
                shutil.copyfile(extra, paper / f"compiled_main.{suffix}")
    (paper / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    print("\n".join(log.splitlines()[-35:]), flush=True)
    if manifest["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
