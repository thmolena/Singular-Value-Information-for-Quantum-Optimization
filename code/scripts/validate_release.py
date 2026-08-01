#!/usr/bin/env python3
"""Validate the negative-audit release without rerunning the experiment."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
CODE = REPOSITORY / "code"
RESULTS = CODE / "results"
PACKAGE_RESULTS = CODE / "src" / "hqml_spectral_qaoa" / "results"
REQUIRED_ROOT = {
    ".gitignore",
    "LICENSE",
    "README.md",
    "index.html",
    "main.tex",
    "main.pdf",
    "code",
}
REQUIRED_CODE = {
    "CITATION.cff",
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "configs",
    "data",
    "evidence_manifest.json",
    "manuscript_assets",
    "pyproject.toml",
    "reproduce.py",
    "requirements.txt",
    "results",
    "scripts",
    "src",
    "tests",
}
FIGURES = {
    "collision_audit",
    "formula_validation_runtime",
    "grouped_query_curves",
    "matched_kernel_runtime",
    "residual_noncertification",
    "split_sensitivity",
}
TABLES = {
    "collision_rows.tex",
    "formula_runtime_rows.tex",
    "grouped_fold_rows.tex",
    "kernel_runtime_rows.tex",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise SystemExit(message)


def main() -> None:
    root_names = {path.name for path in REPOSITORY.iterdir() if path.name != ".git"}
    if root_names != REQUIRED_ROOT:
        fail(
            json.dumps(
                {
                    "missing_root": sorted(REQUIRED_ROOT - root_names),
                    "extra_root": sorted(root_names - REQUIRED_ROOT),
                },
                indent=2,
            )
        )
    code_names = {path.name for path in CODE.iterdir()}
    missing_code = sorted(REQUIRED_CODE - code_names)
    if missing_code:
        fail(f"missing code artifacts: {missing_code}")

    manuscript = (REPOSITORY / "main.tex").read_text(encoding="utf-8")
    forbidden = (
        "65,000",
        "G81",
        "claim ladder",
        "residual-gated spectral prior",
        "RGS-Q",
        "quantum advantage is established",
    )
    found = [token for token in forbidden if token.lower() in manuscript.lower()]
    if found:
        fail(f"stale or forbidden manuscript claims: {found}")
    table_count = len(re.findall(r"\\begin\{table\*?\}", manuscript))
    figure_count = len(re.findall(r"\\begin\{figure\*?\}", manuscript))
    theorem_like = len(
        re.findall(r"\\begin\{(?:theorem|proposition|corollary|lemma)\}", manuscript)
    )
    if (table_count, figure_count, theorem_like) != (1, 3, 2):
        fail(
            f"manuscript contract mismatch: tables={table_count}, "
            f"figures={figure_count}, theorem_like={theorem_like}"
        )

    pdf_names = {path.stem for path in (RESULTS / "figures").glob("*.pdf")}
    png_names = {path.stem for path in (RESULTS / "figures").glob("*.png")}
    if pdf_names != FIGURES or png_names != FIGURES:
        fail("figure artifact set mismatch")
    table_names = {path.name for path in (RESULTS / "tables").glob("*.tex")}
    if table_names != TABLES:
        fail("table artifact set mismatch")

    locked = json.loads((RESULTS / "results.json").read_text(encoding="utf-8"))
    if locked["isomorphism_class_count"] != 16:
        fail("isomorphism class count is not locked to 16")
    if locked["grouped_fold_loads"] != [12, 12, 12, 12]:
        fail("grouped folds are not balanced")
    if locked["spectral_residual_passes"] != 2:
        fail("residual audit no longer records two passes")
    if locked["formula_validation"]["max_absolute_error"] > 1e-12:
        fail("exact formula validation exceeds tolerance")
    if locked["formula_validation"]["target_quantum_objective_queries"] != 0:
        fail("exact formula query count is not zero")

    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        path = RESULTS / record["path"]
        if not path.is_file():
            fail(f"manifest file missing: {record['path']}")
        if sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            fail(f"manifest mismatch: {record['path']}")
    for relative in ("results.json", "results.tex", "manifest.json"):
        if sha256(RESULTS / relative) != sha256(PACKAGE_RESULTS / relative):
            fail(f"installed-package mirror mismatch: {relative}")

    stale = []
    for path in CODE.rglob("*"):
        if path.name in {"__pycache__", ".pytest_cache"} or path.name.endswith(".egg-info"):
            stale.append(path.relative_to(REPOSITORY).as_posix())
    if stale:
        fail(f"generated cache directories remain: {stale}")
    if (RESULTS / "tables" / "results_results.tex").exists():
        fail("recursive table artifact remains")

    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo is None:
        fail("pdfinfo is required for final PDF validation")
    info = subprocess.run(
        [pdfinfo, str(REPOSITORY / "main.pdf")],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    match = re.search(r"^Pages:\s+(\d+)$", info, re.MULTILINE)
    if match is None:
        fail("could not read main.pdf page count")
    pages = int(match.group(1))
    if not 3 <= pages <= 12:
        fail(f"unexpected manuscript length: {pages} pages")

    newest_source = max(
        [REPOSITORY / "main.tex"]
        + list((RESULTS / "figures").glob("*.pdf"))
        + list((RESULTS / "tables").glob("*.tex")),
        key=lambda path: path.stat().st_mtime,
    )
    if (REPOSITORY / "main.pdf").stat().st_mtime < newest_source.stat().st_mtime:
        fail("main.pdf is older than a manuscript source artifact")

    print(
        json.dumps(
            {
                "status": "PASS",
                "pages": pages,
                "tables": table_count,
                "figures": figure_count,
                "theorem_like": theorem_like,
                "isomorphism_classes": locked["isomorphism_class_count"],
                "residual_gate_passes": locked["spectral_residual_passes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
