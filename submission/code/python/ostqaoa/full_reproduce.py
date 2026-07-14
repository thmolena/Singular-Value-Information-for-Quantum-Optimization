"""End-to-end regeneration of *every* OST-QAOA manuscript display item.

The manuscript (``submission/main.tex``) mirrors the section/figure/table
structure of Hashimoto et al., "Spectral Truncation Kernels" (JMLR;
arXiv:2405.17823): **seven main-body figures and two main-body tables.** Those
display items are produced by three cooperating drivers that all consume the
same deterministic result CSVs:

1. ``ostqaoa.paper_artifacts`` -- the matched-budget benchmark, truncation
   sweep, ablation, and per-family analysis. Emits ``figures/fig01..fig04`` and
   ``tables/table01_headline.tex`` (plus supporting CSVs).
2. ``make_jmlr_mirror_artifacts.py`` -- the four ``fig_jmlr_*`` figures and the
   conceptual ``tables/table_prior_summary.tex``, all re-plotted from the CSVs
   written in step 1 (and the multi-seed CSV from step 3).
3. ``multiseed_robustness.py`` -- the ten-seed robustness study feeding
   ``fig_jmlr_robustness`` and ``tables/table06_multiseed.tex``.
4. ``trace_evaluations.py`` (legacy, pure-NumPy) -- the per-query trace
   ``tables/trace_all_evaluations.csv`` feeding ``fig_jmlr_query_gap``.

This module chains them in dependency order so a single console command,
``ostqaoa-reproduce``, rebuilds all seven figures and both tables directly into
the directories ``submission/main.tex`` reads from (``code/figures`` and
``code/tables``). Every artifact is a deterministic function of the global seed.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

GLOBAL_SEED = 260424803


def _code_root() -> Path:
    """Return ``submission/code`` -- the artifact root the manuscript reads."""
    # python/ostqaoa/full_reproduce.py -> submission/code
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ostqaoa-reproduce",
        description=(
            "Deterministically regenerate all seven figures and both tables of "
            "the OST-QAOA manuscript (global seed 260424803) into the code/ "
            "figures and tables directories the manuscript reads."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory holding figures/, tables/, results/. Defaults to the "
            "installed artifact's submission/code directory so the manuscript "
            "figures/tables are updated in place."
        ),
    )
    parser.add_argument("--depth", type=int, default=3, help="QAOA depth p (operator dimension 2p).")
    parser.add_argument("--budget", type=int, default=24, help="Matched objective-query budget.")
    parser.add_argument("--rank", type=int, default=4, help="Operating-point spectral truncation parameter n.")
    parser.add_argument("--commutator-weight", type=float, default=4.0, help="Noncommutative commutator interaction weight.")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED, help="Global deterministic seed.")
    parser.add_argument(
        "--seeds",
        type=str,
        default="260424803,7,101,2024,31337,555,99991,12345,424242,8675309",
        help="Comma-separated seed list for the ten-seed robustness study.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast reduced-size pass for CI/package smoke checks (fewer seeds, lower depth).",
    )
    parser.add_argument(
        "--skip-multiseed",
        action="store_true",
        help="Skip the ten-seed robustness study (the slowest stage, ~minutes).",
    )
    parser.add_argument(
        "--skip-trace",
        action="store_true",
        help="Skip the legacy per-query trace regeneration (fig_jmlr_query_gap input).",
    )
    return parser


def _run_paper_artifacts(args: argparse.Namespace, code_root: Path) -> None:
    from .paper_artifacts import main as regenerate

    forwarded = [
        "--output-dir", str(code_root),
        "--depth", str(args.depth),
        "--budget", str(args.budget),
        "--rank", str(args.rank),
        "--commutator-weight", str(args.commutator_weight),
        "--seed", str(args.seed),
    ]
    if args.quick:
        forwarded.append("--quick")
    print(">> [1/4] paper_artifacts: benchmark, truncation sweep, ablation, per-family")
    regenerate(forwarded)


def _run_multiseed(args: argparse.Namespace, script_root: Path) -> None:
    seeds = [s for s in args.seeds.split(",") if s]
    if args.quick:
        seeds = seeds[:3]
    code_root = script_root
    script = code_root / "multiseed_robustness.py"
    if not script.is_file():
        print(f"!! multiseed_robustness.py not found at {script}; skipping")
        return
    print(f">> [3/4] multiseed_robustness: {len(seeds)}-seed study -> table06 + multiseed_results.csv")
    old_argv = sys.argv
    sys.argv = [str(script), *seeds]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:  # the script calls raise SystemExit(main())
        if exc.code not in (0, None):
            raise
    finally:
        sys.argv = old_argv
    # multiseed_robustness.py writes multiseed_results.csv / multiseed_agg.json to
    # the code root; the JMLR figure driver reads them from results/. Mirror the
    # fresh output there so fig_jmlr_robustness reflects this run.
    import shutil

    for name in ("multiseed_results.csv", "multiseed_agg.json"):
        src = code_root / name
        if src.is_file():
            shutil.copy2(src, code_root / "results" / name)


def _run_trace(script_root: Path) -> None:
    code_root = script_root
    script = code_root / "trace_evaluations.py"
    if not script.is_file():
        print(f"!! trace_evaluations.py not found at {script}; skipping")
        return
    print(">> [trace] trace_evaluations: per-query trace -> tables/trace_all_evaluations.csv")
    sys.path.insert(0, str(code_root))
    os.environ.setdefault("MPLCONFIGDIR", str(code_root / ".mplconfig"))
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        if sys.path and sys.path[0] == str(code_root):
            sys.path.pop(0)


def _run_jmlr(script_root: Path) -> None:
    code_root = script_root
    script = code_root / "make_jmlr_mirror_artifacts.py"
    if not script.is_file():
        print(f"!! make_jmlr_mirror_artifacts.py not found at {script}; skipping")
        return
    print(">> [4/4] make_jmlr_mirror_artifacts: fig_jmlr_* figures + table_prior_summary")
    runpy.run_path(str(script), run_name="__main__")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # The stand-alone drivers (multiseed, trace, JMLR figures) write next to
    # themselves in the installed submission/code tree; --output-dir only
    # redirects the package-native paper_artifacts stage.
    script_root = _code_root()
    code_root = args.output_dir.resolve() if args.output_dir else script_root
    for base in {code_root, script_root}:
        (base / "figures").mkdir(parents=True, exist_ok=True)
        (base / "tables").mkdir(parents=True, exist_ok=True)
        (base / "results").mkdir(parents=True, exist_ok=True)

    print(f"OST-QAOA full reproduction -> {code_root}")
    _run_paper_artifacts(args, code_root)

    if not args.skip_trace:
        _run_trace(script_root)
    else:
        print(">> [trace] skipped (--skip-trace)")

    if not args.skip_multiseed:
        _run_multiseed(args, script_root)
    else:
        print(">> [3/4] multiseed skipped (--skip-multiseed)")

    _run_jmlr(script_root)
    print("OST-QAOA reproduction complete: 7 figures + 2 tables regenerated.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
