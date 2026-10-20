"""Ablation harness: the numbers that justify every stage in the pipeline.

Run with:

    python -m eval.run --pages 6

Each variant sees *identical* input frames, so the difference between rows is
attributable to our code and nothing else:

* `raw_ocr`      -- OCR straight on the camera frame. The honest baseline: what
                    you get from a naive "point the camera and call OCR" build.
* `rectify_only` -- page detection + perspective rectification, no dewarping.
* `full`         -- rectification + text-line dewarping.

Reported per variant: CER, WER, milliseconds per page, and whether the page was
located at all. A stage that cannot pay for itself in this table does not belong
in the pipeline.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scriptorium.config import Config
from scriptorium.ocr.registry import get_engine
from scriptorium.pipeline.worker import PageWorker
from scriptorium.types import PageCandidate, QualityReport
from scriptorium.vision import detect

from . import synth
from .metrics import error_rates

CORPORA = Path(__file__).parent / "corpora"
RUNS = Path(__file__).parent / "runs"

DIFFICULTIES = {
    "easy": synth.SceneConfig.easy(),
    "default": synth.SceneConfig(),
    "hard": synth.SceneConfig.hard(),
}


@dataclass
class Sample:
    """One evaluation item: a camera frame and the text that is really on it."""

    name: str
    frame: np.ndarray
    truth: str
    difficulty: str


def build_synthetic_samples(pages: int, difficulties: list[str]) -> list[Sample]:
    text = _load_corpus()
    chunks = synth.chunk_corpus(text, chunks=pages)
    if not chunks:
        raise SystemExit("corpus produced no page-sized chunks")
    if len(chunks) < pages:
        # Sample size backs every number in the report; never let it differ from
        # what was asked for without saying so.
        print(
            f"NOTE: corpus yields only {len(chunks)} distinct page(s), not {pages}. "
            f"Add more text to eval/corpora/ for a larger sample."
        )

    samples = []
    for difficulty in difficulties:
        scene_cfg = DIFFICULTIES[difficulty]
        for index, chunk in enumerate(chunks):
            frame, truth = synth.make_scene(chunk, scene_cfg, seed=index)
            samples.append(
                Sample(
                    name=f"synth_{difficulty}_{index:02d}",
                    frame=frame,
                    truth=truth,
                    difficulty=difficulty,
                )
            )
    return samples


def load_real_samples(directory: str | Path) -> list[Sample]:
    """Load real captures: `page.jpg` paired with a reviewed `page.txt`.

    Drafts seeded by `scriptorium groundtruth` are **skipped**, loudly. They
    start as the pipeline's own output, so scoring against one that nobody
    corrected would compare the hypothesis to itself and report a perfect result.
    A reviewed file is one a person deliberately unmarked.
    """
    import cv2

    from scriptorium.export.text import read_reviewed_truth

    folder = Path(directory)
    samples = []
    unreviewed: list[str] = []
    for image_path in sorted(folder.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        truth_path = image_path.with_suffix(".txt")
        if not truth_path.exists():
            continue

        truth = read_reviewed_truth(truth_path)
        if truth is None:
            unreviewed.append(truth_path.name)
            continue
        if not truth.strip():
            continue

        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        samples.append(
            Sample(name=image_path.stem, frame=frame, truth=truth, difficulty="real")
        )

    if unreviewed:
        print(
            "SKIPPED {} unreviewed ground-truth draft(s): {}\n"
            "         They still hold the pipeline's own output, so scoring "
            "against them would be circular.\n"
            "         Correct each one against its photo and delete the "
            "UNREVIEWED line.".format(len(unreviewed), ", ".join(unreviewed))
        )
    return samples


def variants(base: Config) -> dict[str, tuple[Config, bool]]:
    """name -> (config, rectify?)."""
    return {
        "raw_ocr": (replace(base, dewarp=replace(base.dewarp, enabled=False)), False),
        "rectify_only": (replace(base, dewarp=replace(base.dewarp, enabled=False)), True),
        "full": (base, True),
    }


def run_variant(sample: Sample, cfg: Config, rectify: bool, worker: PageWorker) -> dict:
    start = time.perf_counter()

    detected = False
    image = sample.frame
    if rectify:
        quad = detect.find_page(sample.frame, cfg.detect)
        if quad is not None:
            detected = True
            image = detect.rectify(sample.frame, quad, cfg.detect)

    candidate = PageCandidate(
        frame_index=0,
        timestamp=0.0,
        quad=[],
        quality=QualityReport(0.0, 0.0, 0.0, 0.0, True),
        phash="0" * 16,
    )
    output = worker.process(candidate, image)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    hypothesis = "\n".join(result.text for result in output.results)
    rates = error_rates(sample.truth, hypothesis)
    return {
        "sample": sample.name,
        "difficulty": sample.difficulty,
        "detected": detected or not rectify,
        "dewarp_applied": any(r.applied for r in output.dewarp_reports),
        "ms": round(elapsed_ms, 1),
        **rates.as_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scriptorium ablation harness")
    parser.add_argument("--pages", type=int, default=4, help="synthetic pages per difficulty")
    parser.add_argument(
        "--difficulty",
        default="default",
        choices=[*DIFFICULTIES, "all"],
        help="synthetic degradation preset",
    )
    parser.add_argument("--engine", default="auto", help="OCR engine: auto|tesseract|paddle|stub")
    parser.add_argument("--lang", default="spa+eng")
    parser.add_argument("--real-dir", default=None, help="folder of real captures + .txt truth")
    parser.add_argument("--save", action="store_true", help="write JSON report to eval/runs/")
    args = parser.parse_args(argv)

    engine = get_engine(args.engine, args.lang)
    if getattr(engine, "name", "") == "stub":
        print(
            "WARNING: the stub OCR engine is selected, so error rates below are "
            "meaningless (it emits placeholder text).\n"
            "         Install Tesseract and pytesseract for real numbers:\n"
            "         pip install 'scriptorium[tesseract]'\n"
        )

    difficulties = list(DIFFICULTIES) if args.difficulty == "all" else [args.difficulty]
    samples = build_synthetic_samples(args.pages, difficulties)
    if args.real_dir:
        real = load_real_samples(args.real_dir)
        print(f"loaded {len(real)} real samples from {args.real_dir}")
        samples.extend(real)

    base = Config(ocr_engine=args.engine, ocr_lang=args.lang)
    rows: list[dict] = []
    for name, (cfg, rectify) in variants(base).items():
        worker = PageWorker(engine=engine, cfg=cfg)
        for sample in samples:
            row = run_variant(sample, cfg, rectify, worker)
            row["variant"] = name
            rows.append(row)
            print(
                "  {:<13} {:<18} CER {:.3f}  WER {:.3f}  {:>6.0f} ms".format(
                    name, row["sample"], row["cer"], row["wer"], row["ms"]
                )
            )

    table = aggregate(rows)
    print()
    print(render_markdown(table, engine_name=getattr(engine, "name", "?")))

    if args.save:
        RUNS.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report = RUNS / f"ablation_{stamp}.json"
        report.write_text(
            json.dumps(
                {"engine": getattr(engine, "name", "?"), "rows": rows, "aggregate": table},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nreport: {report}")
    return 0


def aggregate(rows: list[dict]) -> list[dict]:
    """Mean metrics per (variant, difficulty), ordered for the README table."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["variant"], row["difficulty"]), []).append(row)

    order = {"raw_ocr": 0, "rectify_only": 1, "full": 2}
    table = []
    for (variant, difficulty), items in sorted(
        groups.items(), key=lambda kv: (kv[0][1], order.get(kv[0][0], 9))
    ):
        table.append(
            {
                "variant": variant,
                "difficulty": difficulty,
                "pages": len(items),
                "cer": round(float(np.mean([i["cer"] for i in items])), 4),
                "wer": round(float(np.mean([i["wer"] for i in items])), 4),
                "ms": round(float(np.mean([i["ms"] for i in items])), 1),
                "detected": round(float(np.mean([i["detected"] for i in items])), 3),
                "dewarp_applied": round(
                    float(np.mean([i["dewarp_applied"] for i in items])), 3
                ),
            }
        )
    return table


def render_markdown(table: list[dict], engine_name: str) -> str:
    lines = [
        f"### Ablation (engine: {engine_name})",
        "",
        "| difficulty | variant | pages | CER | WER | ms/page | page found | dewarp fired |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    baseline: dict[str, float] = {}
    for row in table:
        if row["variant"] == "raw_ocr":
            baseline[row["difficulty"]] = row["cer"]
        lines.append(
            "| {difficulty} | `{variant}` | {pages} | {cer:.3f} | {wer:.3f} | {ms:.0f} | "
            "{detected:.0%} | {dewarp_applied:.0%} |".format(**row)
        )

    lines.append("")
    for difficulty, base_cer in baseline.items():
        best = min(
            (r for r in table if r["difficulty"] == difficulty), key=lambda r: r["cer"]
        )
        # A "100% reduction" from a baseline that was already 0.000 is not a
        # result, it is a division artefact. Say what actually happened instead.
        if base_cer < 0.001:
            lines.append(
                f"- **{difficulty}**: naive OCR already scores "
                f"CER {base_cer:.3f}; nothing to recover, and the pipeline does "
                f"not make it worse ({best['cer']:.3f})."
            )
            continue
        reduction = (base_cer - best["cer"]) / base_cer
        lines.append(
            f"- **{difficulty}**: CER {base_cer:.3f} -> {best['cer']:.3f} with "
            f"`{best['variant']}` (**{reduction:.0%} relative reduction** vs naive OCR)"
        )
    return "\n".join(lines)


def _load_corpus() -> str:
    files = sorted(CORPORA.glob("*.txt"))
    if not files:
        raise SystemExit(f"no corpus found in {CORPORA}")
    return "\n\n".join(f.read_text(encoding="utf-8") for f in files)


if __name__ == "__main__":
    raise SystemExit(main())
