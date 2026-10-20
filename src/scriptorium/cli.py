"""Command line entry point.

    scriptorium scan 0 --out out/session        # live webcam
    scriptorium scan samples/book.mp4 --out out # recorded session
    scriptorium scan samples/pages/ --out out   # folder of stills
    scriptorium doctor                          # what is installed here
    scriptorium serve                           # HTTP API

`scan` works on a camera index, a video or a folder, because the pipeline reads
from a `FrameSource` and does not care which.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .capture.source import Frame, open_source
from .config import Config
from .export import epub as epub_export
from .export import pdf as pdf_export
from .export import text as text_export
from .ocr.registry import available_engines, diagnose, get_engine
from .pipeline.book import BookSession


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scriptorium", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="capture a book from a camera, video or folder")
    scan.add_argument("source", help="camera index, video file, image or folder")
    scan.add_argument("--out", default="out/session", help="output directory")
    scan.add_argument("--engine", default="auto", help="auto|tesseract|paddle|stub")
    scan.add_argument("--lang", default="spa+eng")
    scan.add_argument("--max-pages", type=int, default=None)
    scan.add_argument("--stride", type=int, default=1, help="frame subsampling for videos")
    scan.add_argument("--no-dewarp", action="store_true")
    scan.add_argument("--no-split", action="store_true")
    scan.add_argument("--save-images", action="store_true", help="also write page PNGs")

    demo = sub.add_parser(
        "demo", help="run the full pipeline on simulated captures (no camera needed)"
    )
    demo.add_argument("--out", default="out/demo")
    demo.add_argument("--pages", type=int, default=3)
    demo.add_argument(
        "--difficulty", default="default", choices=["easy", "default", "hard"]
    )
    demo.add_argument("--engine", default="auto")

    read = sub.add_parser(
        "read", help="scan and read aloud: the headless reading-aid mode"
    )
    read.add_argument("source", help="camera index, video file, image or folder")
    read.add_argument("--engine", default="auto")
    read.add_argument("--lang", default="spa+eng")
    read.add_argument("--voice", default="es", help="language hint for voice selection")
    read.add_argument("--rate", type=float, default=1.0, help="speech rate multiplier")
    read.add_argument("--max-pages", type=int, default=None)
    read.add_argument(
        "--silent",
        action="store_true",
        help="print the reading script without speaking (no audio device needed)",
    )

    truth = sub.add_parser(
        "groundtruth",
        help="prepare ground-truth drafts for a folder of real photos",
    )
    truth.add_argument("directory", help="folder of page photographs")
    truth.add_argument("--engine", default="auto")
    truth.add_argument("--lang", default="spa+eng")

    sub.add_parser("doctor", help="report which OCR engines are usable here")

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command == "scan":
        return _scan(args)
    if args.command == "demo":
        return _demo(args)
    if args.command == "read":
        return _read(args)
    if args.command == "groundtruth":
        return _groundtruth(args)
    if args.command == "doctor":
        return _doctor()
    if args.command == "serve":
        return _serve(args)
    parser.error("unknown command")
    return 2


def _scan(args) -> int:
    cfg = Config(ocr_engine=args.engine, ocr_lang=args.lang)
    if args.no_dewarp:
        cfg.dewarp.enabled = False
    if args.no_split:
        cfg.split_spreads = False

    engine = get_engine(cfg.ocr_engine, cfg.ocr_lang)
    if getattr(engine, "name", "") == "stub":
        print(
            "note: no real OCR engine installed, using the stub "
            "(placeholder text). Install with: pip install 'scriptorium[tesseract]'",
            file=sys.stderr,
        )

    session = BookSession(cfg=cfg, engine=engine)
    source = open_source(args.source, stride=args.stride)

    print(f"scanning {args.source} ...")
    try:
        session.run(source, max_pages=args.max_pages)
    except KeyboardInterrupt:
        print("\ninterrupted -- exporting what was captured")

    if not session.pages:
        print("no pages captured.")
        print(json.dumps(session.summary(), indent=2, ensure_ascii=False))
        return 1

    out = Path(args.out)
    pdf_path = pdf_export.write_pdf(session.pages, session.images, out / "book.pdf")
    # Both formats, because they are good at opposite things: the PDF keeps the
    # page so a transcription can be checked against the original, the EPUB
    # drops it so the text actually reflows for a screen reader or a phone.
    epub_export.write_epub(
        session.pages,
        out / "book.epub",
        epub_export.EpubMetadata(title=out.name, language=args.lang.split("+")[0]),
    )
    text_export.write_text(session.pages, out / "book.txt")
    text_export.write_markdown(session.pages, out / "book.md")
    text_export.write_json(session.pages, out / "book.json", summary=session.summary())
    if args.save_images:
        session.save_images(out / "pages")

    summary = session.summary()
    print("\n{} page(s) -> {}".format(summary["pages"], pdf_path))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _demo(args) -> int:
    """Simulated capture session: the fastest way to see the pipeline work.

    Lives here rather than in `eval/` because its job is different: the harness
    measures accuracy, this just proves the thing runs. A reviewer with no
    webcam, no Tesseract and no sample photos can still watch pages come out.
    """
    from pathlib import Path as _Path

    from eval import synth

    corpus_dir = _Path(__file__).resolve().parents[2] / "eval" / "corpora"
    corpus = "\n\n".join(f.read_text(encoding="utf-8") for f in sorted(corpus_dir.glob("*.txt")))
    chunks = synth.chunk_corpus(corpus, chunks=args.pages)
    scene_cfg = {
        "easy": synth.SceneConfig.easy(),
        "default": synth.SceneConfig(),
        "hard": synth.SceneConfig.hard(),
    }[args.difficulty]

    cfg = Config(ocr_engine=args.engine)
    engine = get_engine(cfg.ocr_engine, cfg.ocr_lang)
    session = BookSession(cfg=cfg, engine=engine)
    session.edge.require_stability = False

    print(
        f"simulating {len(chunks)} page(s) at '{args.difficulty}' difficulty "
        f"(curl {scene_cfg.curl_px:.0f}px, tilt {scene_cfg.tilt_deg:.0f}deg, "
        f"blur {scene_cfg.blur_sigma:.1f}) ..."
    )

    truths = []
    for index, chunk in enumerate(chunks):
        frame, truth = synth.make_scene(chunk, scene_cfg, seed=index)
        truths.append(truth)
        pages = session.feed(Frame(index=index, timestamp=float(index), image=frame))
        for page in pages:
            print(
                "  page {}: {} lines, dewarp={}, {:.0f} ms OCR".format(
                    page.page_number,
                    len(page.ocr.lines),
                    page.dewarped,
                    page.timings_ms.get("ocr_ms", 0.0),
                )
            )

    if not session.pages:
        print(f"no pages captured -- last rejection: {session.edge.last_rejection}")
        print(json.dumps(session.summary(), indent=2, ensure_ascii=False))
        return 1

    out = Path(args.out)
    pdf_export.write_pdf(session.pages, session.images, out / "demo.pdf")
    epub_export.write_epub(session.pages, out / "demo.epub")
    text_export.write_text(session.pages, out / "demo.txt")
    text_export.write_json(session.pages, out / "demo.json", summary=session.summary())
    (out / "ground_truth.txt").write_text("\n\n".join(truths), encoding="utf-8")

    print("\n-> {}".format(out / "demo.pdf"))
    print(json.dumps(session.summary(), indent=2, ensure_ascii=False))

    if getattr(engine, "name", "") != "stub":
        from eval.metrics import error_rates

        rates = error_rates("\n\n".join(truths), session.text)
        print(
            f"\naccuracy vs ground truth: CER {rates.cer:.3f}  WER {rates.wer:.3f}"
        )
    else:
        print("\n(stub engine: text is placeholder, run `scriptorium doctor` for real OCR)")
    return 0


def _read(args) -> int:
    """Scan and speak: capture a page, read it aloud, wait for the next one.

    The accessibility mode without a screen. Pages are read as they are captured
    rather than at the end, because a reader holding a book needs to hear the
    page they just turned to, not a transcript later.
    """
    from .export import speech

    cfg = Config(ocr_engine=args.engine, ocr_lang=args.lang)
    engine = get_engine(cfg.ocr_engine, cfg.ocr_lang)
    if getattr(engine, "name", "") == "stub":
        print(
            "note: no real OCR engine installed, so there is nothing meaningful "
            "to read. Run `scriptorium doctor`.",
            file=sys.stderr,
        )

    narrator = None
    if not args.silent:
        from .export.voice import Narrator, VoiceUnavailable

        try:
            narrator = Narrator(rate=args.rate, lang_hint=args.voice)
        except VoiceUnavailable as exc:
            print(f"no speech output ({exc}); falling back to --silent", file=sys.stderr)

    session = BookSession(cfg=cfg, engine=engine)
    source = open_source(args.source)
    session.edge.require_stability = not getattr(source, "is_still", False)

    print(f"reading {args.source} ... (ctrl-c to stop)")
    try:
        for frame in source.frames():
            for page in session.feed(frame):
                script = speech.build_script(page.ocr)
                print(f"\n--- page {page.page_number} ({len(script.words)} words) ---")
                if script.rejoined_words:
                    print(f"    rejoined {script.rejoined_words} hyphenated word(s)")

                if narrator is None:
                    print(script.text)
                else:
                    narrator.speak(script, on_word=lambda w: print(w.text, end=" ", flush=True))
                    print()
            if args.max_pages is not None and len(session.pages) >= args.max_pages:
                break
    except KeyboardInterrupt:
        print("\nstopped")

    print(f"\n{len(session.pages)} page(s) read")
    return 0 if session.pages else 1


def _groundtruth(args) -> int:
    """Seed a .txt transcription beside each photo, for `eval/run.py --real-dir`.

    Synthetic pages measure what the pipeline contributes; only real paper
    measures what a reader would actually get. The blocker is never the code, it
    is that transcribing pages by hand is slow enough not to happen -- so each
    draft starts from the recognised text and is marked unreviewed until a person
    corrects it and deletes the marker. The harness refuses to score a file that
    still carries it, which is what keeps the comparison from being circular.
    """
    import cv2

    cfg = Config(ocr_engine=args.engine, ocr_lang=args.lang)
    engine = get_engine(cfg.ocr_engine, cfg.ocr_lang)
    if getattr(engine, "name", "") == "stub":
        print("refusing to seed drafts from the stub engine -- run doctor", file=sys.stderr)
        return 1

    folder = Path(args.directory)
    if not folder.is_dir():
        print(f"not a directory: {folder}", file=sys.stderr)
        return 1

    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    photos = sorted(p for p in folder.iterdir() if p.suffix.lower() in suffixes)
    if not photos:
        print(f"no photographs found in {folder}", file=sys.stderr)
        return 1

    written, skipped, failed = 0, 0, 0
    for photo in photos:
        target = photo.with_suffix(".txt")
        if target.exists():
            skipped += 1
            continue
        session = BookSession(cfg=cfg, engine=engine)
        image = cv2.imread(str(photo))
        if image is None:
            print(f"  {photo.name}: unreadable", file=sys.stderr)
            failed += 1
            continue
        pages = session.run(_StillImage(image))
        if not pages:
            print(f"  {photo.name}: no page detected ({session.edge.last_rejection})")
            failed += 1
            continue
        text_export.write_draft_truth("\n\n".join(p.text for p in pages), target)
        print(f"  {photo.name} -> {target.name} ({len(pages)} page(s))")
        written += 1

    print(f"\n{written} draft(s) written, {skipped} already present, {failed} failed")
    if written:
        print(
            "\nNow correct each .txt against the photo, delete the UNREVIEWED line, "
            "and run:\n"
            f"  python -m eval.run --real-dir {folder}"
        )
    return 0


class _StillImage:
    """One decoded photograph as a FrameSource."""

    is_still = True

    def __init__(self, image) -> None:
        self._image = image

    def frames(self):
        yield Frame(index=0, timestamp=0.0, image=self._image)


def _doctor() -> int:
    status = available_engines()
    detail = diagnose()
    print("OCR engines:")
    for name, ok in status.items():
        print(f"  {name:<10} {detail.get(name, 'available' if ok else 'not installed')}")
    real = [n for n, ok in status.items() if ok and n != "stub"]
    if not real:
        print(
            "\nNo real OCR engine found. Install one:\n"
            "  pip install 'scriptorium[tesseract]'   (plus the Tesseract binary)\n"
            "  pip install 'scriptorium[paddle]'"
        )
    return 0


def _serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print("pip install 'scriptorium[api]' to use the API", file=sys.stderr)
        return 1
    uvicorn.run("scriptorium.api.app:app", host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
