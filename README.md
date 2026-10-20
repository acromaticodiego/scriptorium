<div align="center">

# Scriptorium

**Point a camera at an open book and get back a searchable PDF** — then have it
read aloud, with the highlight following the voice word by word.

[![ci](https://github.com/acromaticodiego/scriptorium/actions/workflows/ci.yml/badge.svg)](https://github.com/acromaticodiego/scriptorium/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![tests](https://img.shields.io/badge/tests-75-brightgreen)
![CER](https://img.shields.io/badge/CER%20vs%20naive%20OCR-−97%25-brightgreen)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

*Computer vision pipeline · Python · OpenCV · FastAPI*

</div>

Not a photo of a page — a document whose text you can select, search, and hand to
a screen reader.

| | naive approach | Scriptorium |
| --- | --- | --- |
| hard captures (curl, tilt, glare, defocus) | CER **0.640** | CER **0.018** |
| words wrong | ~97 in 100 | ~3 in 100 |
| time per page | 962 ms | 808 ms |

<sub>Tesseract 5.4, Spanish, 6 pages per difficulty. Reproduce with `make eval`.
Measured on rendered pages through a camera simulator — see
[what these numbers are and are not](#results).</sub>

```
┌─ camera ─────────────────────────────────────────────────────────────────┐
│                                                                          │
│   30 fps ──►  EDGE LOOP  ── a page settled and we have not seen it ──┐   │
│               7.7 ms/frame                                           │   │
│               motion · page detection                                │   │
│                                                                      ▼   │
│                                                            ┌─ WORKER ──┐ │
│                                                            │ rectify   │ │
│                                                            │ split     │ │
│                                                            │ dewarp    │ │
│                                                            │ OCR       │ │
│                                                            └─────┬─────┘ │
│                                                                  ▼       │
│                                              searchable PDF · EPUB · TTS │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## The idea this project is built around

The naive version of this is three lines: grab a frame, call an OCR library, append
the text. It produces garbage, for a reason worth stating precisely.

**A book is not a video.** It is a sequence of discrete, static pages that happen to
arrive over a video feed. So the camera loop's job is not "transcribe what you see"
— it is **"tell me when a new page is sitting still in front of you"**. That one
reframing is what the whole architecture follows from:

| Naive approach | Here |
| --- | --- |
| OCR every frame | OCR once per page — ~30× less compute |
| Reconcile 30 contradictory transcriptions of one page | One transcription, from the sharpest frame |
| Accept whatever the camera gave you | Reject blurred, glared and badly lit frames before paying for them |
| Feed the engine a curved, tilted spread | Rectify, split the two pages, flatten the text lines |

The hard parts are not the OCR. Recognition is commoditised — it sits behind a
one-method interface ([`ocr/base.py`](src/scriptorium/ocr/base.py)) and Tesseract,
PaddleOCR or a fine-tuned handwriting model are drop-in. The engineering is
**everything around it**.

---

## Try it in 30 seconds

No camera, no OCR engine, no sample photos needed — the repo ships a capture
*simulator* and a dependency-free OCR stub, so a fresh clone runs end to end:

```bash
make install            # or: python -m venv .venv && pip install -e ".[api,dev]"
make demo               # simulates curled, tilted, unevenly lit captures
                        # -> out/demo/demo.pdf
make test               # 75 tests, no OCR binary required
```

For real recognition, add an engine — `make install-ocr` prints the one command for
your OS, and `scriptorium doctor` tells you what is actually usable, including
*which languages* are installed (the Windows installer ships English only, which
is the failure you hit when scanning Spanish). Then:

```bash
scriptorium scan 0 --out out/session      # live webcam
scriptorium scan video.mp4 --out out      # a recorded capture session
scriptorium scan pages/ --out out         # a folder of photos
scriptorium read 0 --voice es             # capture and read aloud, no screen
scriptorium doctor                        # what is installed here
make serve                                # API + browser client on :8000
```

`make serve` opens a browser client that streams webcam frames to the API, shows the
live capture state — *looking for a page → hold still → captured → turn the page* —
reads each captured page aloud with the highlight tracking the voice, and hands you
the PDF.

---

## The five stages that do the work

### 1. Frame triage — [`vision/quality.py`](src/scriptorium/vision/quality.py)

A blurred frame costs exactly as much to OCR as a good one and yields text that has
to be detected and thrown away later. Three cheap no-reference measures reject it
first: variance of Laplacian for focus, clipping ratio for exposure, and a glare
detector.

The glare detector is the interesting one. Glare is bright and locally flat — but
**so is a blank margin**, so brightness and flatness alone flag every page with
white space on it. The discriminator is *locality*: a reflection stands above the
paper around it, while paper sits level with its own neighbourhood. Comparing each
pixel against a large-scale background separates them.

The thresholds are calibrated against **measured CER, not appearance** — see
[Measurements](#a-gate-calibrated-by-eye-was-refusing-pages-the-pipeline-could-read).

### 2. Page-turn state machine — [`vision/stability.py`](src/scriptorium/vision/stability.py)

```
SEARCHING ──page found──► MOVING ──motion drops──► SETTLING
SETTLING  ──held N frames──► fire capture ──► CAPTURED
CAPTURED  ──motion spikes──► MOVING
```

The `CAPTURED` state is what stops the page the reader is still holding from being
transcribed forty times. Costs 3.7 ms/frame on a 160×120 grayscale copy.

### 3. Rectification — [`vision/detect.py`](src/scriptorium/vision/detect.py)

Finds the page quadrilateral (edge-based, falling back to brightness for a white
page on a light table) and warps it upright. The output aspect ratio comes from the
detected edge lengths, so a landscape spread is not squeezed into portrait.

### 4. Dewarping — [`vision/dewarp.py`](src/scriptorium/vision/dewarp.py)

**The stage that doesn't exist in the naive version, and the one that matters most.**

A page lying open is not a plane — it curls towards the spine. A homography maps a
*plane* to a rectangle, so after rectification the text lines are still bowed, and a
bowed line gets shredded by the engine's line segmentation.

No model, no training data. Binarise, cut the page into vertical slices, take the
ink projection of each, link the peaks across slices into baselines, fit a
polynomial to each, and resample through the resulting displacement field.

Then the part I'd argue is the real idea here:

> **Verify instead of predict.** Every threshold above is a *guess* about whether
> the estimate is trustworthy. But once the correction exists, you can simply
> measure whether it helped — with a metric that doesn't come from the fit — and
> throw it away if it didn't.

That replaced a pile of increasingly desperate heuristics, and it is why the stage
can be tuned aggressively without risking good pages.

### 5. Spread splitting — [`vision/spread.py`](src/scriptorium/vision/spread.py)

An open book is two pages. Fed whole to OCR, sentences from the left page continue
into the right one. The gutter is found as a dark vertical valley near the centre —
the paper curves away from the light there.

Splitting happens **before** dewarping, deliberately: the two halves curve in
*opposite* directions, so one displacement field across the gutter averages the two
and flattens neither.

---

## Measurements

Every threshold in [`config.py`](src/scriptorium/config.py) carries the measurement
that chose it, in a comment. Three that changed the design:

### Dewarping got to 57% in four measured steps, not one idea

It recovers 57% of the line-straightness gap back to a flat page (48 rendered
pages × 4 curvatures). Each step came from a measurement rather than a hunch:

| Fix | Recovery | What was wrong |
| --- | --- | --- |
| first version | — | polynomials extrapolated past the text that supported them, inventing 19 px of displacement on *flat* pages |
| clamp fits to their own span | 28% | |
| sub-pixel ink centroids instead of `argmax` | 34% | peak row moved with the glyph mix in each slice |
| line spacing from ink runs, not peak NMS | **57%** | a guessed minimum distance read one text line as three, so a 30-line page produced 85 "baselines" |

### A gate calibrated by eye was refusing pages the pipeline could read

The quality gate's first thresholds came from looking at degraded pages and judging
them unreadable. Then the end-to-end demo at `hard` difficulty captured **zero
pages** — all rejected for glare — while the ablation, which calls the worker
directly and bypasses the gate, transcribed those same frames at **CER 0.013**.

The gate was refusing pages the system reads almost perfectly. Sweeping glare and
measuring CER at each step instead of judging it:

| synth glare | measured glare | clipped | CER |
| --- | --- | --- | --- |
| 0.00 | 0.034 | 0.022 | 0.000 |
| 0.35 | 0.045 | 0.118 | 0.001 |
| 0.50 | 0.066 | 0.163 | 0.012 |
| 0.65 | 0.072 | 0.201 | **0.111** ← degradation starts |
| 0.80 | 0.076 | 0.230 | 0.170 |

The old limits (glare 0.03, clipping 0.08) sat below the *first* row — they rejected
pages with no glare blob at all, because a strong lighting gradient alone crosses
them. The limits now sit between rows 3 and 4, where error actually climbs.

The lesson generalises past this one threshold: **a quality gate's job is to predict
downstream failure, so it has to be calibrated against downstream failure.** Tuned
by eye, it becomes a deadlock — a reader stuck on "too much glare" with nothing ever
captured, and no way to tell that the system could have read the page fine.

### The dedup threshold is deliberately tight, and that is not the obvious choice

Perceptual hashing gates pages before OCR. Measured on rectified pages across
varying tilt, focus and noise: the same page hashes within 3.1% of the bits;
different prose pages sit 12–18% apart — *but* two pages sharing a layout pattern
can come as close as 6.6%. The failure modes are not symmetric:

- too tight → a page is OCRed twice, and the text-similarity gate catches it. Cost: one wasted OCR call.
- too loose → a real page is dropped **before OCR**, so nothing downstream can recover it. Cost: a page missing from the book, with nothing in the output to say so.

So the bar sits just above the same-page population, and the second gate does the
content matching. Two gates, because one is not sufficient.

**Latency** (1589×2248 frame, 1083×1600 rectified page, CPU only):

| Stage | Time | |
| --- | --- | --- |
| motion / stability | 3.7 ms | per frame |
| page detection | 4.0 ms | per frame |
| **edge loop total** | **7.7 ms** | **23% of the 33 ms budget at 30 fps** |
| OCR (Tesseract, 1083x1600) | ~750 ms | per capture — why it never runs per frame |
| rectify | 4.1 ms | per capture |
| quality gate | 8.6 ms | per capture |
| perceptual hash | 2.5 ms | per capture |
| spread analysis | 0.5 ms | per capture |
| dewarp | 46.9 ms | per capture |

### Ground truth without a scanner

Evaluating a scanning pipeline normally needs somebody to photograph pages and type
out what they say. [`eval/synth.py`](eval/synth.py) sidesteps it: take a
public-domain text, **render** it as a page, then simulate the camera — page curl,
perspective tilt, uneven lighting, specular glare, defocus, sensor noise. The source
text is exact ground truth, free, in unlimited quantity, and the degradations are
*controllable*, which turns a single score into a diagnosis.

```bash
make eval        # ablation: raw OCR vs rectify-only vs full pipeline,
                 # at three difficulty levels, on identical input frames
```

Each variant sees byte-identical frames, so the difference between rows belongs to
this code and not to a version bump in somebody's model.

### Results

Tesseract 5.4, Spanish, 6 pages per difficulty, `python -m eval.run --pages 6
--difficulty all --lang spa`:

| difficulty | variant | CER | WER | ms/page | dewarp fired |
| --- | --- | --- | --- | --- | --- |
| easy | `raw_ocr` | 0.000 | 0.002 | 1007 | 0% |
| easy | `rectify_only` | 0.000 | 0.000 | 769 | 0% |
| easy | `full` | 0.000 | 0.000 | 795 | 0% |
| default | `raw_ocr` | 0.030 | 0.056 | 963 | 0% |
| default | `rectify_only` | 0.027 | 0.053 | 749 | 0% |
| default | **`full`** | **0.000** | **0.001** | 840 | 100% |
| hard | `raw_ocr` | 0.640 | 0.971 | 962 | 0% |
| hard | `rectify_only` | 0.395 | 0.634 | 739 | 0% |
| hard | **`full`** | **0.018** | **0.031** | 808 | 100% |

**On hard captures — 44 px of page curl, 11° tilt, specular glare, defocus — naive
OCR fails outright at CER 0.640 / WER 0.971: almost every word is wrong. The full
pipeline brings it to CER 0.018, a 97% relative reduction.**

Three things in that table are worth more than the headline:

- **Rectification alone is not the win.** It takes hard CER from 0.640 only to
  0.395. Dewarping is what collapses the error: 0.395 → 0.018. Neither stage is
  worth much without the other, which is not what I expected before measuring.
- **`easy` is the control**, not a weak result. Everything scores zero, and the
  point is that the pipeline does not *damage* captures that were already fine —
  a real risk, since every stage resamples pixels.
- **The pipeline is faster than the baseline** (808 ms vs 962 ms on hard). Not a
  trick: a rectified, cropped page is a smaller image with less background for the
  engine to hunt through, so the extra ~60 ms of vision work buys back more than
  it costs in OCR.

> **What these numbers are and are not.** They come from *rendered* pages pushed
> through a camera simulator, so the glyphs are geometrically perfect — real paper
> adds texture, show-through and print variation that no simulator here reproduces.
> Read them as a measurement of **what the pipeline contributes**, which is exactly
> what an ablation on identical frames is for, and not as the CER you would get
> photographing a paperback. For that, `eval/run.py --real-dir` takes a folder of
> real captures with `.txt` ground truth alongside, and that number is still to
> come. The synthetic set is what makes these reproducible.

---

## API

```
POST   /v1/sessions                        open a book session
POST   /v1/sessions/{id}/frames            push a frame -> capture state + any new page
GET    /v1/sessions/{id}                   pages, confidences, timings
GET    /v1/sessions/{id}/pages/{n}/script  read-aloud script: text + word boxes
GET    /v1/sessions/{id}/pages/{n}/image   the dewarped page the boxes refer to
GET    /v1/sessions/{id}/pdf               the assembled searchable PDF
DELETE /v1/sessions/{id}                   close
POST   /v1/pages                           stateless: one photo -> queued job
GET    /v1/jobs/{id}                       job status + result
GET    /health                             engines available, sessions, queue depth
```

Two client shapes, one pipeline: **stateless** for a mobile app or a batch
digitisation script, **session** for a live capture where pages accumulate and dedup
state has to persist. A session is the *book*, not the request.

Requests enqueue and return immediately — the API answers in milliseconds, OCR takes
hundreds of them.

## Scaling

What is already right: the worker is a pure function of `(image, config)`, the queue
sits behind a protocol ([`jobs/base.py`](src/scriptorium/jobs/base.py)), and the
edge loop never touches OCR. So moving recognition onto its own machines is a
swap of `InMemoryQueue` for a Redis-backed implementation.

What is honestly not: **sessions live in process memory**, so more than one replica
needs them in shared storage (page images to object storage, session state to Redis)
before a load balancer can put two frames from one book on two different hosts. The
in-process queue is good for a single machine and says so.

## Reading aloud

Capture a page and it is read back, with the highlight following the voice word by
word. In the browser (`make serve`, auto-reads each page as it is captured) and from
the terminal for a station with no screen:

```bash
scriptorium read 0 --voice es        # camera -> speech, page after page
scriptorium read page.jpg --silent   # print the reading script instead
```

### The one hard problem in it

Speech engines report progress as a **character offset into the string you gave
them** — Web Speech via `boundary` events, pyttsx3 via `started-word`. To highlight
the right word, that offset has to be resolved against the same string the engine is
speaking. So the utterance and the word geometry are built together, once, on the
server ([`export/speech.py`](src/scriptorium/export/speech.py)), and both clients
consume the result.

Letting the browser rebuild the utterance by joining words itself is the obvious
shortcut and it silently breaks: any disagreement about joining — one space or two,
whether a line break becomes a space, how a hyphenated word is put back together —
shifts every offset after it, and the highlight drifts a few words behind the voice
with nothing to indicate why.

### Hyphenation is not a detail

Typeset books break words across lines. Read literally, that is not reading, it is
spelling: *"conti-"*, *"nuación"*. So a line-final hyphen whose continuation starts
lowercase is stitched back together — guarded, because `Madrid-` / `Barcelona` is a
compound, not a broken word — and the rejoined word keeps **both** bounding boxes,
so the reader sees both halves light up.

It also turned out to matter for accuracy, not just for the voice. The capture
simulator now hyphenates like a real book, and on those pages the repair is worth
most of the remaining error:

| text | CER |
| --- | --- |
| raw line-joined OCR | 0.0082 |
| hyphen breaks repaired | **0.0002** |

A 98% reduction of what was left — the unrepaired text claims words like
*"emen-dado"* that are not words. That is why the repair lives in
`PageResult.text` and reaches the txt, Markdown and JSON exports too, not only the
speech path.

## Why the PDF carries a text layer

An image-only PDF is completely opaque to a screen reader. With an *invisible but
selectable* text layer sitting on the matching pixels, the page is readable by
assistive technology, searchable, and still checkable against the original — a
reader can verify the transcription instead of trusting it.

## Layout

```
src/scriptorium/
  config.py          every threshold, with the measurement that chose it
  vision/            quality · detect · stability · dewarp · spread · dedup
  ocr/               engine interface + tesseract / paddle / dependency-free stub
  pipeline/          edge (per-frame) · worker (per-page) · book (session)
  export/            searchable PDF · txt / md / json · reading script · voice
  jobs/              queue protocol + in-process implementation
  api/               FastAPI app
  capture/           camera · video · image-folder sources
eval/
  synth.py           the capture simulator: ground truth without a scanner
  metrics.py         CER / WER
  run.py             ablation harness
web/index.html       browser client, streams webcam frames to the API
```

## Roadmap

- [ ] Error rates on **real** captures (`--real-dir`) to sit beside the synthetic ones
- [ ] Compare PaddleOCR against Tesseract on identical frames
- [x] Read-aloud mode with word-level highlighting, in the browser and headless
- [ ] Join a word split across a *page* break, not only a line break
      (`ReadingScript.ends_mid_word` already flags it; the join needs both pages)
- [ ] EPUB export with reflowable text
- [ ] Layout analysis: multi-column, footnotes, page numbers, reading order
- [ ] Handwriting via a fine-tuned TrOCR, behind the same `OcrEngine` interface
- [ ] Redis queue + shared session storage for multi-replica deployment
- [ ] On-device edge loop (the per-frame stages are already cheap enough for a phone)

## Licence

MIT. The sample corpus is public domain (*Don Quijote*, Cervantes).
