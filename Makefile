.PHONY: install install-ocr test lint fix demo eval serve clean \
        docker docker-logs docker-stop docker-rebuild

VENV ?= .venv
PY   ?= $(VENV)/bin/python
ifeq ($(OS),Windows_NT)
PY := $(VENV)/Scripts/python.exe
endif

install:               ## create the venv and install the package with dev extras
	python -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[api,dev]" pypdf httpx

install-ocr:           ## add the Tesseract python binding (the binary is a system package)
	$(PY) -m pip install -e ".[tesseract]"
	@echo
	@echo "The Tesseract *binary* is separate:"
	@echo "  Debian/Ubuntu : sudo apt install tesseract-ocr tesseract-ocr-spa"
	@echo "  macOS         : brew install tesseract tesseract-lang"
	@echo "  Windows       : winget install UB-Mannheim.TesseractOCR"
	@echo
	@$(PY) -m scriptorium.cli doctor

test:                  ## run the test suite (no OCR binary required)
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src eval tests

fix:
	$(PY) -m ruff check --fix src eval tests

demo:                  ## end-to-end run on simulated captures, writes out/demo/demo.pdf
	$(PY) -m scriptorium.cli demo --out out/demo --pages 4

eval:                  ## ablation table: needs a real OCR engine to mean anything
	$(PY) -m eval.run --pages 4 --difficulty all --save

serve:                 ## API + browser client on http://127.0.0.1:8000
	$(PY) -m scriptorium.cli serve

docker:                ## build and (re)launch the stack in the background
	docker compose up -d --build

docker-logs:           ## follow the container's output
	docker compose logs -f

docker-stop:           ## stop the stack; books on disk are untouched
	docker compose down

docker-rebuild:        ## full rebuild ignoring the layer cache (slow: re-installs Tesseract)
	docker compose build --no-cache
	docker compose up -d

clean:
	rm -rf out eval/runs .pytest_cache .ruff_cache
