LATEXMK ?= latexmk
PYTHON ?= python3
BUILD := build
MAIN := debug_report
TARGET := $(MAIN).pdf

.PHONY: all clean check

all: $(TARGET)

$(TARGET): $(MAIN).tex $(wildcard sections/*.tex) ../report/numbers.tex
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
		-outdir=$(BUILD) $(MAIN).tex
	@cp $(BUILD)/$(MAIN).pdf $(TARGET)
	@$(MAKE) check

check:
	$(PYTHON) ../scripts/check_no_dashes.py .

clean:
	$(LATEXMK) -C -outdir=$(BUILD) $(MAIN).tex 2>/dev/null || true
	rm -rf $(BUILD) $(TARGET)
