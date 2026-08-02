# Builds the main report. latexmk resolves the number of passes needed for
# cross references and the bibliography on its own.
#
# The dash gate runs last, over the compiled PDF as well as the sources. TeX
# turns two hyphens into an en dash silently, so checking only the sources
# would miss exactly the mistake the rule exists to prevent.

LATEXMK ?= latexmk
PYTHON ?= python3
BUILD := build
MAIN := main
TARGET := $(MAIN).pdf

.PHONY: all clean check

all: $(TARGET)

$(TARGET): $(MAIN).tex $(wildcard chapters/*.tex) $(wildcard tables/*.tex) numbers.tex refs.bib
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
		-outdir=$(BUILD) $(MAIN).tex
	@cp $(BUILD)/$(MAIN).pdf $(TARGET)
	@$(MAKE) check

check:
	$(PYTHON) ../scripts/check_no_dashes.py .

clean:
	$(LATEXMK) -C -outdir=$(BUILD) $(MAIN).tex 2>/dev/null || true
	rm -rf $(BUILD) $(TARGET)
