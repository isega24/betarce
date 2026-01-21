# Makefile for LaTeX compilation
# Usage:
#   make        - Build the PDF
#   make clean  - Remove auxiliary files
#   make distclean - Remove all generated files including PDF

MAIN = main
LATEX = pdflatex
BIBTEX = bibtex

# Default target
all: $(MAIN).pdf

# Build PDF with bibliography
$(MAIN).pdf: $(MAIN).tex references.bib Sections/*.tex
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

# Quick build without bibliography update
quick:
	$(LATEX) $(MAIN)

# Clean auxiliary files
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
	rm -f Sections/*.aux

# Clean everything including PDF
distclean: clean
	rm -f $(MAIN).pdf

# Watch for changes and rebuild (requires latexmk)
watch:
	latexmk -pdf -pvc $(MAIN)

.PHONY: all quick clean distclean watch
