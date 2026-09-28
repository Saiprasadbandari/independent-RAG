# Extraction Report

## Corpus

- `EU AI Regulations.pdf` — the EU Artificial Intelligence Act (Regulation (EU) 2024/1689), 144 pages.

## Method

- PDF text pulled page-by-page with PyMuPDF (`fitz.get_text()`).
- Cleanup steps applied per page:
  - Collapsed repeated whitespace and blank lines.
  - Detected and removed repeated header/footer lines (the `ELI: http://data.europa.eu/eli/reg/2024/1689/oj` footer, the `OJ L, 12.7.2024` line, and the `EN` language marker) by checking which lines recur near the top/bottom of a large share of pages.
  - Stripped page-counter lines like `61/144` with a regex, since the number changes per page and wouldn't otherwise be caught as an exact repeat.
  - Rejoined lines that PDF extraction breaks mid-sentence, and fixed hyphenated word splits across line breaks (e.g. `regu-\nlation` → `regulation`).

## Issues encountered

- **Repeated headers/footers**: every page carried the same 3-4 line footer block (citation, language code, page number). Handled as described above — removal was verified by spot-checking pages 11, 61, 101, and 141 before/after cleanup.
- **Mid-sentence line breaks**: raw PyMuPDF output frequently breaks sentences at the PDF's line width rather than at sentence boundaries. Handled by rejoining lines that don't end in sentence-ending punctuation.
- **Numbered/lettered sub-clauses**: the regulation's legal structure uses inline markers like `(a)`, `(9)`, `5.` at the start of clauses. These were kept as-is since they're part of the actual content, not extraction noise.
- **No scanned/image-only pages**: this PDF has a proper text layer throughout, so there were no empty extractions. OCR wasn't needed.
- **No tables**: the document is prose/legal text with no tables, so PyMuPDF's default extraction was sufficient — didn't need to fall back to pdfplumber.
- **No HTML sources**: current corpus is PDF-only, so the BeautifulSoup path in `extract.py` is untested against real data for this run (kept in the script for when web sources are added).

## Stats (from `extract.py` output)

```
source files: 1
total entries: 144
total characters: 584877
avg characters/entry: 4061
near-empty entries (<50 chars): 0
```

No empty or near-empty (< 50 char) entries were produced.
