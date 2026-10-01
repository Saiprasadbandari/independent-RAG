import io

import pymupdf
from PIL import Image

from app.utils import DATA_DIR, split_sentences

SOURCES_DIR = DATA_DIR / "sources"
MAX_PAGES = 4        # pages shown per answer
MAX_FRAGMENTS = 8    # sentences searched per chunk
MIN_FRAGMENT = 25    # ignore tiny fragments


def render_page(source: str, page: int, chunk_text: str, zoom: float = 1.6):
    """PIL image of one PDF page with the chunk's sentences highlighted, or None (not a PDF)."""
    path = SOURCES_DIR / source
    if path.suffix.lower() != ".pdf" or not path.exists():
        return None
    with pymupdf.open(path) as doc:
        if not 1 <= page <= len(doc):
            return None
        pg = doc[page - 1]                                   # PyMuPDF counts from 0, corpus pages from 1
        sentences = [s for s in split_sentences(chunk_text) if len(s) >= MIN_FRAGMENT]
        for sentence in sentences[:MAX_FRAGMENTS]:
            for probe in (sentence[:120], sentence[:50]):    # longer match first, shorter as fallback
                rects = pg.search_for(probe)
                if rects:
                    pg.add_highlight_annot(quads=rects)
                    break
        pix = pg.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), annots=True)
        return Image.open(io.BytesIO(pix.tobytes("png")))    # highlights are in memory only; the PDF is never saved


def source_pages(sources: list[dict]) -> list[tuple]:
    """[(image, caption), ...] for gr.Gallery, built from the answer's cited sources."""
    pages = []
    for s in sources:
        img = render_page(s["source"], s["page"], s["text"])
        if img is not None:
            pages.append((img, f"[{s['ref']}] {s['source']}, p. {s['page']}"))
        if len(pages) >= MAX_PAGES:
            break
    return pages
