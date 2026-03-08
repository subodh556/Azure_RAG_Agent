"""
UI helper functions: citation rendering, HTML badge construction, score bars.

render_answer()   — three-step citation rendering pipeline
cite_badge()      — circular superscript with hover tooltip
badge()           — generic inline label badge
score_bar_html()  — mini horizontal score bar
"""
from __future__ import annotations

import re

from ui.styles import AZ_BLUE, AZ_YELLOW

try:
    import markdown as _md_lib
    _MARKDOWN_AVAILABLE = True
except ImportError:
    _MARKDOWN_AVAILABLE = False


# ── Citation rendering ────────────────────────────────────────────────────────

def cite_badge(num: int, doc_name: str = "", page: int = 0, snippet: str = "") -> str:
    """Circular superscript citation badge with hover tooltip (pure HTML)."""
    safe_doc     = (doc_name
                    .replace("&", "&amp;").replace("<", "&lt;")
                    .replace(">", "&gt;").replace('"', "&quot;"))
    safe_snippet = (snippet
                    .replace("&", "&amp;").replace("<", "&lt;")
                    .replace(">", "&gt;").replace('"', "&quot;"))
    page_str     = f" · p.{page}" if page else ""
    title_html   = (
        f'<div class="cite-tip-title">[{num}] {safe_doc}{page_str}</div>'
        if safe_doc else ""
    )
    body_html    = (
        f'<div class="cite-tip-body">{safe_snippet[:300]}'
        f'{"…" if len(safe_snippet) > 300 else ""}</div>'
        if safe_snippet else ""
    )
    tooltip_html = (
        f'<span class="cite-tooltip">{title_html}{body_html}</span>'
        if (title_html or body_html) else ""
    )
    return (
        f'<span class="cite-wrap">'
        f'<span class="cite-num">{num}</span>'
        f'{tooltip_html}'
        f'</span>'
    )


def render_answer(text: str, chunks: list | None = None) -> str:
    """
    Three-step citation rendering pipeline.

    Step 1 — Replace [N] with @@CITE_N@@ before markdown parsing.
             Word-boundary regex avoids false matches in ordered lists or
             numeric ranges.
    Step 2 — Parse markdown → HTML (bold, bullets, headings, etc.).
    Step 3 — Replace @@CITE_N@@ with full tooltip badge HTML, keyed from
             the chunks list.

    Also strips any ## References / ## Sources section the model adds
    despite being instructed not to.
    """
    chunk_map: dict[int, object] = {}
    if chunks:
        for i, c in enumerate(chunks, 1):
            chunk_map[i] = c

    # Step 1
    protected = re.sub(
        r'(?<!\d)\[(\d+)\](?!\d)',
        lambda m: f"@@CITE_{m.group(1)}@@",
        text,
    )
    protected = re.sub(
        r'\n{1,2}#{1,3}\s*(References|Sources|Citations)\b.*',
        '',
        protected,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Step 2
    if _MARKDOWN_AVAILABLE:
        html = _md_lib.markdown(protected, extensions=["extra", "nl2br"])
    else:
        html = protected.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = html.replace("\n", "<br>")

    # Step 3
    def _to_badge(m: re.Match) -> str:
        n = int(m.group(1))
        c = chunk_map.get(n)
        return (
            cite_badge(n, doc_name=c.doc_name, page=c.page_num, snippet=c.text)
            if c else cite_badge(n)
        )

    return re.sub(r'@@CITE_(\d+)@@', _to_badge, html)


# ── Misc UI helpers ───────────────────────────────────────────────────────────

def badge(label: str, color: str = AZ_BLUE) -> str:
    """Generic inline label badge (non-citation uses)."""
    return (
        f'<span style="background:{color};color:#fff;font-size:11px;'
        f'font-weight:700;padding:2px 8px;border-radius:4px;margin:0 2px">'
        f'{label}</span>'
    )


def score_bar_html(score: float, max_score: float = 1.0, width: int = 120) -> str:
    """A small inline horizontal progress bar in HTML."""
    pct    = min(score / max(max_score, 0.001), 1.0)
    filled = int(pct * width)
    color  = AZ_BLUE if pct > 0.4 else AZ_YELLOW
    return (
        f'<div style="display:inline-block;width:{width}px;height:8px;'
        f'background:#1e3a5f;border-radius:4px;vertical-align:middle">'
        f'<div style="width:{filled}px;height:8px;background:{color};'
        f'border-radius:4px"></div></div>'
    )
