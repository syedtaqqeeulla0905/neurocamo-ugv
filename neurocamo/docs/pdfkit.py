"""
Tiny Platypus wrapper used by the viva-document builders.
=========================================================
The two builders (build_explainer.py, build_questions.py) describe their pages
as a flat list of ("kind", payload) tuples so the prose stays readable in the
source file. This module turns that list into a styled PDF.

Palette is lifted from the NeuroCamo dashboard so the printed handout and the
live demo look like one project.
"""

from __future__ import annotations

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, ListFlowable, ListItem, PageBreak,
    PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

INK = colors.HexColor("#101014")
MUTED = colors.HexColor("#55555f")
ACCENT = colors.HexColor("#00815a")     # darkened dashboard green, print-safe
RULE = colors.HexColor("#d0d0d6")
BAND = colors.HexColor("#f2f2f5")
WARN = colors.HexColor("#a35c00")

_ss = getSampleStyleSheet()


def _styles() -> dict:
    body = ParagraphStyle(
        "body", parent=_ss["BodyText"], fontName="Helvetica", fontSize=9.6,
        leading=14.2, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=5,
    )
    return {
        "body": body,
        "h1": ParagraphStyle(
            "h1", parent=body, fontName="Helvetica-Bold", fontSize=16.5,
            leading=20, textColor=INK, spaceBefore=6, spaceAfter=9,
            alignment=TA_JUSTIFY,
        ),
        "h2": ParagraphStyle(
            "h2", parent=body, fontName="Helvetica-Bold", fontSize=11.6,
            leading=15, textColor=ACCENT, spaceBefore=11, spaceAfter=5,
        ),
        "h3": ParagraphStyle(
            "h3", parent=body, fontName="Helvetica-Bold", fontSize=10,
            leading=13.5, textColor=INK, spaceBefore=8, spaceAfter=3,
        ),
        "q": ParagraphStyle(
            "q", parent=body, fontName="Helvetica-Bold", fontSize=9.8,
            leading=13.4, textColor=INK, spaceBefore=9, spaceAfter=2,
            alignment=TA_JUSTIFY,
        ),
        "code": ParagraphStyle(
            "code", parent=body, fontName="Courier", fontSize=8.3, leading=11.4,
            textColor=INK, alignment=0, spaceBefore=3, spaceAfter=6,
            leftIndent=7, backColor=BAND, borderPadding=5,
        ),
        "note": ParagraphStyle(
            "note", parent=body, fontName="Helvetica-Oblique", fontSize=9,
            leading=13, textColor=MUTED,
        ),
        "cover_title": ParagraphStyle(
            "ct", parent=body, fontName="Helvetica-Bold", fontSize=27,
            leading=31, textColor=INK, alignment=TA_CENTER, spaceAfter=8,
        ),
        "cover_sub": ParagraphStyle(
            "cs", parent=body, fontName="Helvetica", fontSize=12.5, leading=17,
            textColor=MUTED, alignment=TA_CENTER, spaceAfter=4,
        ),
        "cell": ParagraphStyle(
            "cell", parent=body, fontSize=8.6, leading=11.6, spaceAfter=0,
            alignment=0,
        ),
        "cellb": ParagraphStyle(
            "cellb", parent=body, fontName="Helvetica-Bold", fontSize=8.6,
            leading=11.6, spaceAfter=0, alignment=0,
        ),
    }


S = _styles()
def _esc(t: str) -> str:
    """Escape for Platypus mini-HTML, but keep <b>/<i>/<font> we author ourselves."""
    t = t.replace("&", "&amp;")
    for tag in ("b", "i", "font", "br", "sub", "super"):
        t = t.replace(f"&lt;{tag}", f"<{tag}").replace(f"&lt;/{tag}", f"</{tag}")
    return t


def _table(rows: list, widths: list | None = None, head: bool = True) -> Table:
    data = [[Paragraph(_esc(str(c)), S["cellb"] if (head and r == 0) else S["cell"])
             for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1 if head else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]
    if head:
        style += [("BACKGROUND", (0, 0), (-1, 0), BAND),
                  ("LINEBELOW", (0, 0), (-1, 0), 0.9, ACCENT)]
    t.setStyle(TableStyle(style))
    return t


def render(blocks: list) -> list:
    """("kind", payload) tuples -> Platypus flowables."""
    out = []
    for kind, payload in blocks:
        if kind == "h1":
            out += [Paragraph(_esc(payload), S["h1"]),
                    Table([[""]], colWidths=[168 * mm], rowHeights=[1.1],
                          style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), ACCENT)])),
                    Spacer(1, 7)]
        elif kind in ("h2", "h3", "note", "q", "code"):
            out.append(Paragraph(_esc(payload), S[kind]))
        elif kind == "p":
            out.append(Paragraph(_esc(payload), S["body"]))
        elif kind == "bul":
            out.append(ListFlowable(
                [ListItem(Paragraph(_esc(x), S["body"]), leftIndent=13)
                 for x in payload],
                bulletType="bullet", bulletFontSize=6, bulletColor=ACCENT,
                leftIndent=11, spaceAfter=5,
            ))
        elif kind == "num":
            out.append(ListFlowable(
                [ListItem(Paragraph(_esc(x), S["body"]), leftIndent=15)
                 for x in payload],
                bulletType="1", bulletFontName="Helvetica-Bold",
                bulletColor=ACCENT, leftIndent=13, spaceAfter=5,
            ))
        elif kind == "table":
            rows, widths = payload if isinstance(payload, tuple) else (payload, None)
            out += [Spacer(1, 2), _table(rows, widths), Spacer(1, 7)]
        elif kind == "qa":
            # keep a question welded to its answer across a page break
            q, a = payload
            out.append(KeepTogether([Paragraph(_esc(q), S["q"]),
                                     Paragraph(_esc(a), S["body"])]))
        elif kind == "gap":
            out.append(Spacer(1, payload))
        elif kind == "break":
            out.append(PageBreak())
        else:
            raise ValueError(f"unknown block kind: {kind}")
    return out
def cover(title: str, subtitle: str, tagline: str, facts: list) -> list:
    return [
        Spacer(1, 42 * mm),
        Paragraph(_esc(title), S["cover_title"]),
        Paragraph(_esc(subtitle), S["cover_sub"]),
        Spacer(1, 4),
        Table([[""]], colWidths=[62 * mm], rowHeights=[1.6], hAlign="CENTER",
              style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), ACCENT)])),
        Spacer(1, 9),
        Paragraph(_esc(tagline), S["cover_sub"]),
        Spacer(1, 20 * mm),
        _table(facts, widths=[52 * mm, 96 * mm], head=False),
        PageBreak(),
    ]


def build(path: str, blocks: list, running_head: str) -> str:
    doc = BaseDocTemplate(
        path, pagesize=A4,
        leftMargin=21 * mm, rightMargin=21 * mm,
        topMargin=19 * mm, bottomMargin=17 * mm,
        title=running_head, author="NeuroCamo project team",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="f")

    def decorate(canvas, _doc):
        canvas.saveState()
        if _doc.page > 1:
            canvas.setFont("Helvetica", 7.4)
            canvas.setFillColor(MUTED)
            canvas.drawString(doc.leftMargin, A4[1] - 13 * mm, running_head.upper())
            canvas.drawRightString(A4[0] - doc.rightMargin, A4[1] - 13 * mm,
                                   "NEUROCAMO / SNN CAMOUFLAGE DETECTION")
            canvas.setStrokeColor(RULE)
            canvas.setLineWidth(0.4)
            canvas.line(doc.leftMargin, A4[1] - 15 * mm,
                        A4[0] - doc.rightMargin, A4[1] - 15 * mm)
            canvas.drawCentredString(A4[0] / 2, 10 * mm, str(_doc.page))
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=decorate)])
    doc.build(list(blocks))
    return path


