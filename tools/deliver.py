"""Helpers to build the milestone report (.docx) and slide deck (.pptx) with one consistent look."""
from __future__ import annotations

import os

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from pptx import Presentation
from pptx.dml.color import RGBColor as PRGB
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches
from pptx.util import Pt as PPt

PROJECT = "AI-Driven Rehabilitation Exercise Monitoring and Correction Using Computer Vision"
SYSTEM = "RehabNet"
TEAM = [("Vasantha Kumar G R", "CB.EN.U4CCE23062"), ("Tarun Sri Vathsan K", "CB.EN.U4CCE23058"),
        ("Vethaviyash K", "CB.EN.U4CCE23063")]
MENTOR = "Dr. Anusha K S"
BATCH = "ECE 067"

NAVY = (0x12, 0x2B, 0x4A)
TEAL = (0x0F, 0x76, 0x6E)
GREY = (0x55, 0x5F, 0x6B)
LIGHT = (0xF2, 0xF5, 0xF8)


# ----------------------------------------------------------------- report (docx)
class Report:
    def __init__(self, milestone: str, subtitle: str, period: str):
        self.d = Document()
        st = self.d.styles["Normal"]
        st.font.name = "Calibri"
        st.font.size = Pt(11)
        for s in self.d.sections:
            s.left_margin = s.right_margin = Cm(2.2)
            s.top_margin = s.bottom_margin = Cm(2.0)
        self._fig = 0
        self._tab = 0
        self._title(milestone, subtitle, period)

    def _title(self, milestone, subtitle, period):
        d = self.d
        for _ in range(3):
            d.add_paragraph()
        p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(SYSTEM); r.bold = True; r.font.size = Pt(30); r.font.color.rgb = RGBColor(*NAVY)
        p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(PROJECT); r.font.size = Pt(14); r.font.color.rgb = RGBColor(*GREY)
        d.add_paragraph()
        p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(milestone); r.bold = True; r.font.size = Pt(20); r.font.color.rgb = RGBColor(*TEAL)
        p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(subtitle); r.font.size = Pt(13)
        p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(period); r.italic = True; r.font.size = Pt(11); r.font.color.rgb = RGBColor(*GREY)
        for _ in range(4):
            d.add_paragraph()
        t = d.add_table(rows=len(TEAM) + 1, cols=2)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.style = "Light List Accent 1"
        t.cell(0, 0).text, t.cell(0, 1).text = "Team member", "Roll no."
        for i, (n, r_) in enumerate(TEAM, 1):
            t.cell(i, 0).text, t.cell(i, 1).text = n, r_
        d.add_paragraph()
        p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run(f"Faculty mentor: {MENTOR}    |    Batch: {BATCH}").font.size = Pt(11)
        d.add_page_break()

    # Headings are numbered automatically ("3." / "3.2"); any number typed in the text is replaced.
    UNNUMBERED = ("References", "Appendix", "Abstract", "Contents", "Acknowledgement")

    def _strip(self, text):
        import re
        return re.sub(r"^\s*\d+(\.\d+)*\.?\s+", "", text)

    def h1(self, text):
        t = self._strip(text)
        if not t.startswith(self.UNNUMBERED):
            self._h1 = getattr(self, "_h1", 0) + 1; self._h2 = 0
            t = f"{self._h1}. {t}"
        if getattr(self, "_h1", 0) > 1 and getattr(self, "chapter_breaks", False) and not getattr(self, "_fresh_page", False):
            self.d.add_page_break()
        self._fresh_page = False
        h = self.d.add_heading(t, level=1)
        for r in h.runs:
            r.font.color.rgb = RGBColor(*NAVY)

    def h2(self, text):
        t = self._strip(text)
        if getattr(self, "_h1", 0) and not t.startswith(self.UNNUMBERED):
            self._h2 = getattr(self, "_h2", 0) + 1
            t = f"{self._h1}.{self._h2} {t}"
        h = self.d.add_heading(t, level=2)
        for r in h.runs:
            r.font.color.rgb = RGBColor(*TEAL)

    def p(self, text, bold_lead: str | None = None):
        para = self.d.add_paragraph()
        if bold_lead:
            para.add_run(bold_lead + " ").bold = True
        para.add_run(text)
        para.paragraph_format.space_after = Pt(6)
        return para

    def bullets(self, items):
        for it in items:
            para = self.d.add_paragraph(style="List Bullet")
            if isinstance(it, tuple):
                para.add_run(it[0] + " ").bold = True
                para.add_run(it[1])
            else:
                para.add_run(it)

    def table(self, df: pd.DataFrame, caption: str, index=False, font=9):
        self._tab += 1
        if index:
            df = df.reset_index()
        cap = self.d.add_paragraph()
        r = cap.add_run(f"Table {self._tab}. {caption}"); r.bold = True; r.font.size = Pt(10)
        t = self.d.add_table(rows=len(df) + 1, cols=len(df.columns))
        t.style = "Light Grid Accent 1"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for j, c in enumerate(df.columns):
            t.cell(0, j).text = str(c)
        for i, row in enumerate(df.itertuples(index=False), 1):
            for j, v in enumerate(row):
                t.cell(i, j).text = _fmt(v)
        for row in t.rows:
            for c in row.cells:
                for para in c.paragraphs:
                    for run in para.runs:
                        run.font.size = Pt(font)
        self.d.add_paragraph()

    def figure(self, path, caption, width_cm=15.5):
        self._fig += 1
        self.d.add_picture(path, width=Cm(width_cm))
        self.d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = self.d.add_paragraph(); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cap.add_run(f"Figure {self._fig}. {caption}"); r.italic = True; r.font.size = Pt(9.5)

    def code(self, text):
        para = self.d.add_paragraph()
        r = para.add_run(text); r.font.name = "Consolas"; r.font.size = Pt(9)
        shd = OxmlElement("w:shd"); shd.set(qn("w:fill"), "F2F5F8"); para._p.get_or_add_pPr().append(shd)

    def page_break(self):
        self.d.add_page_break()

    # ------------------------------------------------------------ richer report blocks
    def _field(self, para, code):
        r = para.add_run()
        b = OxmlElement("w:fldChar"); b.set(qn("w:fldCharType"), "begin"); r._r.append(b)
        t = OxmlElement("w:instrText"); t.set(qn("xml:space"), "preserve"); t.text = code; r._r.append(t)
        sep = OxmlElement("w:fldChar"); sep.set(qn("w:fldCharType"), "separate"); r._r.append(sep)
        r2 = para.add_run("1")
        e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), "end"); r2._r.append(e)
        return r2

    def header_footer(self, milestone):
        sec = self.d.sections[0]
        sec.different_first_page_header_footer = True
        hp = sec.header.paragraphs[0]
        hp.text = f"{SYSTEM}  |  {milestone}"; hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        for r in hp.runs:
            r.font.size = Pt(9); r.font.color.rgb = RGBColor(*GREY)
        fp = sec.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = fp.add_run(f"Batch {BATCH}  ·  Page "); r.font.size = Pt(9); r.font.color.rgb = RGBColor(*GREY)
        self._field(fp, "PAGE").font.size = Pt(9)

    def toc(self):
        h = self.d.add_paragraph()
        r = h.add_run("Contents"); r.bold = True; r.font.size = Pt(16); r.font.color.rgb = RGBColor(*NAVY)
        self._field(self.d.add_paragraph(), 'TOC \\o "1-2" \\h \\z \\u')
        self.d.add_page_break()

    def abstract(self, text, keywords=None):
        h = self.d.add_paragraph()
        r = h.add_run("Abstract"); r.bold = True; r.font.size = Pt(14); r.font.color.rgb = RGBColor(*NAVY)
        p = self.d.add_paragraph(text); p.paragraph_format.space_after = Pt(6)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if keywords:
            k = self.d.add_paragraph(); k.add_run("Keywords: ").bold = True; k.add_run(", ".join(keywords)).italic = True
        self.d.add_paragraph()

    def callout(self, text, title="Key point", fill="E6F4F2"):
        t = self.d.add_table(rows=1, cols=1); t.alignment = WD_TABLE_ALIGNMENT.CENTER
        c = t.cell(0, 0)
        shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:fill"), fill)
        c._tc.get_or_add_tcPr().append(shd)
        p = c.paragraphs[0]
        r = p.add_run(title + ": "); r.bold = True; r.font.color.rgb = RGBColor(*TEAL)
        lines = text if isinstance(text, list) else [text]
        p.add_run(lines[0])
        for ln in lines[1:]:
            q = c.add_paragraph(); q.add_run("•  " + ln)
        self.d.add_paragraph()

    def numbered(self, items):
        for k, it in enumerate(items, 1):
            para = self.d.add_paragraph()
            para.paragraph_format.left_indent = Cm(0.9); para.paragraph_format.first_line_indent = Cm(-0.6)
            para.paragraph_format.space_after = Pt(3)
            para.add_run(f"{k}.  ").bold = True
            if isinstance(it, tuple):
                para.add_run(it[0] + " ").bold = True
                para.add_run(it[1])
            else:
                para.add_run(it)

    def references(self, refs, title="References"):
        self.h1(title)
        for i, ref in enumerate(refs, 1):
            p = self.d.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.9); p.paragraph_format.first_line_indent = Cm(-0.9)
            p.paragraph_format.space_after = Pt(3)
            r = p.add_run(f"[{i}]  "); r.bold = True
            p.add_run(ref).font.size = Pt(10)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.d.save(path)


def _fmt(v):
    if isinstance(v, float):
        if v != v:
            return "–"
        if v == 0:
            return "0"
        return f"{v:.3f}" if abs(v) < 1 else f"{v:.1f}"
    return str(v)


# ----------------------------------------------------------------- slides (pptx)
MINT = (0x2D, 0xD4, 0xBF)
PALE = (0xE6, 0xF4, 0xF2)
WHITE = (0xFF, 0xFF, 0xFF)
INK = (0x33, 0x41, 0x55)


class Deck:
    """16:9 deck. Every content slide can take `takeaway` (key-point bar) and `say` (speaker notes)."""
    W, H = Inches(13.333), Inches(7.5)

    def __init__(self, milestone: str, subtitle: str, period: str):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = self.W, self.H
        self.blank = self.prs.slide_layouts[6]
        self.n = 0
        self.milestone = milestone.split(":")[0]
        self._title(milestone, subtitle, period)

    # ---------------------------------------------------------- primitives
    def _shape(self, s, kind, x, y, w, h, rgb, line=None):
        from pptx.enum.shapes import MSO_SHAPE
        sh = s.shapes.add_shape(getattr(MSO_SHAPE, kind), x, y, w, h)
        sh.fill.solid(); sh.fill.fore_color.rgb = PRGB(*rgb)
        if line:
            sh.line.color.rgb = PRGB(*line); sh.line.width = PPt(1)
        else:
            sh.line.fill.background()
        sh.shadow.inherit = False
        return sh

    def _rect(self, s, x, y, w, h, rgb):
        return self._shape(s, "RECTANGLE", x, y, w, h, rgb)

    def _text(self, s, x, y, w, h, text, size=18, bold=False, rgb=NAVY, align=PP_ALIGN.LEFT, anchor=None, spacing=None):
        from pptx.enum.text import MSO_ANCHOR
        tb = s.shapes.add_textbox(x, y, w, h)
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.05)
        if anchor:
            tf.vertical_anchor = getattr(MSO_ANCHOR, anchor)
        lines = text if isinstance(text, list) else [text]
        for i, ln in enumerate(lines):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.alignment = align
            if spacing:
                para.space_after = PPt(spacing)
            r = para.add_run(); r.text = ln; r.font.size = PPt(size); r.font.bold = bold
            r.font.color.rgb = PRGB(*rgb); r.font.name = "Calibri"
        return tb

    def _rich(self, tf, parts, size, first=True, align=PP_ALIGN.LEFT, space=6):
        """parts: list of (text, bold, rgb) runs forming one paragraph."""
        para = tf.paragraphs[0] if first else tf.add_paragraph()
        para.alignment = align
        para.space_after = PPt(space)
        for t, b, c in parts:
            r = para.add_run(); r.text = t; r.font.bold = b; r.font.size = PPt(size)
            r.font.color.rgb = PRGB(*c); r.font.name = "Calibri"
        return para

    def _title(self, milestone, subtitle, period):
        s = self.prs.slides.add_slide(self.blank)
        self._rect(s, 0, 0, self.W, self.H, NAVY)
        self._rect(s, Inches(9.4), 0, Inches(3.933), self.H, (0x18, 0x36, 0x5C))
        for k, (x, y, d) in enumerate([(10.2, 1.0, 1.6), (11.3, 2.9, 1.0), (10.0, 4.3, 2.2)]):
            self._shape(s, "OVAL", Inches(x), Inches(y), Inches(d), Inches(d), (0x1F, 0x4E, 0x6E) if k != 1 else TEAL)
        self._rect(s, Inches(0.8), Inches(2.55), Inches(1.2), Inches(0.08), MINT)
        self._text(s, Inches(0.8), Inches(1.2), Inches(8.5), Inches(1.2), SYSTEM, 46, True, WHITE)
        self._text(s, Inches(0.8), Inches(2.8), Inches(8.5), Inches(1.2), milestone, 30, True, MINT)
        self._text(s, Inches(0.8), Inches(4.0), Inches(8.3), Inches(0.9), subtitle, 19, False, (0xE2, 0xE8, 0xF0))
        self._text(s, Inches(0.8), Inches(4.9), Inches(8.3), Inches(0.5), period, 14, False, (0xA0, 0xAE, 0xC0))
        team = "   •   ".join(n for n, _ in TEAM)
        self._text(s, Inches(0.8), Inches(6.0), Inches(8.5), Inches(0.4), team, 13, False, (0xE2, 0xE8, 0xF0))
        self._text(s, Inches(0.8), Inches(6.4), Inches(8.5), Inches(0.7),
                   [f"Mentor: {MENTOR}   |   Batch {BATCH}", PROJECT], 11, False, (0xA0, 0xAE, 0xC0))

    def _base(self, title, kicker=None, takeaway=None, say=None):
        from pptx.enum.text import MSO_ANCHOR
        s = self.prs.slides.add_slide(self.blank)
        self.n += 1
        self._rect(s, 0, 0, Inches(0.18), self.H, TEAL)
        if kicker:
            self._text(s, Inches(0.6), Inches(0.3), Inches(12), Inches(0.4), kicker.upper(), 12, True, TEAL)
        self._text(s, Inches(0.6), Inches(0.6), Inches(12.2), Inches(0.9), title, 28, True, NAVY)
        self._rect(s, Inches(0.6), Inches(7.02), Inches(12.1), Inches(0.012), (0xD5, 0xDB, 0xE3))
        self._text(s, Inches(0.6), Inches(7.05), Inches(9), Inches(0.35), f"{SYSTEM}  ·  {self.milestone}", 10, False, GREY)
        self._text(s, Inches(11.95), Inches(7.05), Inches(0.75), Inches(0.35), str(self.n + 1), 10, True, GREY, PP_ALIGN.RIGHT)
        if takeaway:
            self._rect(s, Inches(0.6), Inches(6.22), Inches(12.1), Inches(0.66), PALE)
            self._rect(s, Inches(0.6), Inches(6.22), Inches(0.09), Inches(0.66), TEAL)
            tb = s.shapes.add_textbox(Inches(0.85), Inches(6.22), Inches(11.7), Inches(0.66))
            tb.text_frame.word_wrap = True
            tb.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
            self._rich(tb.text_frame, [("Key point:  ", True, TEAL), (takeaway, True, NAVY)], 15 if len(takeaway) < 120 else 13)
        if say:
            s.notes_slide.notes_text_frame.text = say if isinstance(say, str) else "\n".join(say)
        return s

    def _area(self, takeaway):
        """Content area (x, y, w, h) below the title and above the key-point bar."""
        return Inches(0.6), Inches(1.65), Inches(12.1), Inches(4.35 if takeaway else 5.15)

    # ---------------------------------------------------------- slide types
    def section(self, number, title, subtitle=""):
        s = self.prs.slides.add_slide(self.blank)
        self.n += 1
        self._rect(s, 0, 0, self.W, self.H, NAVY)
        self._rect(s, Inches(9.4), 0, Inches(3.933), self.H, (0x18, 0x36, 0x5C))
        self._text(s, Inches(0.9), Inches(1.4), Inches(4), Inches(2), f"{number:02d}" if isinstance(number, int) else str(number),
                   96, True, MINT)
        self._rect(s, Inches(0.95), Inches(3.55), Inches(1.0), Inches(0.07), MINT)
        self._text(s, Inches(0.9), Inches(3.8), Inches(8.3), Inches(1), title, 36, True, WHITE)
        if subtitle:
            self._text(s, Inches(0.9), Inches(4.75), Inches(8.3), Inches(1.5), subtitle, 18, False, (0xC8, 0xD3, 0xE0))
        return s

    def agenda(self, items, title="Agenda", say=None):
        s = self._base(title, "Overview", say=say)
        x0, y0, w, h = self._area(None)
        half = (len(items) + 1) // 2
        rh = min(Inches(1.55), int(h / max(half, 1)))
        for i, it in enumerate(items):
            col, row = divmod(i, half)
            x = x0 + col * Inches(6.15); y = y0 + row * rh
            self._shape(s, "OVAL", x, y + Inches(0.05), Inches(0.75), Inches(0.75), TEAL if col == 0 else NAVY)
            self._text(s, x, y + Inches(0.05), Inches(0.75), Inches(0.75), f"{i + 1}", 22, True, WHITE, PP_ALIGN.CENTER, "MIDDLE")
            head, sub = (it if isinstance(it, tuple) else (it, ""))
            tb = s.shapes.add_textbox(x + Inches(0.95), y + Inches(0.02), Inches(5.1), rh)
            tb.text_frame.word_wrap = True
            self._rich(tb.text_frame, [(head, True, NAVY)], 22, space=2)
            if sub:
                self._rich(tb.text_frame, [(sub, False, GREY)], 15, first=False)
        return s

    def cards(self, title, items, kicker=None, cols=None, takeaway=None, say=None, icons=True):
        """Grid of content cards. items: (heading, body) or (heading, body, tag); body may be a list of lines."""
        s = self._base(title, kicker, takeaway, say)
        x0, y0, w, h = self._area(takeaway)
        n = len(items)
        cols = cols or (n if n <= 3 else 2 if n == 4 else 3)
        rows = -(-n // cols)
        gap = Inches(0.22)
        cw = int((w - gap * (cols - 1)) / cols); ch = int((h - gap * (rows - 1)) / rows)
        chi = ch / Inches(1)
        words = max(sum(len(" ".join(it[1]) if isinstance(it[1], list) else str(it[1])) for it in items) / n, 1)
        if chi >= 2.0 and words < 90:
            hs, bs = (22, 18) if cols <= 3 else (20, 16)
        elif chi >= 1.5:
            hs, bs = (19, 15) if cols <= 3 else (18, 14)
        else:
            hs, bs = 16, 13
        for i, it in enumerate(items):
            r_, c_ = divmod(i, cols)
            x = x0 + c_ * (cw + gap); y = y0 + r_ * (ch + gap)
            self._shape(s, "RECTANGLE", x, y, cw, ch, LIGHT)
            self._rect(s, x, y, cw, Inches(0.07), TEAL if (i % 2 == 0) else NAVY)
            tag = it[2] if len(it) > 2 else (f"{i + 1:02d}" if icons else None)
            tx = x + Inches(0.22)
            if tag:
                self._text(s, tx, y + Inches(0.16), cw - Inches(0.4), Inches(0.4), str(tag), 12, True, TEAL)
            tb = s.shapes.add_textbox(tx, y + Inches(0.48 if tag else 0.25), cw - Inches(0.4), ch - Inches(0.6))
            tf = tb.text_frame; tf.word_wrap = True
            if words < 70 and chi >= 1.6:
                from pptx.enum.text import MSO_ANCHOR
                tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            self._rich(tf, [(it[0], True, NAVY)], hs, space=6)
            body = it[1] if isinstance(it[1], list) else [it[1]]
            for ln in body:
                if ln:
                    self._rich(tf, [(ln, False, INK)], bs, first=False, space=4)
        return s

    def bullets(self, title, items, kicker=None, size=20, takeaway=None, say=None):
        """Tuples become a card grid; plain strings become a numbered list that fills the slide."""
        if items and all(isinstance(it, tuple) for it in items):
            return self.cards(title, items, kicker, takeaway=takeaway, say=say)
        s = self._base(title, kicker, takeaway, say)
        x0, y0, w, h = self._area(takeaway)
        main = [it for it in items if not (isinstance(it, str) and it.startswith("  "))]
        rh = min(Inches(1.05), int(h / max(len(main), 1)))
        y = y0; k = 0
        for it in items:
            if isinstance(it, str) and it.startswith("  "):
                self._text(s, x0 + Inches(1.0), y - Inches(0.45), w - Inches(1.2), Inches(0.4), it.strip(), size - 5, False, GREY)
                continue
            k += 1
            self._rect(s, x0, y + Inches(0.06), Inches(0.08), rh - Inches(0.22), TEAL)
            self._text(s, x0 + Inches(0.25), y, Inches(0.7), rh - Inches(0.1), f"{k:02d}", size, True, TEAL)
            if isinstance(it, tuple):
                tb = s.shapes.add_textbox(x0 + Inches(1.0), y, w - Inches(1.2), rh - Inches(0.1)); tb.text_frame.word_wrap = True
                self._rich(tb.text_frame, [(it[0] + "  ", True, NAVY), (it[1], False, INK)], size)
            else:
                self._text(s, x0 + Inches(1.0), y, w - Inches(1.2), rh - Inches(0.1), it, size, False, NAVY)
            y += rh
        return s

    def stats(self, title, items, kicker=None, takeaway=None, say=None, points=None):
        """Big-number cards: items = (value, label, detail). Optional `points` under the cards."""
        s = self._base(title, kicker, takeaway, say)
        x0, y0, w, h = self._area(takeaway)
        n = len(items); gap = Inches(0.22)
        cw = int((w - gap * (n - 1)) / n)
        ch = Inches(2.35) if points else min(h, Inches(3.3))
        for i, (val, lab, det) in enumerate(items):
            x = x0 + i * (cw + gap)
            self._shape(s, "RECTANGLE", x, y0, cw, ch, NAVY if i % 2 == 0 else TEAL)
            self._text(s, x, y0 + Inches(0.18), cw, Inches(0.9), str(val), 40 if n <= 4 else 32, True, WHITE, PP_ALIGN.CENTER)
            self._text(s, x + Inches(0.1), y0 + Inches(1.1), cw - Inches(0.2), Inches(0.5), lab, 15, True,
                       MINT if i % 2 == 0 else WHITE, PP_ALIGN.CENTER)
            if det:
                self._text(s, x + Inches(0.15), y0 + Inches(1.55), cw - Inches(0.3), ch - Inches(1.6), det, 12, False,
                           (0xE2, 0xE8, 0xF0), PP_ALIGN.CENTER)
        if points:
            tb = s.shapes.add_textbox(x0, y0 + ch + Inches(0.3), w, h - ch - Inches(0.3))
            tf = tb.text_frame; tf.word_wrap = True
            for j, p in enumerate(points):
                parts = [("▪ ", True, TEAL)] + ([(p[0] + "  ", True, NAVY), (p[1], False, INK)] if isinstance(p, tuple) else [(p, False, NAVY)])
                self._rich(tf, parts, 16, first=(j == 0), space=8)
        return s

    def flow(self, title, steps, kicker=None, takeaway=None, say=None, note=None):
        """Process diagram: steps = (name, detail). Boxes joined by arrows, detail text under each."""
        s = self._base(title, kicker, takeaway, say)
        x0, y0, w, h = self._area(takeaway)
        n = len(steps); arrow = Inches(0.42)
        bw = int((w - arrow * (n - 1)) / n); bh = Inches(1.2)
        by = y0 + Inches(0.2)
        for i, (name, det) in enumerate(steps):
            x = x0 + i * (bw + arrow)
            self._shape(s, "ROUNDED_RECTANGLE", x, by, bw, bh, NAVY if i % 2 == 0 else TEAL)
            self._text(s, x + Inches(0.08), by, bw - Inches(0.16), bh, name, 16 if n <= 5 else 14, True, WHITE, PP_ALIGN.CENTER, "MIDDLE")
            if i < n - 1:
                self._shape(s, "RIGHT_ARROW", x + bw + Inches(0.06), by + int(bh / 2) - Inches(0.17), arrow - Inches(0.12), Inches(0.34), MINT)
            det_l = det if isinstance(det, list) else [det]
            dh = min(h - bh - Inches(0.45) - (Inches(0.45) if note else 0), Inches(2.6))
            self._shape(s, "RECTANGLE", x, by + bh + Inches(0.18), bw, dh, LIGHT)
            tb = s.shapes.add_textbox(x + Inches(0.1), by + bh + Inches(0.3), bw - Inches(0.2), dh - Inches(0.2))
            tf = tb.text_frame; tf.word_wrap = True
            for j, d in enumerate(det_l):
                self._rich(tf, [(d, False, INK)], 16 if n <= 4 else 15 if n == 5 else 13, first=(j == 0), space=6)
        if note:
            self._text(s, x0, y0 + h - Inches(0.4), w, Inches(0.4), note, 13, False, GREY, PP_ALIGN.CENTER)
        return s

    def image(self, title, path, caption=None, kicker=None, notes: list | None = None, takeaway=None, say=None):
        s = self._base(title, kicker, takeaway, say)
        from PIL import Image
        x0, y0, w, h = self._area(takeaway)
        iw, ih = Image.open(path).size
        maxw = Inches(8.2) if notes else w
        maxh = h - (Inches(0.4) if caption else 0)
        sc = min(maxw / iw, maxh / ih)
        pw, ph = int(iw * sc), int(ih * sc)
        x = x0 if notes else int((self.W - pw) / 2)
        s.shapes.add_picture(path, x, y0, pw, ph)
        if caption:
            self._text(s, x, y0 + ph + Inches(0.02), pw, Inches(0.4), caption, 12, False, GREY)
        if notes:
            self._side_notes(s, notes, x0 + Inches(8.45), y0, Inches(3.65), h)
        return s

    def _side_notes(self, s, notes, x, y, w, h):
        self._shape(s, "RECTANGLE", x, y, w, h, LIGHT)
        self._rect(s, x, y, w, Inches(0.07), TEAL)
        tb = s.shapes.add_textbox(x + Inches(0.18), y + Inches(0.25), w - Inches(0.3), h - Inches(0.4))
        tf = tb.text_frame; tf.word_wrap = True
        total = sum(len(" ".join(n) if isinstance(n, tuple) else str(n)) for n in notes)
        sz = 15 if total < 300 else 13 if total < 480 else 12
        for i, n in enumerate(notes):
            parts = [("▪ ", True, TEAL)] + ([(n[0] + " ", True, NAVY), (n[1], False, INK)] if isinstance(n, tuple) else [(n, False, NAVY)])
            self._rich(tf, parts, sz, first=(i == 0), space=10)

    def images(self, title, paths, captions=None, kicker=None, takeaway=None, say=None):
        """Two or three screenshots side by side."""
        s = self._base(title, kicker, takeaway, say)
        from PIL import Image
        x0, y0, w, h = self._area(takeaway)
        n = len(paths); gap = Inches(0.25)
        cw = int((w - gap * (n - 1)) / n); maxh = h - Inches(0.45)
        for i, p in enumerate(paths):
            iw, ih = Image.open(p).size
            sc = min(cw / iw, maxh / ih); pw, ph = int(iw * sc), int(ih * sc)
            x = x0 + i * (cw + gap) + int((cw - pw) / 2)
            pic = s.shapes.add_picture(p, x, y0, pw, ph)
            pic.line.color.rgb = PRGB(0xD5, 0xDB, 0xE3)
            if captions:
                self._text(s, x0 + i * (cw + gap), y0 + ph + Inches(0.05), cw, Inches(0.4), captions[i], 13, True, NAVY, PP_ALIGN.CENTER)
        return s

    def table(self, title, df: pd.DataFrame, kicker=None, notes: list | None = None, font=14, col_w=None, takeaway=None, say=None):
        s = self._base(title, kicker, takeaway, say)
        x0, y0, w, h = self._area(takeaway)
        rows, cols = len(df) + 1, len(df.columns)
        width = Inches(8.2) if notes else w
        rh = min(Inches(0.75 if rows <= 6 else 0.6), int(h / rows))
        if rows <= 7 and font < 16 and cols <= 5:
            font = 15 if notes else 16
        gt = s.shapes.add_table(rows, cols, x0, y0, width, rh * rows).table
        for i in range(rows):
            gt.rows[i].height = rh
        for j, c in enumerate(df.columns):
            cell = gt.cell(0, j); cell.text = str(c)
            cell.fill.solid(); cell.fill.fore_color.rgb = PRGB(*NAVY)
            for para in cell.text_frame.paragraphs:
                for r in para.runs:
                    r.font.size = PPt(font); r.font.bold = True; r.font.color.rgb = PRGB(255, 255, 255)
        for i, row in enumerate(df.itertuples(index=False), 1):
            for j, v in enumerate(row):
                cell = gt.cell(i, j); cell.text = _fmt(v)
                cell.fill.solid(); cell.fill.fore_color.rgb = PRGB(*(LIGHT if i % 2 else (255, 255, 255)))
                for para in cell.text_frame.paragraphs:
                    for r in para.runs:
                        r.font.size = PPt(font); r.font.color.rgb = PRGB(*NAVY); r.font.bold = (j == 0)
        if col_w:
            tot = sum(col_w); scale = (width / Inches(1)) / tot
            for j, cw in enumerate(col_w):
                gt.columns[j].width = Inches(cw * scale)
        if notes:
            self._side_notes(s, notes, x0 + Inches(8.45), y0, Inches(3.65), h)
        return s

    def closing(self, title="Thank you", lines=None):
        s = self.prs.slides.add_slide(self.blank)
        self.n += 1
        self._rect(s, 0, 0, self.W, self.H, NAVY)
        self._text(s, Inches(0.9), Inches(2.3), Inches(11.5), Inches(1.2), title, 48, True, WHITE)
        self._rect(s, Inches(0.95), Inches(3.45), Inches(1.2), Inches(0.08), MINT)
        self._text(s, Inches(0.9), Inches(3.7), Inches(11.5), Inches(2.2), lines or ["Questions and discussion"], 20, False,
                   (0xC8, 0xD3, 0xE0), spacing=6)
        self._text(s, Inches(0.9), Inches(6.3), Inches(11.8), Inches(0.5),
                   "   •   ".join(n for n, _ in TEAM) + f"   |   Mentor: {MENTOR}", 13, False, (0xA0, 0xAE, 0xC0))
        return s

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.prs.save(path)
