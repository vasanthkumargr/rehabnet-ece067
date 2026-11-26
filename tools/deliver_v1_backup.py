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

    def h1(self, text):
        h = self.d.add_heading(text, level=1)
        for r in h.runs:
            r.font.color.rgb = RGBColor(*NAVY)

    def h2(self, text):
        h = self.d.add_heading(text, level=2)
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
class Deck:
    W, H = Inches(13.333), Inches(7.5)

    def __init__(self, milestone: str, subtitle: str, period: str):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = self.W, self.H
        self.blank = self.prs.slide_layouts[6]
        self.n = 0
        self._title(milestone, subtitle, period)

    def _rect(self, s, x, y, w, h, rgb):
        from pptx.enum.shapes import MSO_SHAPE
        sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
        sh.fill.solid(); sh.fill.fore_color.rgb = PRGB(*rgb); sh.line.fill.background()
        return sh

    def _text(self, s, x, y, w, h, text, size=18, bold=False, rgb=NAVY, align=PP_ALIGN.LEFT):
        tb = s.shapes.add_textbox(x, y, w, h)
        tf = tb.text_frame; tf.word_wrap = True
        lines = text if isinstance(text, list) else [text]
        for i, ln in enumerate(lines):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.alignment = align
            r = para.add_run(); r.text = ln; r.font.size = PPt(size); r.font.bold = bold
            r.font.color.rgb = PRGB(*rgb); r.font.name = "Calibri"
        return tb

    def _title(self, milestone, subtitle, period):
        s = self.prs.slides.add_slide(self.blank)
        self._rect(s, 0, 0, self.W, self.H, NAVY)
        self._rect(s, Inches(0.8), Inches(2.55), Inches(1.2), Inches(0.08), (0x2D, 0xD4, 0xBF))
        self._text(s, Inches(0.8), Inches(1.2), Inches(11.5), Inches(1.2), SYSTEM, 44, True, (255, 255, 255))
        self._text(s, Inches(0.8), Inches(2.8), Inches(11.5), Inches(0.8), milestone, 30, True, (0x2D, 0xD4, 0xBF))
        self._text(s, Inches(0.8), Inches(3.6), Inches(11.5), Inches(0.9), subtitle, 20, False, (0xE2, 0xE8, 0xF0))
        self._text(s, Inches(0.8), Inches(4.4), Inches(11.5), Inches(0.5), period, 14, False, (0xA0, 0xAE, 0xC0))
        team = "   •   ".join(n for n, _ in TEAM)
        self._text(s, Inches(0.8), Inches(6.1), Inches(11.8), Inches(0.4), team, 13, False, (0xE2, 0xE8, 0xF0))
        self._text(s, Inches(0.8), Inches(6.5), Inches(11.8), Inches(0.4),
                   f"Mentor: {MENTOR}   |   Batch {BATCH}   |   {PROJECT}", 11, False, (0xA0, 0xAE, 0xC0))

    def _base(self, title, kicker=None):
        s = self.prs.slides.add_slide(self.blank)
        self.n += 1
        self._rect(s, 0, 0, Inches(0.18), self.H, TEAL)
        if kicker:
            self._text(s, Inches(0.6), Inches(0.3), Inches(12), Inches(0.4), kicker.upper(), 12, True, TEAL)
        self._text(s, Inches(0.6), Inches(0.6), Inches(12.2), Inches(0.9), title, 28, True, NAVY)
        self._text(s, Inches(12.3), Inches(7.0), Inches(0.9), Inches(0.4), str(self.n + 1), 11, False, GREY, PP_ALIGN.RIGHT)
        return s

    def bullets(self, title, items, kicker=None, size=20):
        s = self._base(title, kicker)
        tb = s.shapes.add_textbox(Inches(0.7), Inches(1.7), Inches(12), Inches(5.2))
        tf = tb.text_frame; tf.word_wrap = True
        for i, it in enumerate(items):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            sub = isinstance(it, str) and it.startswith("  ")
            para.level = 1 if sub else 0
            para.space_after = PPt(10)
            if isinstance(it, tuple):
                r = para.add_run(); r.text = "▪ " + it[0] + "  "; r.font.bold = True
                r.font.size = PPt(size); r.font.color.rgb = PRGB(*TEAL)
                r2 = para.add_run(); r2.text = it[1]; r2.font.size = PPt(size); r2.font.color.rgb = PRGB(*NAVY)
            else:
                r = para.add_run(); r.text = ("– " if sub else "▪ ") + it.strip()
                r.font.size = PPt(size - (3 if sub else 0)); r.font.color.rgb = PRGB(*(GREY if sub else NAVY))
        return s

    def image(self, title, path, caption=None, kicker=None, notes: list[str] | None = None):
        s = self._base(title, kicker)
        from PIL import Image
        iw, ih = Image.open(path).size
        maxw = Inches(8.4 if notes else 12.0)
        maxh = Inches(5.0)
        sc = min(maxw / iw, maxh / ih)
        w, h = int(iw * sc), int(ih * sc)
        x = Inches(0.6) if notes else int((self.W - w) / 2)
        s.shapes.add_picture(path, x, Inches(1.6), w, h)
        if caption:
            self._text(s, x, Inches(1.65) + h, w, Inches(0.4), caption, 12, False, GREY)
        if notes:
            tb = s.shapes.add_textbox(Inches(9.3), Inches(1.7), Inches(3.7), Inches(5))
            tf = tb.text_frame; tf.word_wrap = True
            for i, n in enumerate(notes):
                para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                para.space_after = PPt(12)
                r = para.add_run(); r.text = "▪ " + n; r.font.size = PPt(15); r.font.color.rgb = PRGB(*NAVY)
        return s

    def table(self, title, df: pd.DataFrame, kicker=None, notes: list[str] | None = None, font=14, col_w=None):
        s = self._base(title, kicker)
        rows, cols = len(df) + 1, len(df.columns)
        width = Inches(8.4 if notes else 12.0)
        gt = s.shapes.add_table(rows, cols, Inches(0.6), Inches(1.7), width, Inches(0.42) * rows).table
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
                        r.font.size = PPt(font); r.font.color.rgb = PRGB(*NAVY)
        if col_w:
            for j, w in enumerate(col_w):
                gt.columns[j].width = Inches(w)
        if notes:
            tb = s.shapes.add_textbox(Inches(9.3), Inches(1.7), Inches(3.7), Inches(5))
            tf = tb.text_frame; tf.word_wrap = True
            for i, n in enumerate(notes):
                para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                para.space_after = PPt(12)
                r = para.add_run(); r.text = "▪ " + n; r.font.size = PPt(15); r.font.color.rgb = PRGB(*NAVY)
        return s

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.prs.save(path)
