"""Build report/thesis_report.docx from report/report.md (no pandoc on this machine).

Supports the subset report.md uses: # title, ## / ### headings, wrapped paragraphs, nested "- "
bullets and "1. " lists with indented continuation lines, pipe tables, ![caption](image) figures
(auto-numbered), and inline **bold**, *italic*, `code` and \\* escapes.

    /opt/anaconda3/envs/ddd-repo/bin/python report/build_docx.py
"""
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
SRC, OUT = HERE / "report.md", HERE / "thesis_report.docx"
INK = RGBColor(0x1F, 0x24, 0x30)
TOKEN = re.compile(r"(\*\*.+?\*\*|`[^`]+`|(?<![\\\w])\*(?!\s).+?(?<![\s\\])\*)")


def add_inline(par, text, size=None, bold=False):
    text = text.replace("\\*", "\ue000")
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run, val = par.add_run(part[2:-2]), dict(bold=True)
        elif part.startswith("`") and part.endswith("`"):
            run, val = par.add_run(part[1:-1]), dict(code=True)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            run, val = par.add_run(part[1:-1]), dict(italic=True)
        else:
            run, val = par.add_run(part), {}
        run.text = run.text.replace("\ue000", "*")
        run.bold = bold or val.get("bold", False)
        run.italic = val.get("italic", False)
        if val.get("code"):
            run.font.name = "Menlo"
            run.font.size = Pt(8.5)
        elif size:
            run.font.size = Pt(size)


def shade(cell, hex_fill):
    tc = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), hex_fill)
    tc.append(sh)


def add_table(doc, rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(re.fullmatch(r":?-{3,}:?", c) for c in r)]
    t = doc.add_table(rows=len(cells), cols=len(cells[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, r in enumerate(cells):
        for j, c in enumerate(r[:len(cells[0])]):
            cell = t.cell(i, j)
            cell.text = ""
            add_inline(cell.paragraphs[0], c, size=8.5, bold=(i == 0))
            if i == 0:
                shade(cell, "E8EDF5")
    doc.add_paragraph()


def collect_items(lines, i):
    """Bullet / numbered items starting at lines[i]; returns ([(level, kind, text)], next index)."""
    items = []
    while i < len(lines):
        ln = lines[i]
        m = re.match(r"^(\s*)(- |\d+\. )(.*)$", ln)
        if m:
            items.append([len(m.group(1)) // 2, "num" if m.group(2)[0].isdigit() else "bul", m.group(3)])
        elif ln.strip() and ln.startswith("  ") and items:
            items[-1][2] += " " + ln.strip()
        else:
            break
        i += 1
    return items, i


def main():
    lines = SRC.read_text().splitlines()
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)
    for s in ["Heading 1", "Heading 2", "Title"]:
        doc.styles[s].font.color.rgb = INK
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Inches(1)
    fig_n, i = 0, 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        if ln.startswith("# "):
            p = doc.add_paragraph(ln[2:].strip(), style="Title")
            p.runs[0].font.size = Pt(20)
            i += 1
        elif ln.startswith("## "):
            doc.add_heading(ln[3:].strip(), level=1)
            i += 1
        elif ln.startswith("### "):
            doc.add_heading(ln[4:].strip(), level=2)
            i += 1
        elif ln.startswith("!["):
            m = re.match(r"!\[(.*)\]\((.*)\)", ln.strip())
            fig_n += 1
            img = (HERE / m.group(2)).resolve()
            doc.add_picture(str(img), width=Inches(6.4))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap = doc.add_paragraph()
            add_inline(cap, f"**Figure {fig_n}.** " + m.group(1), size=9)
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            i += 1
        elif ln.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(lines[i])
                i += 1
            add_table(doc, rows)
        elif re.match(r"^\s*(- |\d+\. )", ln):
            items, i = collect_items(lines, i)
            for level, kind, text in items:
                style = ("List Number" if kind == "num" else "List Bullet") + (" 2" if level >= 1 else "")
                if "|" in text and text.strip().startswith("|"):
                    continue
                add_inline(doc.add_paragraph(style=style), text)
            # an indented table directly under a bullet
            while i < len(lines) and lines[i].strip() == "":
                if i + 1 < len(lines) and lines[i + 1].strip().startswith("|") and lines[i + 1].startswith(" "):
                    i += 1
                    rows = []
                    while i < len(lines) and lines[i].strip().startswith("|"):
                        rows.append(lines[i])
                        i += 1
                    add_table(doc, rows)
                else:
                    break
        else:
            buf = [ln.strip()]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#|!\[|\s*\||\s*- |\s*\d+\. )", lines[i]):
                buf.append(lines[i].strip())
                i += 1
            p = doc.add_paragraph()
            add_inline(p, " ".join(buf))
            if buf[0].startswith("z = "):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.save(OUT)
    print(f"wrote {OUT} ({fig_n} figures)")


if __name__ == "__main__":
    main()
