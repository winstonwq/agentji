---
name: docx-template
description: >
  Python/python-docx template for producing branded Word documents. Provides
  brand color constants, four helper functions (set_cell_shading, add_table,
  add_heading_styled, add_callout), document setup (margins, cover page,
  page breaks), and matplotlib chart patterns. Use when building a Python
  script that generates a .docx report.
---

# docx-template

Use this template when writing a Python script that generates a Word document.
All data must be read from files at runtime — never hardcode values from the
analyst findings as string literals in the script.

## Required pattern

```python
findings_path = "<path passed in brief>"
findings_text = open(findings_path, encoding='utf-8').read()
# Parse findings_text to extract tables, numbers, and narrative
```

Parse the markdown: split on `##` headings, extract pipe-table rows with
`.split('|')`, strip whitespace. Do not truncate — read and use all rows.

---

## Font selection

Detect the primary language of the findings file and set fonts accordingly.

```python
def is_cjk(text):
    """Return True if the text contains significant CJK content."""
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return cjk / max(len(text), 1) > 0.05

findings_text = open(findings_path, encoding='utf-8').read()

if is_cjk(findings_text):
    FONT_HEADING = '微软雅黑'   # Microsoft YaHei — clean, modern, works on Windows/Office
    FONT_BODY    = '微软雅黑'
else:
    FONT_HEADING = 'Arial'
    FONT_BODY    = 'Arial'
```

Apply when adding runs:
```python
run.font.name = FONT_HEADING   # for headings
run.font.name = FONT_BODY      # for body text
```

For CJK documents, also set the East Asian font via XML (python-docx doesn't expose this directly):
```python
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

def set_run_cjk_font(run, font_name):
    """Set East Asian font on a run for proper CJK rendering in Word."""
    rPr = run._r.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.insert(0, rFonts)
    rFonts.set(qn('w:eastAsia'), font_name)
    rFonts.set(qn('w:hint'), 'eastAsia')
```

Call `set_run_cjk_font(run, FONT_HEADING)` after setting `run.font.name` for any run that contains Chinese text.

---

## Brand colors

```python
# python-docx (RGBColor)
from docx.shared import RGBColor
DARK       = RGBColor(0x14, 0x14, 0x13)
ORANGE     = RGBColor(0xd9, 0x77, 0x57)
BLUE       = RGBColor(0x6a, 0x9b, 0xcc)
GREEN      = RGBColor(0x78, 0x8c, 0x5d)
MID_GRAY   = RGBColor(0xb0, 0xae, 0xa5)
LIGHT_GRAY = RGBColor(0xe8, 0xe6, 0xdc)

# matplotlib (hex strings)
DARK_HEX       = '#141413'
LIGHT_HEX      = '#faf9f5'
MID_GRAY_HEX   = '#b0aea5'
LIGHT_GRAY_HEX = '#e8e6dc'
ORANGE_HEX     = '#d97757'
BLUE_HEX       = '#6a9bcc'
GREEN_HEX      = '#788c5d'
```

---

## Helper functions — copy verbatim into your script

```python
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

def set_cell_shading(cell, color_hex):
    """Set cell background color. color_hex is a 6-char hex string, e.g. 'd97757'."""
    shading = OxmlElement('w:shd')
    shading.set(qn('w:fill'), color_hex)
    cell._tc.get_or_add_tcPr().append(shading)

def add_table(doc, headers, rows):
    """
    Add a styled table.
    - headers: list of column header strings
    - rows: list of lists (each inner list is one row of cell values)
    Orange header row, alternating light-gray shading on data rows.
    """
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'

    # Header row — orange background, white bold text
    header_row = table.rows[0]
    for i, header in enumerate(headers):
        cell = header_row.cells[i]
        cell.text = header
        set_cell_shading(cell, 'd97757')
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                run.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
                run.font.size = Pt(10)

    # Data rows — alternating shading
    for row_idx, row_data in enumerate(rows):
        row = table.rows[row_idx + 1]
        for col_idx, cell_data in enumerate(row_data):
            cell = row.cells[col_idx]
            cell.text = str(cell_data)
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for run in paragraph.runs:
                    run.font.size = Pt(9)
                    run.font.color.rgb = DARK
            if row_idx % 2 == 1:
                set_cell_shading(cell, 'f5f5f5')

    return table

def add_heading_styled(doc, text, level=1):
    """
    Add a heading. Level 1 = orange, 18pt. Level 2 = blue, 14pt.
    """
    heading = doc.add_heading(text, level=level)
    for run in heading.runs:
        if level == 1:
            run.font.color.rgb = ORANGE
            run.font.size = Pt(18)
        else:
            run.font.color.rgb = BLUE
            run.font.size = Pt(14)
    return heading

def add_callout(doc, text):
    """
    Add an indented italic insight/callout paragraph (gray, 10pt).
    Use for key insights after tables.
    """
    para = doc.add_paragraph()
    para.paragraph_format.left_indent = Inches(0.5)
    para.paragraph_format.space_before = Pt(6)
    para.paragraph_format.space_after = Pt(12)
    run = para.add_run(text)
    run.italic = True
    run.font.color.rgb = MID_GRAY
    run.font.size = Pt(10)
    return para
```

---

## Document setup

```python
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from datetime import datetime

doc = Document()

# Margins: 0.75" top/bottom, 1" left/right
for section in doc.sections:
    section.top_margin    = Inches(0.75)
    section.bottom_margin = Inches(0.75)
    section.left_margin   = Inches(1)
    section.right_margin  = Inches(1)
```

---

## Cover page pattern

```python
# Two blank lines before title
doc.add_paragraph()
doc.add_paragraph()

title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title_run = title.add_run("<Report Title>")
title_run.bold = True
title_run.font.size = Pt(32)
title_run.font.color.rgb = ORANGE

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub_run = subtitle.add_run("<Subtitle>")
sub_run.font.size = Pt(24)
sub_run.font.color.rgb = DARK

doc.add_paragraph()

date_para = doc.add_paragraph()
date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
date_run = date_para.add_run(datetime.now().strftime("%B %Y"))
date_run.font.size = Pt(14)
date_run.font.color.rgb = MID_GRAY

doc.add_paragraph()
prep = doc.add_paragraph()
prep.alignment = WD_ALIGN_PARAGRAPH.CENTER
prep.add_run("Prepared by agentji").font.color.rgb = MID_GRAY

doc.add_page_break()
```

---

## Chart patterns (matplotlib)

### Horizontal bar chart

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
fig.patch.set_facecolor(LIGHT_HEX)
ax.set_facecolor(LIGHT_HEX)

labels  = [...]   # list of strings, parsed from findings
values  = [...]   # list of floats, parsed from findings
colors  = [ORANGE_HEX, BLUE_HEX, GREEN_HEX, MID_GRAY_HEX]  # cycle as needed

y_pos = np.arange(len(labels))
bar_colors = [colors[i % len(colors)] for i in range(len(labels))]
bars = ax.barh(y_pos, values, color=bar_colors, edgecolor=DARK_HEX, linewidth=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=11, color=DARK_HEX)
ax.invert_yaxis()
ax.set_title('<Chart Title>', fontsize=16, color=DARK_HEX, fontweight='bold', pad=20)

# Value labels on bars
for bar, val in zip(bars, values):
    ax.text(bar.get_width() + max(values) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f'{val}', va='center', fontsize=9, color=DARK_HEX)

ax.set_xlim(0, max(values) * 1.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(MID_GRAY_HEX)
ax.spines['bottom'].set_color(MID_GRAY_HEX)
ax.tick_params(colors=DARK_HEX)

plt.tight_layout()
plt.savefig('<chart_path>.png', dpi=150, facecolor=LIGHT_HEX, bbox_inches='tight')
plt.close()
```

### Grouped bar chart

```python
fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
fig.patch.set_facecolor(LIGHT_HEX)
ax.set_facecolor(LIGHT_HEX)

group_labels = [...]   # x-axis groups, parsed from findings
series = {             # dict of series_name → list of values
    'Series A': [...],
    'Series B': [...],
}
colors = [ORANGE_HEX, BLUE_HEX, GREEN_HEX, MID_GRAY_HEX]

x = np.arange(len(group_labels))
width = 0.8 / len(series)

for idx, (name, vals) in enumerate(series.items()):
    offset = width * idx
    ax.bar(x + offset, vals, width, label=name,
           color=colors[idx % len(colors)], edgecolor=DARK_HEX, linewidth=0.5)

ax.set_xticks(x + width * (len(series) - 1) / 2)
ax.set_xticklabels(group_labels, fontsize=10, color=DARK_HEX)
ax.set_title('<Chart Title>', fontsize=16, color=DARK_HEX, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('<chart_path>.png', dpi=150, facecolor=LIGHT_HEX, bbox_inches='tight')
plt.close()
```

### Embed chart in document

```python
doc.add_picture('<chart_path>.png', width=Inches(5.5))
doc.add_paragraph()
```

---

## Section pattern

```python
add_heading_styled(doc, "Section Title", 1)
add_heading_styled(doc, "Subsection Title", 2)

# Narrative paragraph
p = doc.add_paragraph()
p.add_run("Bold label: ").bold = True
p.add_run("Body text here.")

# Table from parsed findings data
add_table(doc, headers, rows)
doc.add_paragraph()

# Key insight
add_callout(doc, "Insight: ...")

doc.add_page_break()
```

---

## Save

```python
output_path = "<path from brief>"
doc.save(output_path)
print(f"Saved: {output_path}")
```
