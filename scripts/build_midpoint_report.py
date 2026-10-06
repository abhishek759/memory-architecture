"""Build standalone evidence figures and a PDF from the authored midpoint Markdown.

Install requirements-report.txt separately. This script does not run experiments.
The report is an authored snapshot: review its claims after refreshing results.
"""
import argparse
import html
import json
import re
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                               Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]


def figures(report, output):
    output.mkdir(parents=True, exist_ok=True)
    names = ['raw_history', 'vector_retrieval', 'running_summary']
    labels = ['Raw history', 'Vector retrieval', 'Running summary']
    palette = ['#28536B', '#238A87', '#B56B34']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold'})
    fig, ax = plt.subplots(figsize=(8, 2.7), layout='constrained')
    values = [report['approaches'][n]['successful'] for n in names]
    ax.barh(labels[::-1], values[::-1], color=palette[::-1], height=.55)
    for i, n in enumerate(names[::-1]):
        r = report['approaches'][n]
        ax.text(r['successful'] + .15, i, f"{r['successful']}/{r['expected']}", va='center')
    ax.set(xlim=(0, 16), xlabel='Completed predictions', title='Pilot coverage at the midpoint')
    ax.set_xticks(range(0, 15, 2))
    fig.savefig(output / 'pilot_coverage.png', dpi=180)
    plt.close(fig)

    # These two completed approaches share the full fixed pilot; summary is incomplete.
    for name in names[:2]:
        if report['approaches'][name]['successful'] != report['approaches'][name]['expected']:
            raise ValueError('Full-pilot chart requires complete raw-history and retrieval runs')
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), layout='constrained')
    for ax, metric, title in zip(axes, ['input_tokens', 'total_seconds'],
                                 ['Mean answer-prompt tokens', 'Mean total seconds per question']):
        means = [report['approaches'][n]['measurements_successful_only'][metric]['mean'] for n in names[:2]]
        ax.bar(labels[:2], means, color=palette[:2], width=.5)
        for i, value in enumerate(means):
            ax.text(i, value * 1.035, f'{value:,.2f}', ha='center', fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(means) * 1.25)
    fig.suptitle('Same 14 pilot questions • ungraded answers', fontsize=11)
    fig.savefig(output / 'pilot_costs.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.6), layout='constrained')
    ax.set(xlim=(0, 10), ylim=(0, 4)); ax.axis('off')
    def box(x, y, w, h, text, fill):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08',
                                   facecolor=fill, edgecolor='white'))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=9, color='white')
    box(.1, 1.5, 1.7, .9, 'Dated history\n500-question dataset\n14 fixed pilot IDs', '#28536B')
    for y, text in [(2.9, 'Raw history\nDrop oldest turns'), (1.6, 'Vector retrieval\nLocal embeddings + Chroma'), (.3, 'Running summary\nChronological updates')]:
        box(3, y, 2.7, .8, text, '#238A87')
        ax.annotate('', xy=(2.95, y+.4), xytext=(1.9, 1.95), arrowprops={'arrowstyle':'->', 'color':'#64748B'})
        ax.annotate('', xy=(6.9, 1.95), xytext=(5.85, y+.4), arrowprops={'arrowstyle':'->', 'color':'#64748B'})
    box(7, 1.5, 2.7, .9, 'Shared Llama 3.1 8B\nAnswer + usage + timing\nAtomic result records', '#28536B')
    ax.text(5, 3.95, 'Memory preparation excludes reference answers and evidence annotations', ha='center', fontsize=10)
    fig.savefig(output / 'architecture.png', dpi=180)
    plt.close(fig)


def inline(text):
    text = html.escape(text)
    text = re.sub(r'`([^`]+)`', r'<font name="StudyMono">\1</font>', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1', text)
    return text


def render_pdf(source, output):
    fonts = Path(matplotlib.get_data_path()) / 'fonts/ttf'
    for name, file in [('Study', 'DejaVuSans.ttf'), ('StudyBold', 'DejaVuSans-Bold.ttf'),
                       ('StudyItalic', 'DejaVuSans-Oblique.ttf'), ('StudyMono', 'DejaVuSansMono.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(fonts / file)))
    pdfmetrics.registerFontFamily('Study', normal='Study', bold='StudyBold', italic='StudyItalic', boldItalic='StudyBold')
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Body', fontName='Study', fontSize=9.4, leading=13,
                              spaceAfter=7, textColor=colors.HexColor('#253346')))
    styles.add(ParagraphStyle(name='SmallCell', parent=styles['Body'], fontSize=8, leading=11, spaceAfter=0))
    styles.add(ParagraphStyle(name='ReportTitle', fontName='StudyBold', fontSize=24, leading=29,
                              spaceAfter=12, textColor=colors.HexColor('#173B50')))
    styles.add(ParagraphStyle(name='Section', fontName='StudyBold', fontSize=13.5, leading=18,
                              spaceBefore=10, spaceAfter=9, keepWithNext=True,
                              textColor=colors.HexColor('#173B50')))
    styles.add(ParagraphStyle(name='Subsection', fontName='StudyBold', fontSize=10, leading=14,
                              spaceBefore=5, spaceAfter=5, keepWithNext=True))
    page_width, page_height = 8.5*inch, 11*inch
    width = page_width - 1.3*inch
    document = SimpleDocTemplate(str(output), pagesize=(page_width, page_height),
        leftMargin=.65*inch, rightMargin=.65*inch, topMargin=.62*inch, bottomMargin=.65*inch,
        title='Midterm Project Report — Weeks 1–7', author='Abhishek Kumar Karn')
    lines = source.read_text().splitlines()
    story, index = [], 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1; continue
        if line == '<!-- pagebreak -->':
            story.append(PageBreak()); index += 1; continue
        if line.startswith('|'):
            rows = []
            while index < len(lines) and lines[index].strip().startswith('|'):
                cells = [c.strip() for c in lines[index].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+', cell) for cell in cells):
                    rows.append([Paragraph(inline(c), styles['SmallCell']) for c in cells])
                index += 1
            columns = len(rows[0])
            widths = [width/columns]*columns
            if columns == 2 and rows[0][0].getPlainText() == 'Period':
                widths = [width*.20, width*.80]
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign=TA_LEFT)
            table.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E6EEF3')),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F7F9FB')]),
                ('VALIGN',(0,0),(-1,-1),'TOP'),
                ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#9AB0C0')),
                ('LEFTPADDING',(0,0),(-1,-1),6), ('RIGHTPADDING',(0,0),(-1,-1),6),
                ('TOPPADDING',(0,0),(-1,-1),4.5), ('BOTTOMPADDING',(0,0),(-1,-1),4.5)]))
            story.extend([table, Spacer(1, 9)]); continue
        match = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line)
        if match:
            path = (source.parent / match.group(2)).resolve()
            picture = Image(str(path))
            picture.drawHeight *= width / picture.drawWidth
            picture.drawWidth = width
            story.extend([picture, Spacer(1, 7)])
            index += 1; continue
        if line.startswith('# '):
            story.append(Paragraph(inline(line[2:]), styles['ReportTitle']))
        elif line.startswith('## '):
            story.append(Paragraph(inline(line[3:]), styles['Section']))
        elif line.startswith('### '):
            story.append(Paragraph(inline(line[4:]), styles['Subsection']))
        elif line.startswith('- '):
            story.append(Paragraph('• ' + inline(line[2:]), styles['Body']))
        else:
            text = line
            while index+1 < len(lines) and lines[index+1].strip() and not lines[index+1].startswith(('#','|','!','- ','<!--')):
                index += 1; text += ' ' + lines[index].strip()
            story.append(Paragraph(inline(text), styles['Body']))
        index += 1
    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor('#CDD8E0'))
        canvas.line(.65*inch, .44*inch, page_width-.65*inch, .44*inch)
        canvas.setFont('Study', 7.5); canvas.setFillColor(colors.HexColor('#526579'))
        canvas.drawString(.65*inch, .28*inch, 'Abhishek Kumar Karn | CSCI 411-01 | Weeks 1–7')
        canvas.drawRightString(page_width-.65*inch, .28*inch, str(doc.page))
    document.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'reports/midpoint_project_report.md')
    parser.add_argument('--comparison', type=Path, default=ROOT / 'results/comparison.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/Midpoint_Project_Report.pdf')
    parser.add_argument('--docx', action='store_true', help='Also export editable Word document using installed pandoc')
    args = parser.parse_args()
    figures(json.loads(args.comparison.read_text()), ROOT / 'results/midpoint/figures')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render_pdf(args.source, args.output)
    if args.docx:
        pagebreak = '```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```'
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'report.md'
            source.write_text(args.source.read_text().replace('<!-- pagebreak -->', pagebreak))
            subprocess.run(['pandoc', str(source), '--from=markdown', '--to=docx',
                '--resource-path=' + str(args.source.parent.resolve()),
                '--output=' + str(args.output.with_suffix('.docx'))], check=True)
    print(args.output)


if __name__ == '__main__':
    main()
