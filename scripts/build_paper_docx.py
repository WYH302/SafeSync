"""Edit the archived manuscript's Word styles using the bundled docx runtime."""
from pathlib import Path
import json

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches,Pt,RGBColor

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'archive/originals_2026-10-28/SafeSync_TGRS_Manuscript_Draft.docx'
    document=Document(source)
    body=document._element.body
    for child in list(body):
        if child.tag!=qn('w:sectPr'):
            body.remove(child)
    for section in document.sections:
        section.page_width=Inches(8.5);section.page_height=Inches(11)
        section.left_margin=section.right_margin=Inches(.72)
        section.top_margin=section.bottom_margin=Inches(.65)
        for paragraph in section.header.paragraphs:
            paragraph.text='SAFESYNC DEVELOPMENT RESEARCH DRAFT'
            paragraph.alignment=WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                run.font.name='Times New Roman';run.font.size=Pt(8)
    for name in ('Normal','Title','Heading 1','Heading 2','Caption'):
        style=document.styles[name]
        style.font.name='Times New Roman'
        style.font.color.rgb=RGBColor(0,0,0)
        style._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'SimSun')
        for key in ('asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme'):
            style._element.get_or_add_rPr().rFonts.attrib.pop(qn('w:'+key),None)
        prop=style._element.find(qn('w:pPr'))
        if prop is not None:
            for border in list(prop.findall(qn('w:pBdr'))):prop.remove(border)
    document.styles['Normal'].font.size=Pt(11)
    document.styles['Normal'].paragraph_format.space_after=Pt(6)
    document.styles['Normal'].paragraph_format.line_spacing=1.05
    document.styles['Title'].font.size=Pt(17)
    document.styles['Title'].font.bold=True
    document.styles['Heading 1'].font.size=Pt(12)
    document.styles['Heading 1'].font.bold=True
    document.styles['Heading 1'].paragraph_format.space_before=Pt(10)
    document.styles['Heading 1'].paragraph_format.space_after=Pt(5)
    document.styles['Caption'].font.size=Pt(9)
    blocks=json.loads((ROOT/'docs/paper_content.json').read_text(encoding='utf-8'))
    for block in blocks:
        kind=block['kind']
        if kind=='title':
            p=document.add_paragraph(block['text'],style='Title')
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        elif kind=='heading':
            document.add_paragraph(block['text'],style='Heading 1')
        elif kind=='paragraph':
            p=document.add_paragraph(block['text'])
            p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
        elif kind=='table':
            cap=document.add_paragraph(block['caption'],style='Caption')
            cap.paragraph_format.keep_with_next=True
            table=document.add_table(rows=1,cols=len(block['headers']))
            table.autofit=False
            widths=[Inches(3.0),Inches(.85),Inches(1.1),Inches(1.1)]
            for cell,width in zip(table.rows[0].cells,widths):cell.width=width
            for cell,text in zip(table.rows[0].cells,block['headers']):
                cell.text=text
                shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E8EDF2');cell._tc.get_or_add_tcPr().append(shade)
                for run in cell.paragraphs[0].runs:run.bold=True
            repeat=OxmlElement('w:tblHeader');table.rows[0]._tr.get_or_add_trPr().append(repeat)
            for data in block['rows']:
                cells=table.add_row().cells
                for index,(cell,text) in enumerate(zip(cells,data)):
                    cell.width=widths[index];cell.text=str(text)
            for row in table.rows:
                for index,cell in enumerate(row.cells):
                    cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                    prop=cell._tc.get_or_add_tcPr()
                    borders=OxmlElement('w:tcBorders')
                    for edge in ('top','left','bottom','right'):
                        line=OxmlElement('w:'+edge);line.set(qn('w:val'),'single');line.set(qn('w:sz'),'4');line.set(qn('w:color'),'D9D9D9');borders.append(line)
                    prop.append(borders)
                    for paragraph in cell.paragraphs:
                        paragraph.alignment=WD_ALIGN_PARAGRAPH.LEFT if index==0 else WD_ALIGN_PARAGRAPH.CENTER
                        paragraph.paragraph_format.space_after=Pt(4)
                        paragraph.paragraph_format.space_before=Pt(4)
                        for run in paragraph.runs:run.font.size=Pt(9)
            document.add_paragraph()
        elif kind=='image':
            p=document.add_paragraph()
            p.paragraph_format.keep_with_next=True
            p.add_run().add_picture(str(ROOT/block['path']),width=Inches(6.65))
            document.add_paragraph(block['caption'],style='Caption')
    document.core_properties.title=blocks[0]['text']
    document.core_properties.subject='Evidence-constrained development pilot; not a completed submission'
    document.core_properties.author=''
    document.core_properties.keywords='SafeSync; development pilot; receiver verification'
    output=ROOT/'SafeSync_TGRS_Manuscript_Draft.docx'
    document.save(output)
    print('Saved revised development manuscript')


if __name__=='__main__':main()
