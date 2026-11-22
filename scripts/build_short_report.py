"""Build the sole README and concise PDF from the same structured manuscript."""
from __future__ import annotations
import hashlib,json,re,shutil
from html import escape
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate,PageTemplate,Frame,Paragraph,Table,TableStyle,PageBreak,Image,Spacer,Preformatted
from pypdf import PdfReader


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def markdown(document,equations):
    parts=[]
    for page in document['pages']:
        parts += ['#'*page['level']+' '+page['title'],'']
        for block in page['blocks']:
            kind=block['type']
            if kind=='paragraph':parts += [block['text'],'']
            elif kind=='equation':
                record=equations[block['id']]
                parts += ['<p align="center">',f'  <img src="{record["file"]}" alt="{escape(record["alt"],quote=True)}" width="{record["display_width"]}">','</p>','']
            elif kind=='table':
                parts += ['| '+' | '.join(block['headers'])+' |','|'+'|'.join(['---']*len(block['headers']))+'|']
                parts += ['| '+' | '.join(row)+' |' for row in block['rows']];parts.append('')
            elif kind in ('list','references'):
                parts += [(f'{i+1}. ' if kind=='references' else '- ')+value for i,value in enumerate(block['items'])];parts.append('')
            elif kind=='figure':parts += [f'![{block["caption"]}]({block["file"]})','',f'*Рисунок {block["number"]}. {block["caption"]}*','']
            elif kind=='code':parts += ['```'+block['language'],block['text'],'```','']
            else:raise ValueError(kind)
    return '\n'.join(parts).rstrip()+'\n'


def pdf_text(value,root):
    value=value.translate(str.maketrans({'–':'-','—':'-','−':'-','‑':'-'}));value=escape(value)
    value=value.replace('&lt;sup&gt;','<sup>').replace('&lt;/sup&gt;','</sup>')
    value=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',value)
    def link(match):
        label,target=match.groups()
        if not re.match(r'[a-z]+://',target):target='https://github.com/Alexandch/hawkes-criticality-lab/blob/main/'+target
        return f'<link href="{target}" color="#225b88">{label}</link>'
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,value)


def main():
    root=Path(__file__).resolve().parents[1];source=root/'report/short_report.json';document=json.loads(source.read_text(encoding='utf-8'))
    formulas=json.loads((root/'report/equations.json').read_text(encoding='utf-8'))['equations']
    saved=json.loads((root/'report/assets/equations_manifest.json').read_text(encoding='utf-8'))['equations']
    equations={a['id']:{**a,**b} for a,b in zip(formulas,saved)}
    readme=markdown(document,equations);(root/'README.md').write_text(readme,encoding='utf-8')
    temporary=root/'tmp/pdfs/short';temporary.mkdir(exist_ok=True,parents=True)
    for name,file in [('ShortTimes','times.ttf'),('ShortBold','timesbd.ttf'),('ShortItalic','timesi.ttf')]:pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
    pdfmetrics.registerFontFamily('ShortTimes',normal='ShortTimes',bold='ShortBold',italic='ShortItalic',boldItalic='ShortBold')
    body=ParagraphStyle('Body',fontName='ShortTimes',fontSize=10.8,leading=14.4,spaceAfter=10)
    small=ParagraphStyle('Small',parent=body,fontSize=9.1,leading=12,spaceAfter=8)
    reference=ParagraphStyle('Reference',parent=small,fontSize=9.4,leading=12.8,spaceAfter=10)
    heading=ParagraphStyle('Heading',fontName='ShortBold',fontSize=17,leading=22,spaceAfter=16)
    cover=ParagraphStyle('Cover',parent=heading,fontSize=22,leading=28,spaceAfter=25)
    cell=ParagraphStyle('Cell',fontName='ShortTimes',fontSize=9.2,leading=12)
    code=ParagraphStyle('Code',fontName='Courier',fontSize=7.3,leading=11,spaceAfter=12)
    positions=[]
    class Document(BaseDocTemplate):
        def afterFlowable(self,flowable):
            if isinstance(flowable,Paragraph) and flowable.style.name in ('Heading','Cover'):positions.append(dict(title=flowable.getPlainText(),page=self.page))
    path=temporary/'candidate.pdf';doc=Document(str(path),pagesize=A4,title=document['title'],author='Hawkes Criticality Lab')
    width,height=A4;frame_width=width-36*mm
    def footer(canvas,current):
        canvas.saveState();canvas.setFont('ShortTimes',8);canvas.setFillColor(colors.HexColor('#52616e'))
        canvas.drawString(18*mm,12*mm,'Hawkes Criticality Lab - основной отчёт, 22.11.2026')
        canvas.drawRightString(width-18*mm,12*mm,str(current.page));canvas.restoreState()
    doc.addPageTemplates(PageTemplate(id='main',frames=[Frame(18*mm,24*mm,frame_width,height-44*mm,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=footer))
    story=[]
    def paragraph(text,style=body):return Paragraph(pdf_text(text,root),style)
    used_assets=[]
    for index,page in enumerate(document['pages']):
        if index:story.append(PageBreak())
        story.append(paragraph(page['title'],cover if index==0 else heading))
        for block in page['blocks']:
            kind=block['type']
            if kind=='paragraph':story.append(paragraph(block['text']))
            elif kind=='equation':
                record=equations[block['id']];w=min(frame_width,record['display_width']*.75);h=w*record['pixel_size'][1]/record['pixel_size'][0]
                story += [Image(str(root/record['file']),width=w,height=h),Spacer(1,12)];used_assets.append(record['file'])
            elif kind=='figure':
                from PIL import Image as PillowImage
                asset=root/block['file'];pixel=PillowImage.open(asset).size;w=min(frame_width,block['width_mm']*mm);h=w*pixel[1]/pixel[0]
                story += [Image(str(asset),width=w,height=h),Spacer(1,6),paragraph(f'Рисунок {block["number"]}. {block["caption"]}',small),Spacer(1,5)];used_assets.append(block['file'])
            elif kind=='table':
                rows=[block['headers']]+block['rows'];columns=len(rows[0])
                weights=[max(6,min(65,max(len(row[j]) for row in rows)))**.5 for j in range(columns)]
                proportions=[value/sum(weights) for value in weights]
                table=Table([[paragraph(text,cell) for text in row] for row in rows],colWidths=[frame_width*v for v in proportions],repeatRows=1)
                table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef3')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#c6d1da')),
                    ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
                story += [table,Spacer(1,12)]
            elif kind in ('list','references'):
                for i,value in enumerate(block['items']):story.append(paragraph((str(i+1)+'. ' if kind=='references' else '• ')+value,reference if kind=='references' else body))
            elif kind=='code':story.append(Preformatted(block['text'],code))
            else:raise ValueError(kind)
    doc.build(story);reader=PdfReader(path)
    if len(reader.pages)!=document['expected_pages'] or [r['page'] for r in positions]!=list(range(1,21)):
        raise ValueError(f'Unexpected page flow: {len(reader.pages)}pages; sections={positions}')
    target=root/'output/pdf/hawkes_criticality_report.pdf';shutil.copyfile(path,target)
    sources=[source,root/'scripts/compose_short_report.py',root/'reproducibility/full_report_archive.json',root/'report/equations.json',root/'report/assets/equations_manifest.json']
    sources += [root/name for name in used_assets]
    for name in ['n_estimation_accuracy/summary.csv','background_comparison/summary.csv','indicators_final/summary.csv','genealogical_clusters/summary.csv',
        'extended_neural/metrics.json','spatiotemporal_forecast/test_metrics.csv','expanded_district_detector/headline.csv','delay_forecast_detector/metrics.csv',
        'selective_austin_detector/metrics.csv','cad_context_detector/metrics.csv']:
        sources.append(root/'results'/name)
    record=dict(edition=document['edition'],pages=len(reader.pages),full_edition_pages=70,readme_words=len(readme.split()),
        readme_sha256=digest(root/'README.md'),output_sha256=digest(target),builder_sha256=digest(Path(__file__)),
        archive_sha256=digest(root/document['archive']),source_sha256={p.relative_to(root).as_posix():digest(p) for p in sources},
        sections=positions,experimental_results_changed=False)
    (root/'output/pdf/short_report_build.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Built {len(reader.pages)}pages and{len(readme.split())}README words from one manuscript.')


if __name__=='__main__':main()
