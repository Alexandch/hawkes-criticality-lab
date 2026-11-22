"""Verify the concise edition and byte-exact archive without large raw inputs."""
from __future__ import annotations
import hashlib,json,re,zipfile
from pathlib import Path


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    full=json.loads((root/'reproducibility/full_report_archive.json').read_text(encoding='utf-8'))
    archive=root/full['archive'];assert digest(archive)==full['archive_sha256']
    with zipfile.ZipFile(archive) as handle:
        assert handle.testzip() is None and set(handle.namelist())==set(full['files'])
        for name,record in full['files'].items():
            content=handle.read(name);assert len(content)==record['bytes'] and hashlib.sha256(content).hexdigest()==record['sha256']
        original=handle.read('full_report/README.md').decode('utf-8')
    historical=json.loads((root/'output/pdf/r16_build.json').read_text(encoding='utf-8'))
    assert full['pages']==70 and historical['output_sha256']==full['files']['full_report/hawkes_criticality_report.pdf']['sha256']
    build=json.loads((root/'output/pdf/short_report_build.json').read_text(encoding='utf-8'))
    assert build['pages']==20 and build['full_edition_pages']==70 and not build['experimental_results_changed']
    assert build['readme_sha256']==digest(root/'README.md')
    assert build['output_sha256']==digest(root/'output/pdf/hawkes_criticality_report.pdf')
    assert build['builder_sha256']==digest(root/'scripts/build_short_report.py')
    assert build['archive_sha256']==digest(archive)
    for name,value in build['source_sha256'].items():assert digest(root/name)==value,name
    assert [r['page'] for r in build['sections']]==list(range(1,21))
    source=json.loads((root/'report/short_report.json').read_text(encoding='utf-8'))
    text=(root/'README.md').read_text(encoding='utf-8');assert len(source['pages'])==20
    assert len(text.split())==build['readme_words'] and len(text.split())<len(original.split())*.4
    position=0;figure_numbers=[];equations=[]
    for page in source['pages']:
        needle='#'*page['level']+' '+page['title'];position=text.index(needle,position)+len(needle)
        for block in page['blocks']:
            kind=block['type']
            if kind=='paragraph':position=text.index(block['text'],position)+len(block['text'])
            elif kind=='code':position=text.index(block['text'],position)+len(block['text'])
            elif kind in ('list','references'):
                for item in block['items']:position=text.index(item,position)+len(item)
            elif kind=='figure':
                figure_numbers.append(block['number']);assert (root/block['file']).is_file()
                needle=f'*Рисунок {block["number"]}. {block["caption"]}*';position=text.index(needle,position)+len(needle)
            elif kind=='equation':equations.append(block['id'])
            elif kind=='table':
                for row in [block['headers']]+block['rows']:
                    needle='| '+' | '.join(row)+' |';position=text.index(needle,position)+len(needle)
            else:raise ValueError(kind)
    assert figure_numbers==list(range(1,8)) and len(equations)==7
    assert '$' not in text and len(re.findall(r'(?m)^# ',text))==1
    qa=json.loads((root/'output/pdf/short_report_qa.json').read_text(encoding='utf-8'))
    assert qa['successful'] and qa['pages']==20 and qa['sha256']==build['output_sha256']
    assert qa['visual_reviewed_pages']==list(range(1,21))
    return dict(pages=20,full_pages=70,readme_words=len(text.split()),original_words=len(original.split()),archive_pdf_sha256=historical['output_sha256'])


if __name__=='__main__':print(json.dumps(verify(Path(__file__).resolve().parents[1])))
