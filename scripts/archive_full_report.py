"""Preserve the full R1-R16 edition before publishing a shorter main report."""
from __future__ import annotations
import hashlib,json,subprocess,zipfile
from pathlib import Path

REVISION='ecf0f70790a52d666c7677bc2bbdbf6a2d568492'


def main():
    root=Path(__file__).resolve().parents[1];items={}
    for source,target in [('README.md','full_report/README.md'),('output/pdf/hawkes_criticality_report.pdf','full_report/hawkes_criticality_report.pdf')]:
        items[target]=subprocess.check_output(['git','show',REVISION+':'+source],cwd=root)
    build=json.loads(subprocess.check_output(['git','show',REVISION+':output/pdf/r16_build.json'],cwd=root))
    assert hashlib.sha256(items['full_report/hawkes_criticality_report.pdf']).hexdigest()==build['output_sha256']
    archive=root/'reproducibility/full_report_archive.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as handle:
        for name,content in items.items():
            info=zipfile.ZipInfo(name,date_time=(2026,11,22,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;handle.writestr(info,content)
    record=dict(revision=REVISION,pages=70,archive='reproducibility/full_report_archive.zip',
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        files={name:dict(sha256=hashlib.sha256(value).hexdigest(),bytes=len(value)) for name,value in items.items()},
        purpose='byte-exact full README and70-page PDF; current README/PDF are a shorter main edition')
    (root/'reproducibility/full_report_archive.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record))


if __name__=='__main__':main()
