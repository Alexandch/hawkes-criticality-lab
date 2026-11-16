"""Render README equations as static PNGs without browser TeX dependencies."""
from __future__ import annotations
import argparse,hashlib,io,json
from pathlib import Path


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];source=root/'report/equations.json';manifest=root/'report/assets/equations_manifest.json'
    definitions=json.loads(source.read_text(encoding='utf-8'))['equations']
    if args.verify:
        saved=json.loads(manifest.read_text(encoding='utf-8'))
        assert saved['source_sha256']==digest(source) and saved['renderer_sha256']==digest(Path(__file__))
        assert len(saved['equations'])==len(definitions)==7
        for definition,record in zip(definitions,saved['equations']):
            assert definition['id']==record['id'] and definition['file']==record['file']
            content=(root/record['file']).read_bytes();assert content.startswith(b'\x89PNG\r\n\x1a\n')
            assert hashlib.sha256(content).hexdigest()==record['sha256']
            assert [int.from_bytes(content[16:20],'big'),int.from_bytes(content[20:24],'big')]==record['pixel_size']
        print('Verified seven static formula images, sources and renderer.')
        return
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import rc_context
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import math_to_image
    from PIL import Image
    records=[]
    with rc_context({'mathtext.fontset':'stix','font.family':'STIXGeneral'}):
        for definition in definitions:
            path=(root/definition['file']).resolve()
            if not path.is_relative_to(root/'report/assets'):raise ValueError('formula path escapes asset directory')
            buffer=io.BytesIO();tex=' '.join(definition['latex'].splitlines())
            math_to_image('$'+tex+'$',buffer,prop=FontProperties(size=18),dpi=300,format='png',color='#1f2328')
            buffer.seek(0)
            with Image.open(buffer) as image:
                rgba=image.convert('RGBA');paper=Image.new('RGB',(rgba.width+64,rgba.height+48),'white');paper.paste(rgba,(32,24),rgba)
                path.parent.mkdir(parents=True,exist_ok=True);paper.save(path,format='PNG',dpi=(300,300))
                pixels=[paper.width,paper.height]
            records.append(dict(id=definition['id'],file=definition['file'],sha256=digest(path),pixel_size=pixels,display_width=round(pixels[0]/3.75)))
    manifest.write_text(json.dumps(dict(source_sha256=digest(source),renderer_sha256=digest(Path(__file__)),equations=records),ensure_ascii=False,indent=2),encoding='utf-8')
    print('Rendered seven static PNG equations at 300 dpi.')


if __name__=='__main__':main()
