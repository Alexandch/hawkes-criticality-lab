"""Read byte-exact historical documentation after README consolidation."""
from __future__ import annotations
import hashlib,json
from pathlib import Path,PurePosixPath
from zipfile import ZipFile


def archive_index(root):
    root=Path(root).resolve()
    return json.loads((root/'reproducibility/documentation_archive.json').read_text(encoding='utf-8'))


def archive_member(root,name):
    root=Path(root).resolve();relative=PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or relative.suffix.lower()!='.md':raise ValueError('invalid documentation archive member')
    target=root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root):raise ValueError('documentation path escapes project')
    return target,relative.as_posix()


def archived_bytes(root,name):
    root=Path(root).resolve();_,name=archive_member(root,name);index=archive_index(root)
    record=index['files'][name];path=root/index['archive']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=index['archive_sha256']:raise ValueError('documentation archive changed')
    with ZipFile(path) as archive:content=archive.read(name)
    if len(content)!=record['bytes'] or hashlib.sha256(content).hexdigest()!=record['sha256']:raise ValueError('archived documentation changed: '+name)
    return content


def read_artifact(path,root=None):
    path=Path(path)
    if path.is_file():return path.read_bytes()
    root=Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    relative=path.resolve().relative_to(root).as_posix()
    if path.suffix.lower()!='.md':raise FileNotFoundError(path)
    return archived_bytes(root,relative)


def verify_archive(root):
    root=Path(root).resolve();index=archive_index(root)
    with ZipFile(root/index['archive']) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(index['files']):raise ValueError('archive members differ from index')
    for name in index['files']:archived_bytes(root,name)
    return len(index['files'])
