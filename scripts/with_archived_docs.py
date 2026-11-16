"""Temporarily restore byte-exact documentation for frozen legacy commands."""
from __future__ import annotations
import json,os,runpy,sys
from pathlib import Path
from critical_processes.documentation_archive import archive_index,archive_member,archived_bytes,verify_archive


def main():
    root=Path(__file__).resolve().parents[1];args=sys.argv[1:]
    if not args:raise SystemExit('usage: python scripts/with_archived_docs.py <script.py | -m module> [arguments]')
    if args[0]=='-m' and len(args)<2:raise SystemExit('module name required after -m')
    verify_archive(root);index=archive_index(root);created=[];lock=root/'reproducibility/.documentation_restore.lock'
    handle=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(handle,json.dumps({'pid':os.getpid()}).encode('utf-8'));os.close(handle)
        for name in index['removed_paths']:
            path,_=archive_member(root,name)
            if path.exists():continue
            content=archived_bytes(root,name);path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as output:output.write(content)
            created.append((path,content))
        os.chdir(root)
        if args[0]=='-m':
            sys.argv=args[1:];runpy.run_module(args[1],run_name='__main__',alter_sys=True)
        else:
            target=(root/args[0]).resolve()
            if not target.is_relative_to(root) or not target.is_file():raise ValueError('script must be inside this project')
            sys.argv=[str(target),*args[1:]];sys.path.insert(0,str(target.parent));runpy.run_path(str(target),run_name='__main__')
    finally:
        for path,content in reversed(created):
            if path.is_file() and path.read_bytes()==content:path.unlink()
            elif path.exists():print('Preserved documentation modified by the command: '+str(path),file=sys.stderr)
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()
