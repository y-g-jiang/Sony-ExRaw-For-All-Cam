"""Reproducible local build; intermediates and caches stay under the chosen folders."""
from pathlib import Path
import os,sys,subprocess,hashlib

root=Path(__file__).resolve().parents[1]
temp=root/'release-work/temp';temp.mkdir(parents=True,exist_ok=True)
cache=root/'release-work/pyinstaller-cache';cache.mkdir(parents=True,exist_ok=True)
os.environ.update(TEMP=str(temp),TMP=str(temp),PYINSTALLER_CONFIG_DIR=str(cache),PYTHONDONTWRITEBYTECODE='1')
model=root/'models/exraw_hires_v3.48.onnx'
expected='43fd0e4566d39540c3e3912fda24f0e53bb7b509c2e9d18f8e01b4b94324ab22'
if hashlib.sha256(model.read_bytes()).hexdigest()!=expected:raise SystemExit('Original model checksum mismatch')
subprocess.run([sys.executable,'-B','-m','PyInstaller','--noconfirm','--distpath',str(root/'dist'),
    '--workpath',str(root/'build'),str(root/'SonyExRAWStudio.spec')],cwd=root,check=True)
