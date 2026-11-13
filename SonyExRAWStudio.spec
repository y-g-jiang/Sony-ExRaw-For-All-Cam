# Build with tools/build_release.py; keep the original model as bundled data.
from pathlib import Path
from PyInstaller.utils.hooks import collect_dynamic_libs,collect_data_files
root=Path(SPECPATH)
datas=[(str(root/'ui.html'),'.'),(str(root/'models/exraw_hires_v3.48.onnx'),'models'),
       (str(root/'data/s5_iso100_ptc.json'),'data'),(str(root/'MODEL_INFO.txt'),'.'),
       (str(root/'USAGE.txt'),'.')]
if (root/'THIRD_PARTY_NOTICES').is_dir():datas.append((str(root/'THIRD_PARTY_NOTICES'),'THIRD_PARTY_NOTICES'))
a=Analysis([str(root/'main.py')],pathex=[str(root)],binaries=collect_dynamic_libs('wgpu'),
    datas=datas+collect_data_files('wgpu',includes=['resources/*']),hiddenimports=['wgpu.backends.wgpu_native','_cffi_backend'],hookspath=[],hooksconfig={},
    runtime_hooks=[],excludes=['onnxruntime','onnx.reference','onnx.backend.test','torch','tensorflow','matplotlib','pandas','scipy','IPython','pytest','notebook','jupyter'],noarchive=False)
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,a.binaries,a.datas,[],name='SonyExRAWStudio-v1.1.0-Windows-x64',
    debug=False,bootloader_ignore_signals=False,strip=False,upx=False,console=False,
    disable_windowed_traceback=False,runtime_tmpdir='.')
