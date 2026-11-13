"""Native file chooser, dispatched by the same EXE when frozen."""
import json
from pathlib import Path

def run(kind,result,initial):
    import tkinter as tk
    from tkinter import filedialog
    root=tk.Tk();root.withdraw();root.attributes('-topmost',True)
    try:
        if kind=='files':
            value=list(filedialog.askopenfilenames(title='选择单帧 Bayer RAW',filetypes=[('RAW 文件','*.RW2 *.ARW *.NEF *.DNG *.CR2 *.CR3 *.RAF *.ORF *.PEF *.SRW *.RAW'),('全部文件','*.*')]))
        else:value=filedialog.askdirectory(title='选择导出文件夹',initialdir=initial,mustexist=True)
        Path(result).write_text(json.dumps({'value':value},ensure_ascii=False),encoding='utf8')
    finally:root.destroy()
