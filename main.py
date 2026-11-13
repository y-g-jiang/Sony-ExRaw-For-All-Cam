"""Source entry point and console-free, self-contained EXE entry point."""
import sys,json,traceback
from pathlib import Path
from paths import APP_HOME

def main():
    if '--pick' in sys.argv:
        from pick import run
        i=sys.argv.index('--pick');run(*sys.argv[i+1:i+4]);return
    if '--self-test' in sys.argv:
        from self_test import run
        i=sys.argv.index('--self-test');folder=sys.argv[i+1]
        source=sys.argv[sys.argv.index('--input')+1] if '--input' in sys.argv else None
        run(folder,source);return
    if '--no-browser' not in sys.argv:sys.argv.append('--open')
    import app
    app.main()

if __name__=='__main__':
    try:main()
    except Exception as e:
        text=traceback.format_exc();(APP_HOME/'STARTUP_ERROR.txt').write_text(text,encoding='utf8')
        if '--self-test' in sys.argv:
            i=sys.argv.index('--self-test');p=Path(sys.argv[i+1]);p.mkdir(parents=True,exist_ok=True)
            (p/'SELF_TEST.json').write_text(json.dumps(dict(passed=False,error=str(e),traceback=text),ensure_ascii=False,indent=2),encoding='utf8')
        elif '--no-browser' not in sys.argv:
            import tkinter as tk
            from tkinter import messagebox
            root=tk.Tk();root.withdraw();messagebox.showerror('Sony ExRAW',f'启动失败：{e}\n详情：{APP_HOME / "STARTUP_ERROR.txt"}');root.destroy()
        raise SystemExit(1)
