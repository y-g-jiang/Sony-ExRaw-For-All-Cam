"""Locate bundled read-only assets and writable, portable application data."""
import os,sys
from pathlib import Path

BUNDLE=Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parent))
INSTALL=Path(sys.executable).resolve().parent if getattr(sys,'frozen',False) else Path(__file__).resolve().parent
APP_HOME=Path(os.environ.get('SONY_EXRAW_HOME',str(INSTALL/'SonyExRAW_Data'))).resolve()
APP_HOME.mkdir(parents=True,exist_ok=True)
TEMP_HOME=APP_HOME/'temp';TEMP_HOME.mkdir(exist_ok=True)
os.environ.update(TEMP=str(TEMP_HOME),TMP=str(TEMP_HOME),PYTHONDONTWRITEBYTECODE='1')
# Keep native graphics caches alongside this installation's data.
CACHE_HOME=APP_HOME/'cache';CACHE_HOME.mkdir(exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME',str(CACHE_HOME))
os.environ['LOCALAPPDATA']=str(CACHE_HOME/'localappdata')
os.environ['__GL_SHADER_DISK_CACHE_PATH']=str(CACHE_HOME/'nvidia')
VERSION='1.1.0'

def writable_path(value):
    p=Path(value).expanduser().resolve()
    allowed=os.environ.get('SONY_EXRAW_ALLOWED_DRIVES','')
    drives={v.strip().upper().rstrip(':/')+':' for v in allowed.split(',') if v.strip()}
    if drives and p.drive.upper() not in drives:raise ValueError('保存目录不在配置允许的磁盘中。')
    return p
