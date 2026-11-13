"""Single-user localhost desktop workbench. No uploads or cloud service."""
import os
from paths import APP_HOME,BUNDLE,VERSION
from pathlib import Path
import sys,json,time,secrets,threading,queue,subprocess,traceback,shutil,re,webbrowser,logging
from logging.handlers import RotatingFileHandler
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlsplit,parse_qs
import engine

class LocalServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR can attach another process to an active listening port.
    allow_reuse_address=False
    def server_bind(self):
        import socket
        if os.name=='nt':self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        super().server_bind()

ROOT=APP_HOME;WORK=ROOT/'work';OUT=ROOT/'outputs';THUMBS=WORK/'thumbs';JOBS=WORK/'jobs'
for d in (WORK,OUT,THUMBS,JOBS):d.mkdir(parents=True,exist_ok=True)
log=logging.getLogger('studio');log.setLevel(logging.INFO)
handler=RotatingFileHandler(WORK/'studio.log',maxBytes=256*1024,backupCount=1,encoding='utf8');log.addHandler(handler)
LOCK=threading.RLock();FILES={};TASKS={};PENDING=queue.Queue();ACTIVE=None;TOKEN=secrets.token_urlsafe(32);PORT=52348;SERVER=None
LAST=time.monotonic();STOP=threading.Event()

def public_task(t):return {k:v for k,v in t.items() if k not in ('cancel','settings')}

def state():
    with LOCK:
        return dict(files=list(FILES.values()),jobs=[public_task(t) for t in TASKS.values()],active=ACTIVE,
            output_default=str(OUT),model='Sony ExRAW Hi-Res v3.48',provider='WGPU · 严格 GPU 推理',backend=engine.backend_info(),
            profile={k:engine.S5[k] for k in ('alpha','beta_total','c')},
            free_gib=round(shutil.disk_usage(ROOT).free/2**30,1))

def update(jid,**values):
    with LOCK:TASKS[jid].update(values)

def save_state():
    with LOCK:
        value=dict(files=list(FILES.values()),jobs=[public_task(t) for t in list(TASKS.values())[-40:]])
        temp=WORK/'session.tmp'
        temp.write_text(json.dumps(value,ensure_ascii=False),encoding='utf8');os.replace(temp,WORK/'session.json')

def restore_state():
    try:
        value=json.loads((WORK/'session.json').read_text(encoding='utf8'))
        for f in value.get('files',[])[:100]:
            p=Path(f['path'])
            if p.is_file() and p.stat().st_size==f['meta'].get('bytes'):FILES[f['id']]=f
        for t in value.get('jobs',[])[-40:]:
            if t['file_id'] not in FILES:continue
            t['cancel']=threading.Event()
            if t['status'] in ('queued','running'):t.update(status='cancelled',stage='上次运行已结束，请重新提交')
            TASKS[t['id']]=t
    except (OSError,ValueError,KeyError):pass
    # Only named, reproducible scratch owned by this app; instance lock is already held.
    for folder in JOBS.iterdir():
        if not folder.is_dir() or not re.fullmatch('[a-f0-9]{16}',folder.name):continue
        for name in ('rgb.dat','quant.dat'):
            p=folder/name
            if p.exists():p.unlink()
        if folder.name not in TASKS:
            for name in ('before.jpg','after.jpg'):
                p=folder/name
                if p.exists():p.unlink()
            try:folder.rmdir()
            except OSError:pass
    for p in THUMBS.glob('*.jpg'):
        if p.stem not in FILES:p.unlink()

def worker():
    global ACTIVE
    while not STOP.is_set():
        try:jid=PENDING.get(timeout=1)
        except queue.Empty:continue
        task=TASKS[jid]
        if task['cancel'].is_set():update(jid,status='cancelled',stage='已停止');PENDING.task_done();continue
        with LOCK:ACTIVE=jid;task.update(status='running',stage='准备处理',started=time.time())
        scratch=JOBS/jid
        def progress(stage,n,total):update(jid,stage=stage,completed=n,total=total,elapsed=time.time()-task['started'])
        try:
            meta=FILES[task['file_id']]['meta'];h,w=meta['source_shape']
            scratch_bpp=72 if task['settings'].get('format','dng')=='dng' else 48
            output_bpp=24 if task['settings'].get('format','dng')=='dng' else 48
            if task.get('output') and Path(task['output']).drive.upper()==ROOT.drive.upper():scratch_bpp+=output_bpp
            required=512*1024**2+h*w*(scratch_bpp if task['mode']=='export' else 0)
            if shutil.disk_usage(ROOT).free<required:raise ValueError('工作盘临时空间不足，请先释放空间或更改数据目录。')
            result=engine.process(task['source'],task['settings'],scratch,
                Path(task['output']) if task.get('output') else None,task['mode'],progress,task['cancel'],task.get('center',(.5,.5)))
            update(jid,status='done',stage='处理完成',completed=1,total=1,result=result,elapsed=time.time()-task['started'])
        except engine.Cancelled: update(jid,status='cancelled',stage='已停止')
        except Exception as e:
            log.error('Job %s failed\n%s',jid,traceback.format_exc());update(jid,status='error',stage='处理失败',error=str(e))
        finally:
            with LOCK:ACTIVE=None
            try:save_state()
            except OSError:log.exception('Could not save session')
            PENDING.task_done()

def import_files(paths):
    if not isinstance(paths,list) or not 1<=len(paths)<=100:raise ValueError('每次请选择1–100个RAW文件。')
    added=[];errors=[]
    for value in paths:
        p=Path(str(value).strip().strip('"')).resolve()
        with LOCK:
            existing=next((f for f in FILES.values() if f['path']==str(p)),None)
            if existing:added.append(existing['id']);continue
            if len(FILES)>=100:raise ValueError('当前队列最多100个文件，请重启软件后建立新队列。')
        fid=secrets.token_hex(8)
        try:
            meta=engine.inspect(p,THUMBS/(fid+'.jpg'))
            with LOCK:FILES[fid]=dict(id=fid,name=p.name,path=str(p),meta=meta,thumb='/image/file/'+fid+'/thumb.jpg')
            added.append(fid)
        except Exception as e:errors.append(dict(path=str(p),error=str(e)))
    save_state()
    return dict(added=added,errors=errors)

def start_jobs(body):
    ids=body.get('ids',[]);mode=body.get('mode','preview');settings=body.get('settings',{})
    if mode not in ('preview','export'):raise ValueError('无效任务类型。')
    if not isinstance(ids,list) or not ids or len(ids)>100:raise ValueError('请先选择RAW文件。')
    if mode=='preview' and len(ids)!=1:raise ValueError('细节试算一次处理一张。')
    if settings.get('profile','s5') not in ('s5','manual'):raise ValueError('无效噪声配置。')
    engine.noise_settings(settings)
    if settings.get('format','dng') not in ('dng','tiff'):raise ValueError('无效导出格式。')
    center=list(map(float,body.get('center',[.5,.5])))
    if len(center)!=2 or any(not 0<=v<=1 for v in center):raise ValueError('预览位置无效。')
    with LOCK:
        if ACTIVE or any(t['status']=='queued' for t in TASKS.values()):raise ValueError('正在处理任务，请等待完成或停止当前队列。')
        if any(fid not in FILES for fid in ids):raise ValueError('文件已不在队列中。')
        # Keep only the latest 40 completed jobs and their tiny JPEG previews.
        for key in list(TASKS)[:-40]:
            old=JOBS/key
            if old.is_dir():
                for f in old.iterdir():
                    if f.name in ('before.jpg','after.jpg'):f.unlink()
                try:old.rmdir()
                except OSError:pass
            del TASKS[key]
        folder=None
        if mode=='export':
            folder=engine.ef_path(settings.get('output_dir') or OUT);folder.mkdir(parents=True,exist_ok=True)
            size=sum(FILES[f]['meta']['source_shape'][0]*FILES[f]['meta']['source_shape'][1]*(24 if settings.get('format','dng')=='dng' else 48) for f in ids)
            scratch_peak=max(FILES[f]['meta']['source_shape'][0]*FILES[f]['meta']['source_shape'][1] for f in ids)*(72 if settings.get('format','dng')=='dng' else 48)
            if shutil.disk_usage(folder).free<size+512*1024**2+(scratch_peak if folder.drive.upper()==ROOT.drive.upper() else 0):raise ValueError('导出盘空间不足，请选择其他目录。')
        result=[]
        for fid in dict.fromkeys(ids):
            jid=secrets.token_hex(8);source=FILES[fid]['path'];target=None
            if folder:
                ext='.dng' if settings.get('format','dng')=='dng' else '.tif'
                target=folder/(Path(source).stem+'_SonyExRAW_2x_'+time.strftime('%Y%m%d_%H%M%S')+'_'+jid[:6]+ext)
            TASKS[jid]=dict(id=jid,file_id=fid,name=FILES[fid]['name'],source=source,mode=mode,status='queued',stage='等待处理',completed=0,total=1,
                output=str(target) if target else None,settings=dict(settings),cancel=threading.Event(),center=center,created=time.time())
            result.append(jid);PENDING.put(jid)
        return dict(jobs=result)

def choose(kind):
    if kind not in ('files','folder'):raise ValueError('无效选择类型。')
    result=WORK/('dialog_'+secrets.token_hex(6)+'.json')
    if getattr(sys,'frozen',False):command=[sys.executable,'--pick',kind,str(result),str(OUT)]
    else:command=[sys.executable,'-B',str(BUNDLE/'main.py'),'--pick',kind,str(result),str(OUT)]
    process=subprocess.Popen(command,
        stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=0x08000000)
    try:
        process.wait(timeout=600)
        return json.loads(result.read_text(encoding='utf8')) if result.exists() else {'value':[] if kind=='files' else ''}
    finally:
        if process.poll() is None:process.terminate()
        if result.exists():result.unlink()

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def send(self,data,kind='application/json; charset=utf-8',status=200):
        if isinstance(data,(dict,list)):data=json.dumps(data,ensure_ascii=False).encode('utf8')
        if isinstance(data,str):data=data.encode('utf8')
        self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        try:self.wfile.write(data)
        except (ConnectionAbortedError,BrokenPipeError,ConnectionResetError):pass
    def allowed(self):return self.headers.get('Host')==f'127.0.0.1:{PORT}'
    def do_GET(self):
        global LAST
        if not self.allowed():self.send({'error':'Invalid host'},status=403);return
        LAST=time.monotonic();u=urlsplit(self.path)
        try:
            if u.path in ('/','/index.html'):
                self.send((BUNDLE/'ui.html').read_text(encoding='utf8').replace('__TOKEN__',TOKEN),'text/html; charset=utf-8')
            elif u.path=='/api/health':self.send(dict(app='SonyExRAWStudio',version=VERSION,pid=os.getpid()))
            elif u.path=='/api/state':self.send(state())
            elif re.fullmatch(r'/image/(file|job)/[a-f0-9]{16}/(thumb|before|after)\.jpg',u.path):
                _,_,group,key,name=u.path.split('/')
                name=name[:-4]
                p=THUMBS/(key+'.jpg') if group=='file' and name=='thumb' else JOBS/key/(name+'.jpg')
                if not p.is_file():self.send({'error':'预览尚未生成'},status=404)
                else:self.send(p.read_bytes(),'image/jpeg')
            elif u.path=='/api/result':
                jid=parse_qs(u.query).get('id',[''])[0];task=TASKS.get(jid)
                if not task or task['status']!='done' or not task.get('output'):raise ValueError('结果不可用。')
                p=Path(task['output']);self.send_response(200);self.send_header('Content-Type','application/octet-stream')
                from urllib.parse import quote
                self.send_header('Content-Disposition',"attachment; filename*=UTF-8''"+quote(p.name));self.send_header('Content-Length',str(p.stat().st_size));self.end_headers()
                with open(p,'rb') as f:shutil.copyfileobj(f,self.wfile,4*1024*1024)
            else:self.send({'error':'Not found'},status=404)
        except Exception as e:self.send({'error':str(e)},status=400)
    def do_POST(self):
        global LAST
        LAST=time.monotonic()
        origin=self.headers.get('Origin')
        if not self.allowed() or self.headers.get('X-Studio-Token')!=TOKEN or (origin and origin!=f'http://127.0.0.1:{PORT}'):
            self.send({'error':'Invalid request'},status=403);return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if length>256*1024:raise ValueError('请求过大。')
            body=json.loads(self.rfile.read(length));route=urlsplit(self.path).path
            if route=='/api/import':result=import_files(body['paths'])
            elif route=='/api/run':result=start_jobs(body)
            elif route=='/api/pick':result=choose(body.get('kind','files'))
            elif route=='/api/cancel':
                with LOCK:
                    for t in TASKS.values():
                        if t['status'] in ('queued','running'):t['cancel'].set();t['stage']='正在停止，等待当前图块完成'
                result={'ok':True}
            elif route=='/api/shutdown':
                with LOCK:
                    if ACTIVE or any(t['status']=='queued' for t in TASKS.values()):raise ValueError('请先停止队列，再退出程序。')
                    save_state()
                result={'ok':True}
                threading.Timer(.3,SERVER.shutdown).start()
            elif route=='/api/remove':
                with LOCK:
                    if ACTIVE or any(t['status']=='queued' for t in TASKS.values()):raise ValueError('请先等待或停止当前任务。')
                    fid=body.get('id')
                    if fid not in FILES:raise ValueError('文件不在当前列表。')
                    for key in [k for k,t in TASKS.items() if t['file_id']==fid]:
                        folder=JOBS/key
                        for name in ('before.jpg','after.jpg'):
                            p=folder/name
                            if p.exists():p.unlink()
                        if folder.is_dir():
                            try:folder.rmdir()
                            except OSError:pass
                        del TASKS[key]
                    p=THUMBS/(fid+'.jpg')
                    if p.exists():p.unlink()
                    del FILES[fid];save_state()
                result={'ok':True}
            elif route=='/api/open-output':
                p=engine.ef_path(body.get('path') or OUT)
                if p.is_file():p=p.parent
                if not p.is_dir():raise ValueError('输出文件夹尚未建立。')
                subprocess.Popen(['explorer.exe',str(p)],creationflags=0x08000000);result={'ok':True}
            else:raise ValueError('未知操作。')
            self.send(result)
        except Exception as e:self.send({'error':str(e)},status=400)

def main():
    global PORT,SERVER
    import msvcrt,urllib.request
    instance=open(WORK/'instance.lock','a+b');instance.seek(0);instance.write(b'0');instance.flush();instance.seek(0)
    try:msvcrt.locking(instance.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:
        try:
            existing=json.loads((WORK/'runtime.json').read_text(encoding='utf8'))
            health=json.load(urllib.request.urlopen(existing['url']+'api/health',timeout=2))
            if health.get('app')=='SonyExRAWStudio' and '--open' in sys.argv:webbrowser.open(existing['url'])
        except Exception:pass
        return
    server=None
    for port in range(52348,52358):
        try:server=LocalServer(('127.0.0.1',port),Handler);PORT=port;break
        except OSError:continue
    if server is None:raise RuntimeError('本地端口均被占用。')
    SERVER=server
    server.daemon_threads=True
    restore_state()
    runtime=dict(pid=os.getpid(),url=f'http://127.0.0.1:{PORT}/',root=str(ROOT),started=time.time())
    (WORK/'runtime.json').write_text(json.dumps(runtime,indent=2),encoding='utf8')
    threading.Thread(target=worker,daemon=True).start()
    def idle_stop():
        while not STOP.wait(30):
            if time.monotonic()-LAST>4*3600 and ACTIVE is None and PENDING.empty():STOP.set();server.shutdown()
    threading.Thread(target=idle_stop,daemon=True).start()
    if '--open' in sys.argv:webbrowser.open(runtime['url'])
    log.info('Started pid=%s port=%s',os.getpid(),PORT)
    try:server.serve_forever(poll_interval=.5)
    finally:STOP.set();server.server_close();instance.close()

if __name__=='__main__':main()
