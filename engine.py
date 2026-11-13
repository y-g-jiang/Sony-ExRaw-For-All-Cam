"""Local Sony ExRAW engine. Fixed weights; real per-file LibRaw levels."""
from paths import APP_HOME, BUNDLE, writable_path
from pathlib import Path
import os,sys,json,math,hashlib,threading,time,struct,gc
from fractions import Fraction
import numpy as np
import rawpy,cv2,tifffile
from PIL import Image
from color_preview import thumbnail_matrix
ROOT=APP_HOME
MODEL=BUNDLE/'models/exraw_hires_v3.48.onnx'
MODEL_SHA='43fd0e4566d39540c3e3912fda24f0e53bb7b509c2e9d18f8e01b4b94324ab22'
S5=json.loads((BUNDLE/'data/s5_iso100_ptc.json').read_text(encoding='utf8'))['record']
_session=None

class Cancelled(Exception): pass

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def ef_path(path):
    return writable_path(path)


def check_cancel(cancel):
    if cancel.is_set():raise Cancelled('任务已停止')

def tiff_identity(path):
    """Bounded classic-TIFF metadata read, including Panasonic ISO tag."""
    with open(path,'rb') as f:b=f.read(1024*1024)
    out={}
    if b[:2] not in (b'II',b'MM') or len(b)<8:return out
    endian='<' if b[:2]==b'II' else '>'
    def ifd(off):
        if off<8 or off+2>len(b):return {}
        n=struct.unpack_from(endian+'H',b,off)[0]
        if n>4096 or off+2+12*n>len(b):return {}
        tags={}
        for i in range(n):
            pos=off+2+12*i;code,typ,count,value=struct.unpack_from(endian+'HHII',b,pos)
            if typ==2 and 0<count<4096:
                start=pos+8 if count<=4 else value
                tags[code]=b[start:start+count].split(b'\0')[0].decode('utf8','replace')
            elif typ in (3,4) and count==1:
                tags[code]=struct.unpack_from(endian+('H' if typ==3 else 'I'),b,pos+8)[0]
        return tags
    try:
        tags=ifd(struct.unpack_from(endian+'I',b,4)[0]); ex=ifd(tags.get(34665,0))
        out=dict(make=str(tags.get(271,'')),model=str(tags.get(272,'')),iso=ex.get(34855,tags.get(23)))
    except (struct.error,ValueError):pass
    return out

def metadata(r,path):
    pattern=r.raw_pattern
    if pattern is None or pattern.shape!=(2,2):raise ValueError('当前支持普通 2×2 Bayer RAW；不支持 X-Trans、单色或线性 RGB 输入。')
    desc=r.color_desc.decode('ascii','replace').rstrip('\0')
    colors=np.array([[desc[int(v)] for v in row] for row in pattern])
    if sorted(colors.ravel())!=['B','G','G','R']:raise ValueError('无法识别此 CFA 的 RGB 通道。')
    if not desc.startswith('RGB'):raise ValueError('当前相机颜色通道顺序尚未验证。')
    h,w=r.raw_image_visible.shape
    if min(h,w)<32 or h%2 or w%2:raise ValueError('当前支持宽高均为偶数、至少32像素的 Bayer RAW。')
    black=np.array(r.black_level_per_channel,float)
    raw_white=r.camera_white_level_per_channel
    white=np.array(raw_white if raw_white is not None else [r.white_level]*4,float)
    # LibRaw may return zero for unavailable phase whites; fall back to this file's global readout.
    white=np.where(white>0,white,float(r.white_level))
    if not np.isfinite([*black,*white]).all() or np.any(white<=black):raise ValueError('此文件的黑白位读值无效。')
    wb=np.array(r.camera_whitebalance[:3],float)
    if not np.isfinite(wb).all() or min(wb)<=0:raise ValueError('文件没有可用的相机白平衡。')
    wb/=wb[1];matrix=r.rgb_xyz_matrix[:3,:3].astype(float)
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix))<1e-8:raise ValueError('文件没有可用的相机颜色矩阵。')
    s=r.sizes;crop=[s.crop_left_margin,s.crop_top_margin,s.crop_width,s.crop_height]
    if crop[2]<=0 or crop[3]<=0 or crop[0]+crop[2]>w or crop[1]+crop[3]>h:crop=[0,0,w,h]
    result=dict(source=str(Path(path).resolve()),source_shape=[h,w],pattern=pattern.tolist(),colors=colors.tolist(),
        black=black.tolist(),white=white.tolist(),wb=wb.tolist(),matrix=matrix.tolist(),crop=crop,flip=int(s.flip),
        libraw=list(rawpy.libraw_version),**tiff_identity(path))
    return result

class Profile:
    def __init__(self,meta):self.CM1=np.array(meta['matrix']);self.ASN=(1/np.array(meta['wb'])).tolist()
    def dng_ifd0_tags(self):return []

def display(rgb,meta,max_side=1600,ev=2):
    h,w=rgb.shape[:2];step=max(1,math.ceil(max(h,w)/max_side))
    small=np.asarray(rgb[::step,::step],np.float32)
    mat=thumbnail_matrix(Profile(meta))
    linear=np.maximum(small@mat.T,0)*2**ev
    srgb=np.where(linear<=.0031308,linear*12.92,1.055*np.maximum(linear,0)**(1/2.4)-.055)
    return np.rint(np.clip(srgb,0,1)*255).astype('uint8')

def quick_preview(raw,meta):
    p=np.array(meta['pattern']);bl=np.array(meta['black']);colors=np.array(meta['colors'])
    planes=[(raw[y::2,x::2].astype(np.float32)-bl[p[y,x]])/16384 for y,x in [(0,0),(0,1),(1,0),(1,1)]]
    rgb=np.stack([np.mean([planes[i] for i,c in enumerate(colors.ravel()) if c==letter],axis=0) for letter in 'RGB'],axis=-1)
    return display(rgb,meta,1200)

def inspect(path,thumb):
    path=Path(path).resolve()
    if not path.is_file():raise ValueError('找不到 RAW 文件。')
    with rawpy.imread(str(path)) as r:
        meta=metadata(r,path);im=quick_preview(r.raw_image_visible,meta)
    Image.fromarray(im).save(thumb,quality=90)
    h,w=meta['source_shape'];meta['output_shape']=[h*2,w*2];meta['bytes']=path.stat().st_size
    return meta

def noise_settings(settings):
    preset=settings.get('profile','s5')
    coeff=[S5['alpha'],S5['beta_total'],S5['c']] if preset=='s5' else [float(settings[k]) for k in ('alpha','beta','c')]
    gain=float(settings.get('gain',1))
    if not np.isfinite([*coeff,gain]).all() or min(coeff)<0 or not 0<=gain<=3:raise ValueError('PTC系数须非负且有限；噪声强度范围0–3。')
    if coeff[0]>1 or coeff[1]>1 or coeff[2]>1:raise ValueError('PTC系数单位应为归一化满阱单位，请检查数量级。')
    return dict(profile='S5 ISO100 PTC #1874 (explicit prior; S5II proxy)' if preset=='s5' else 'Manual fullwell-normalized PTC',
        alpha=coeff[0],beta=coeff[1],c=coeff[2],gain=gain)

def packed(raw,meta,noise):
    pattern=np.array(meta['pattern']);black=np.array(meta['black'],np.float32);white=np.array(meta['white'],np.float32)
    colors=np.array(meta['colors']);red=np.argwhere(colors=='R')[0];fy,fx=map(bool,red)
    planes=[];sigmas=[]
    for y,x in [(0,0),(0,1),(1,0),(1,1)]:
        channel=pattern[y,x];signal=(raw[y::2,x::2].astype(np.float32)-black[channel])/16384
        span=white[channel]-black[channel];u=np.maximum(signal,0)*(16384/span)
        sigma=np.sqrt(noise['alpha']*u+noise['beta']+noise['c']*u*u)*(span/16384)*noise['gain']
        planes.append(signal);sigmas.append(sigma)
    z=np.concatenate([np.stack(planes),np.stack(sigmas)])
    # Canonicalize CFA to RGGB by reflecting both the spatial grid and the phase positions.
    if fy:z=z[[2,3,0,1,6,7,4,5],::-1,:]
    if fx:z=z[[1,0,3,2,5,4,7,6],:,::-1]
    return z,(fy,fx)

def session():
    global _session
    if _session is None:
        from wgpu_model import WGPUModel
        _session=WGPUModel(MODEL)
    return _session

def backend_info():
    return _session.report() if _session is not None else dict(provider='WGPU',strict=True,cpu_fallback=False,status='not_initialized')


def infer(raw,meta,noise,floatpath,progress,cancel):
    progress('初始化 WGPU',0,1);s=session();tile=s.tile;halo=112;core=tile-2*halo
    x,flips=packed(raw,meta,noise);_,h,w=x.shape
    ph=math.ceil(h/core)*core;pw=math.ceil(w/core)*core
    pad=np.pad(x,((0,0),(halo,halo+ph-h),(halo,halo+pw-w)),mode='reflect');del x
    progress('WGPU 已就绪',0,1)
    shape=(h*4,w*4,3);out=np.memmap(floatpath,dtype='float32',mode='w+',shape=shape)
    try:
        total=math.ceil(h/core)*math.ceil(w/core);n=0
        for y in range(0,h,core):
            for xx in range(0,w,core):
                check_cancel(cancel);progress('超分处理中',n,total)
                z=s.forward(pad[:,y:y+tile,xx:xx+tile].copy())
                check_cancel(cancel)
                ch=min(core,h-y);cw=min(core,w-xx)
                out[y*4:(y+ch)*4,xx*4:(xx+cw)*4]=z[:,halo*4:halo*4+ch*4,halo*4:halo*4+cw*4].transpose(1,2,0)
                n+=1;progress('超分处理中',n,total)
        out.flush()
    finally:del out,pad
    result=np.memmap(floatpath,dtype='float32',mode='r',shape=shape)
    if flips[0]:result=result[::-1,:,:]
    if flips[1]:result=result[:,::-1,:]
    return result

def rat(values):
    out=[]
    for v in np.asarray(values).ravel():
        q=Fraction(float(v)).limit_denominator(1000000);out.extend((q.numerator,q.denominator))
    return tuple(out)

def write_dng(path,pixels,meta,pedestal,baseline,thumb):
    h,w=pixels.shape[:2];crop=[int(x*2) for x in meta['crop']]
    orientation={0:1,3:3,5:8,6:6}.get(meta.get('flip',0),1)
    make=meta.get('make') or 'RAW camera';model=meta.get('model') or 'Bayer RAW'
    tags=[(50706,'B',4,(1,4,0,0),False),(50707,'B',4,(1,3,0,0),False),
        (271,'s',0,make,False),(272,'s',0,model,False),(50708,'s',0,make+' '+model,False),
        (274,'H',1,orientation,False),(50721,'2i',9,rat(meta['matrix']),False),
        (50728,'2I',3,rat(1/np.array(meta['wb'])),False),(50730,'2i',1,rat([baseline]),False),(50778,'H',1,21,False)]
    rawtags=[(50714,'2I',3,rat([pedestal]*3),False),(50717,'I',3,(65535,)*3,False),
        (50719,'I',2,tuple(crop[:2]),False),(50720,'I',2,tuple(crop[2:]),False),(274,'H',1,orientation,False)]
    with tifffile.TiffWriter(path,bigtiff=(h*w*6>3_900_000_000)) as t:
        t.write(thumb,photometric='rgb',subfiletype=1,subifds=1,metadata=None,software='Sony ExRAW Studio 1.1.0 WGPU',extratags=tags)
        t.write(pixels,photometric=34892,planarconfig='contig',rowsperstrip=128,subfiletype=0,metadata=None,extratags=rawtags)

def encode_export(rgb,meta,kind,target,scratch,progress,cancel):
    h,w=rgb.shape[:2];lo=0.;hi=1.
    for y in range(0,h,128):
        check_cancel(cancel);q=rgb[y:y+128]
        if not np.isfinite(q).all():raise ValueError('模型输出包含非有限值。')
        lo=min(lo,float(q.min()));hi=max(hi,float(q.max()))
    temp=target.with_name('.'+target.name+'.partial')
    try:
        if kind=='tiff':
            progress('写入线性 TIFF',0,1)
            tifffile.imwrite(temp,rgb,photometric='rgb',rowsperstrip=128,
                metadata=None,description=json.dumps(dict(color_space='Linear camera RGB, no gamma/WB baked',source=meta,model_sha256=MODEL_SHA),ensure_ascii=True))
            decoded=tifffile.memmap(temp,mode='r')
            for y in range(0,h,128):
                check_cancel(cancel)
                if not np.array_equal(decoded[y:y+128],rgb[y:y+128]):raise ValueError('TIFF像素回读不一致。')
            del decoded
            contract=dict(format='32-bit float linear camera RGB TIFF',min=lo,max=hi,pixel_readback='ALL_PIXELS_EXACT')
        else:
            pedestal=math.ceil(-lo*(65535/(hi-lo)));scale=(65535-pedestal)/hi
            baseline=round(2+math.log2(hi),2)
            quantpath=scratch/'quant.dat';qmap=np.memmap(quantpath,dtype='uint16',mode='w+',shape=rgb.shape)
            try:
                for y in range(0,h,128):
                    check_cancel(cancel);progress('编码线性 DNG',y,h)
                    q=np.rint(rgb[y:y+128].astype(np.float64)*scale+pedestal)
                    if q.min()<0 or q.max()>65535:raise ValueError('DNG编码超出16位范围。')
                    qmap[y:y+128]=q.astype('uint16')
                qmap.flush();check_cancel(cancel)
                thumb=display(rgb,meta,256);write_dng(temp,qmap,meta,pedestal,baseline,thumb)
                decoded=tifffile.memmap(temp,series=1,mode='r');maxerr=0.
                try:
                    for y in range(0,h,128):
                        check_cancel(cancel);progress('校验导出像素',y,h)
                        if not np.array_equal(decoded[y:y+128],qmap[y:y+128]):raise ValueError('DNG像素回读不一致。')
                        maxerr=max(maxerr,float(np.max(np.abs((decoded[y:y+128].astype(np.float64)-pedestal)/scale-rgb[y:y+128]))))
                finally:del decoded
                if maxerr>.5001/scale:raise ValueError('DNG量化误差超出半码值。')
                with rawpy.imread(str(temp)) as r:
                    if (r.sizes.raw_height,r.sizes.raw_width)!=(h,w):raise ValueError('LibRaw无法读取导出的尺寸。')
                contract=dict(format='16-bit linear RGB DNG',black_pedestal=pedestal,code_scale=scale,baseline_exposure=baseline,
                    float_reconstruction_max_abs=maxerr,quantization_half_step=.5/scale,pixel_readback='ALL_PIXELS_EXACT',
                    negative_preserved_below_black=True,wb_gamma_baked=False,min=lo,max=hi)
            finally:del qmap
        check_cancel(cancel)
        # Hard-link commit is atomic and cannot overwrite another output with the same name.
        os.link(temp,target);temp.unlink()
        return contract
    finally:
        if temp.exists():temp.unlink()

def process(path,settings,scratch,target,mode,progress,cancel,center=(.5,.5)):
    scratch=ef_path(scratch);scratch.mkdir(parents=True,exist_ok=True)
    target=ef_path(target) if target else None
    if target and target.exists():raise ValueError('输出已存在，请换一个文件名。')
    start=time.perf_counter();noise=noise_settings(settings);progress('读取 RAW',0,1)
    rgb=None
    try:
        with rawpy.imread(str(path)) as r:
            meta=metadata(r,path);raw=r.raw_image_visible.copy()
            if mode=='preview':
                hh,ww=raw.shape;side=min(512,hh,ww);side-=side%16
                x=max(0,min(ww-side,int(float(center[0])*ww-side/2)));x=x//16*16
                y=max(0,min(hh-side,int(float(center[1])*hh-side/2)));y=y//16*16
                # Include the same tested 112-packed-pixel halo from the real surrounding RAW.
                halo_native=224;left=max(0,x-halo_native);top=max(0,y-halo_native)
                right=min(ww,x+side+halo_native);bottom=min(hh,y+side+halo_native)
                roi=[x,y,side,side];outer=[left,top,right,bottom]
                raw=raw[top:bottom,left:right].copy()
                base=r.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,user_wb=[1]*4,
                    use_camera_wb=False,use_auto_wb=False,output_color=rawpy.ColorSpace.raw,
                    gamma=(1,1),no_auto_bright=True,output_bps=16,user_flip=0).astype(np.float32)/65535
                span=float(np.mean(np.array(meta['white'])-np.array(meta['black'])))
                before=cv2.resize(base[y:y+side,x:x+side]*(span/16384),(side*2,side*2),interpolation=cv2.INTER_CUBIC)
            else:roi=None;outer=None
        check_cancel(cancel);rgb=infer(raw,meta,noise,scratch/'rgb.dat',progress,cancel);del raw
        check_cancel(cancel)
        if mode=='preview':
            left,top,right,bottom=outer;x,y,side,_=roi
            rgb=rgb[(y-top)*2:(y-top+side)*2,(x-left)*2:(x-left+side)*2]
            Image.fromarray(display(before,meta)).save(scratch/'before.jpg',quality=96)
            Image.fromarray(display(rgb,meta)).save(scratch/'after.jpg',quality=96)
            check_cancel(cancel)
            export=None
        else:
            progress('生成预览',0,1);Image.fromarray(display(rgb,meta)).save(scratch/'after.jpg',quality=94)
            export=encode_export(rgb,meta,settings.get('format','dng'),target,scratch,progress,cancel)
        report=dict(source=meta,source_sha256=sha(path),model_sha256=MODEL_SHA,noise=noise,
            mode=mode,roi_native=roi,preview_outer_native=outer,output=str(target) if target else None,
            output_sha256=sha(target) if target else None,output_shape=list(rgb.shape),export=export,
            seconds=time.perf_counter()-start,provider='WGPU',backend=backend_info(),
            tiling=dict(packed_tile=session().tile,halo=112,core=session().tile-224),display_exposure_ev=2)
        if target:target.with_suffix(target.suffix+'.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        return report
    finally:
        del rgb
        gc.collect()
        for p in [scratch/'rgb.dat',scratch/'quant.dat']:
            if p.exists():p.unlink()
