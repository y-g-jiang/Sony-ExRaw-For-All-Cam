"""Executable-level, offline smoke tests using only bundled dependencies/assets."""
from pathlib import Path
import sys,json,threading,time

def run(folder,source=None):
    import numpy as np,rawpy,tifffile
    import engine
    from paths import BUNDLE,VERSION
    root=engine.ef_path(folder);root.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();cancel=threading.Event()
    if source is None:
        source=root/'synthetic_bayer.dng'
        yy,xx=np.mgrid[:96,:96]
        raw=np.rint(512+300+20*xx+10*yy).astype('uint16')
        tags=[(50706,'B',4,(1,4,0,0),False),(50707,'B',4,(1,1,0,0),False),
            (271,'s',0,'Synthetic',False),(272,'s',0,'Bayer smoke test',False),(50708,'s',0,'Bayer smoke test',False),
            (33421,'H',2,(2,2),False),(33422,'B',4,(0,1,1,2),False),
            (50713,'H',2,(2,2),False),(50714,'2I',4,engine.rat([512]*4),False),(50717,'I',1,16383,False),
            (50721,'2i',9,engine.rat(np.eye(3)),False),(50728,'2I',3,engine.rat([1,1,1]),False),(50778,'H',1,21,False)]
        tifffile.imwrite(source,raw,photometric=32803,metadata=None,extratags=tags)
    source=Path(source).resolve()
    with rawpy.imread(str(source)) as r:
        meta=engine.metadata(r,source);raw=r.raw_image_visible.copy()
    noise=engine.noise_settings({'profile':'s5'});packed,_=engine.packed(raw,meta,noise)
    assert packed.shape[0]==8 and np.isfinite(packed).all()
    for fy,fx in [(0,0),(0,1),(1,0),(1,1)]:
        q=raw.copy();p=np.array(meta['pattern']);colors=np.array(meta['colors'])
        if fy:q=q[::-1];p=p[::-1];colors=colors[::-1]
        if fx:q=q[:,::-1];p=p[:,::-1];colors=colors[:,::-1]
        z,_=engine.packed(q,dict(meta,pattern=p.tolist(),colors=colors.tolist()),noise)
        assert np.array_equal(z,packed)
    events=[]
    def progress(stage,n,total):
        if not events or events[-1]['stage']!=stage:events.append(dict(stage=stage))
        events[-1].update(completed=n,total=total)
    target=root/'inference.dng'
    result=engine.process(source,dict(profile='s5',format='dng',gain=1),root/'scratch',target,'export',progress,cancel)
    assert result['provider']=='WGPU' and result['backend']['strict'] and not result['backend']['cpu_fallback']
    assert result['export']['pixel_readback']=='ALL_PIXELS_EXACT'
    with tifffile.TiffFile(target) as f:
        tags=f.pages[0].tags
        def numeric(code):
            a=np.array(tags[code].value,float).reshape(-1,2);return a[:,0]/a[:,1]
        assert np.allclose(numeric(50721).reshape(3,3),meta['matrix'],atol=5e-6)
        assert np.allclose(numeric(50728),1/np.array(meta['wb']),atol=5e-6)
    with rawpy.imread(str(target)) as r:
        assert (r.sizes.raw_height,r.sizes.raw_width)==tuple(2*x for x in meta['source_shape'])
    rng=np.random.default_rng(44);small=rng.normal(.1,.09,(32,48,3)).astype('float32')
    smallmeta=dict(meta,crop=[0,0,24,16]);scratch=root/'small';scratch.mkdir(exist_ok=True)
    tiff=engine.encode_export(small,smallmeta,'tiff',root/'float.tif',scratch,lambda *a:None,cancel)
    dng=engine.encode_export(small,smallmeta,'dng',root/'negative.dng',scratch,lambda *a:None,cancel)
    assert tiff['pixel_readback']=='ALL_PIXELS_EXACT' and dng['black_pedestal']>0
    for p in scratch.glob('*.dat'):p.unlink()
    stop=threading.Event()
    def cancel_after_inference(stage,n,total):
        if stage=='超分处理中' and n==1:stop.set()
    try:
        engine.process(source,dict(profile='s5',format='dng',gain=1),root/'cancel',root/'cancelled.dng','export',cancel_after_inference,stop)
    except engine.Cancelled:pass
    else:raise AssertionError('Cancellation was ignored')
    assert not (root/'cancelled.dng').exists()
    assert not list((root/'cancel').glob('*.dat'))
    try:engine.process(source,{'profile':'s5'},root/'scratch',target,'export',lambda *a:None,cancel)
    except ValueError:pass
    else:raise AssertionError('Existing output was overwritten')
    model_sha=engine.sha(engine.MODEL);assert model_sha==engine.MODEL_SHA
    assert not any(name in sys.modules for name in ('onnxruntime','torch','tensorflow'))
    report=dict(passed=True,version=VERSION,frozen=bool(getattr(sys,'frozen',False)),
        strict_wgpu=True,cpu_fallback_imports=[],backend=engine.backend_info(),
        model_embedded_path=str(engine.MODEL),model_sha256=model_sha,bundle=str(BUNDLE),
        all_four_cfa_phases=True,cancel_cleanup=True,no_overwrite=True,dng_metadata_matches=True,
        inference=result,negative_dng=dng,float_tiff=tiff,events=events,seconds=time.perf_counter()-started)
    (root/'SELF_TEST.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    return report
