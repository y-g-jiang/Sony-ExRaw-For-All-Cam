"""Strict hardware-WGPU execution of the fixed Sony graph; no CPU evaluator.

ONNX is used only to parse the unchanged model. Every nontrivial graph operation
executes in FP32 WGSL. Unit Mul constants are validated and removed as identities.
"""
from pathlib import Path
from collections import Counter
import hashlib,time
import numpy as np
import wgpu

MODEL_SHA='43fd0e4566d39540c3e3912fda24f0e53bb7b509c2e9d18f8e01b4b94324ab22'

def validate_adapter(info):
    text=' '.join(str(v) for v in info.values()).lower()
    if info.get('adapter_type') not in ('DiscreteGPU','IntegratedGPU','VirtualGPU') or any(s in text for s in ('swiftshader','llvmpipe','lavapipe','software','basic render')):
        raise RuntimeError('需要硬件 WGPU 显卡；拒绝 CPU 或软件适配器。请检查显卡驱动。')

def conv_shader(cpt,spt,ks,stride,pad,k=12):
    """Implicit GEMM: 16x16 threads, explicit register blocking, <=16KiB shared."""
    ct=16*cpt;st=16*spt;we=ct*k;xe=k*st
    assert 4*(we+xe)<=16384 and xe%256==0
    lines=[f'''struct P {{ cin:u32, cout:u32, hi:u32, wi:u32, ho:u32, wo:u32, spatial:u32, inner:u32 }};
@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read> w:array<f32>;
@group(0) @binding(2) var<storage,read_write> y:array<f32>;
@group(0) @binding(3) var<uniform> p:P;
var<workgroup> tw:array<f32,{we}>;
var<workgroup> tx:array<f32,{xe}>;
@compute @workgroup_size(16,16,1)
fn main(@builtin(local_invocation_id) lid:vec3<u32>,@builtin(workgroup_id) wid:vec3<u32>) {{
let lane=lid.y*16u+lid.x;
''']
    add=lines.append
    for i in range(cpt):
        add(f'let oc{i}=wid.y*{ct}u+lid.y+{i*16}u;')
        for j in range(spt):add(f'var a{i}_{j}:f32=0.0;')
    add(f'for(var base:u32=0u;base<p.inner;base+={k}u){{')
    add(f'''for(var step:u32=0u;step<{(we+255)//256}u;step++){{
let z=step*256u+lane;
if(z<{we}u){{let oc=wid.y*{ct}u+z/{k}u;let kk=base+z%{k}u;
if(oc<p.cout && kk<p.inner){{tw[z]=w[oc*p.inner+kk];}}else{{tw[z]=0.0;}} }} }}
let col=lane%{st}u;let row=lane/{st}u;let sp=wid.x*{st}u+col;
let oy=sp/p.wo;let ox=sp%p.wo;
for(var step:u32=0u;step<{xe//256}u;step++){{
let q=row+step*{256//st}u;let kk=base+q;var v:f32=0.0;
if(sp<p.spatial && kk<p.inner){{
 let ic=kk/{ks*ks}u;let rem=kk%{ks*ks}u;
 let iy=i32(oy*{stride}u+rem/{ks}u)-{pad};let ix=i32(ox*{stride}u+rem%{ks}u)-{pad};
 if(iy>=0 && iy<i32(p.hi) && ix>=0 && ix<i32(p.wi)){{v=x[ic*p.hi*p.wi+u32(iy)*p.wi+u32(ix)];}}
}}
tx[q*{st}u+col]=v;
}}
workgroupBarrier();
for(var q:u32=0u;q<{k}u;q++){{''')
    for i in range(cpt):add(f'let w{i}=tw[(lid.y+{i*16}u)*{k}u+q];')
    for j in range(spt):add(f'let x{j}=tx[q*{st}u+lid.x+{j*16}u];')
    for i in range(cpt):
        for j in range(spt):add(f'a{i}_{j}=a{i}_{j}+w{i}*x{j};')
    add('} workgroupBarrier(); }')
    for j in range(spt):add(f'let sp{j}=wid.x*{st}u+lid.x+{j*16}u;')
    for i in range(cpt):
        for j in range(spt):add(f'if(oc{i}<p.cout && sp{j}<p.spatial){{y[oc{i}*p.spatial+sp{j}]=a{i}_{j};}}')
    add('}')
    return '\n'.join(lines)

ELEMENT='''struct P { n:u32, mode:u32, pitch:u32, pad:u32 };
@group(0) @binding(0) var<storage,read> a:array<f32>;
@group(0) @binding(1) var<storage,read> b:array<f32>;
@group(0) @binding(2) var<storage,read_write> c:array<f32>;
@group(0) @binding(3) var<uniform> p:P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.y*p.pitch+gid.x;if(i>=p.n){return;}
if(p.mode==0u){c[i]=a[i]+b[i];}else{c[i]=max(a[i],0.0);}}
'''
SHUFFLE='''struct P { n:u32, cout:u32, hi:u32, wi:u32, pitch:u32, p0:u32, p1:u32, p2:u32 };
@group(0) @binding(0) var<storage,read> a:array<f32>;
@group(0) @binding(1) var<storage,read_write> b:array<f32>;
@group(0) @binding(2) var<uniform> p:P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.y*p.pitch+gid.x;if(i>=p.n){return;}
let wo=p.wi*2u;let spatial=p.hi*p.wi*4u;let oc=i/spatial;let sp=i%spatial;
let oy=sp/wo;let ox=sp%wo;let phase=(oy%2u)*2u+ox%2u;
b[i]=a[(phase*p.cout+oc)*p.hi*p.wi+(oy/2u)*p.wi+ox/2u];}
'''

class WGPUModel:
    def __init__(self,path):
        import onnx
        from onnx import numpy_helper,helper
        started=time.perf_counter();data=Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest()!=MODEL_SHA:raise ValueError('Sony原模型校验不符。')
        graph=onnx.load_model_from_string(data[:-2]).graph
        weights={w.name:numpy_helper.to_array(w).copy() for w in graph.initializer}
        if len(weights)!=58:raise ValueError('Sony权重张量数不符。')
        aliases={};constants={};self.ops=[];self.weight_arrays={}
        def resolve(name):
            while name in aliases:name=aliases[name]
            return name
        for node in graph.node:
            attrs={a.name:helper.get_attribute_value(a) for a in node.attribute}
            inputs=[resolve(x) for x in node.input];out=node.output[0]
            if node.op_type=='Constant':
                constants[out]=numpy_helper.to_array(attrs['value']);continue
            if node.op_type=='Mul':
                if len(inputs)!=2 or inputs[1] not in constants or not np.all(constants[inputs[1]]==1):raise ValueError('仅允许已验证的单位残差系数。')
                aliases[out]=inputs[0];continue
            if node.op_type in ('Conv','ConvTranspose'):
                if len(inputs)!=2 or attrs.get('group',1)!=1 or attrs.get('dilations',[1,1])!=[1,1]:raise ValueError('模型卷积配置尚未支持。')
                weight=weights[inputs[1]];ks=int(weight.shape[2]);stride=attrs['strides'][0];pad=attrs['pads'][0]
                if node.op_type=='ConvTranspose':
                    if ks!=2 or stride!=2 or any(attrs['pads']) or any(attrs['output_padding']):raise ValueError('转置卷积配置尚未支持。')
                    ci,co=weight.shape[:2];weight=weight.transpose(2,3,1,0).reshape(4*co,ci,1,1).copy()
                elif (ks,stride,pad) not in ((3,1,1),(2,2,0)):raise ValueError('卷积配置尚未支持。')
                self.weight_arrays[inputs[1]]=weight
                self.ops.append(dict(op=node.op_type,inputs=[inputs[0]],weight=inputs[1],output=out,ks=ks,stride=stride,pad=pad))
            elif node.op_type in ('Add','Relu'):self.ops.append(dict(op=node.op_type,inputs=inputs,output=out))
            else:raise ValueError('未支持的模型操作：'+node.op_type)
        counts=Counter(o['op'] for o in self.ops)
        if counts!=dict(Conv=53,ConvTranspose=5,Relu=24,Add=28):raise ValueError('模型计算图不符。')
        self.input=graph.input[0].name;self.output=resolve(graph.output[0].name)
        self.uses=Counter(x for op in self.ops for x in op['inputs'])
        try:self.adapter=wgpu.gpu.request_adapter_sync(power_preference='high-performance',force_fallback_adapter=False)
        except Exception as exc:raise RuntimeError('WGPU初始化失败；请检查硬件显卡及驱动。没有CPU推理路径。') from exc
        if self.adapter is None:raise RuntimeError('未找到硬件 WGPU 适配器。')
        info=dict(self.adapter.info);validate_adapter(info)
        maximum=int(self.adapter.limits['max-storage-buffer-binding-size'])
        self.tile=512 if maximum>=268435456 else 352
        binding=268435456 if self.tile==512 else 134217728
        self.device=self.adapter.request_device_sync(required_limits={'max-storage-buffer-binding-size':binding,'max-buffer-size':binding,
            'max-compute-workgroup-storage-size':16384,'max-compute-invocations-per-workgroup':256})
        self.weights={name:self.device.create_buffer_with_data(data=value,usage=wgpu.BufferUsage.STORAGE) for name,value in self.weight_arrays.items()}
        self.weight_shapes={name:value.shape for name,value in self.weight_arrays.items()};del self.weight_arrays,weights,graph,data
        self.pipelines={};self.element=self.pipeline(ELEMENT);self.shuffle=self.pipeline(SHUFFLE)
        self.free={};self.pool=[];self.allocated=0;self.peak=0;self.forwards=0;self.readbacks=0
        self.info=dict(provider='WGPU',strict=True,cpu_fallback=False,precision='FP32',adapter=info,tile=self.tile,
            graph_operations=dict(counts),weighted_gpu_operations=58,initialization_seconds=time.perf_counter()-started)

    def pipeline(self,code):
        return self.device.create_compute_pipeline(layout='auto',compute={'module':self.device.create_shader_module(code=code),'entry_point':'main'})

    def allocate(self,shape):
        size=int(np.prod(shape))*4
        if size>int(self.device.limits['max-storage-buffer-binding-size']):raise RuntimeError('张量超出 WGPU 缓冲区上限。')
        available=self.free.setdefault(size,[])
        if available:buf=available.pop()
        else:
            buf=self.device.create_buffer(size=size,usage=wgpu.BufferUsage.STORAGE|wgpu.BufferUsage.COPY_SRC|wgpu.BufferUsage.COPY_DST)
            self.pool.append(buf);self.allocated+=size;self.peak=max(self.peak,self.allocated)
        return (buf,tuple(shape),size)

    def release(self,value):self.free.setdefault(value[2],[]).append(value[0])

    @staticmethod
    def grid(n):
        groups=(n+255)//256;gx=min(groups,65535);return (gx,(groups+gx-1)//gx,1),gx*256

    def dispatch(self,encoder,pipeline,buffers,values,grid,uniforms):
        ub=self.device.create_buffer_with_data(data=np.array(values,dtype='uint32'),usage=wgpu.BufferUsage.UNIFORM);uniforms.append(ub)
        bg=self.device.create_bind_group(layout=pipeline.get_bind_group_layout(0),entries=[dict(binding=i,resource={'buffer':b,'offset':0,'size':b.size}) for i,b in enumerate([*buffers,ub])])
        compute=encoder.begin_compute_pass();compute.set_pipeline(pipeline);compute.set_bind_group(0,bg);compute.dispatch_workgroups(*grid);compute.end()

    def conv(self,encoder,value,name,ks,stride,pad,uniforms):
        ci,hi,wi=value[1];co,wci,_,_=self.weight_shapes[name]
        if ci!=wci:raise ValueError('WGPU卷积输入通道不符。')
        ho=(hi+2*pad-ks)//stride+1;wo=(wi+2*pad-ks)//stride+1
        out=self.allocate((co,ho,wo))
        cpt,spt,k=(1,16,14) if co<=16 else (2,16,12) if co<=32 else (4,16,12) if co<=64 else (8,8,16)
        key=(cpt,spt,k,ks,stride,pad)
        if key not in self.pipelines:self.pipelines[key]=self.pipeline(conv_shader(cpt,spt,ks,stride,pad,k))
        self.dispatch(encoder,self.pipelines[key],[value[0],self.weights[name],out[0]],
            [ci,co,hi,wi,ho,wo,ho*wo,ci*ks*ks],((ho*wo+16*spt-1)//(16*spt),(co+16*cpt-1)//(16*cpt),1),uniforms)
        return out

    def forward(self,x,capture=False):
        x=np.ascontiguousarray(x,dtype='float32')
        if x.ndim!=3 or x.shape[0]!=8 or x.shape[1]%8 or x.shape[2]%8:raise ValueError('WGPU输入须为8×H×W，宽高为8的倍数。')
        original=self.allocate(x.shape);self.device.queue.write_buffer(original[0],0,x)
        vals={self.input:original};remaining=self.uses.copy();encoder=self.device.create_command_encoder();uniforms=[];captures={};submitted=False
        try:
            for op in self.ops:
                inputs=[vals[name] for name in op['inputs']];kind=op['op']
                if kind=='Conv':out=self.conv(encoder,inputs[0],op['weight'],op['ks'],op['stride'],op['pad'],uniforms)
                elif kind=='ConvTranspose':
                    tmp=self.conv(encoder,inputs[0],op['weight'],1,1,0,uniforms)
                    co4,hi,wi=tmp[1];out=self.allocate((co4//4,hi*2,wi*2));grid,pitch=self.grid(out[2]//4)
                    self.dispatch(encoder,self.shuffle,[tmp[0],out[0]],[out[2]//4,co4//4,hi,wi,pitch,0,0,0],grid,uniforms);self.release(tmp)
                else:
                    if kind=='Add' and inputs[0][1]!=inputs[1][1]:raise ValueError('WGPU残差形状不符。')
                    out=self.allocate(inputs[0][1]);grid,pitch=self.grid(out[2]//4)
                    self.dispatch(encoder,self.element,[inputs[0][0],inputs[-1][0],out[0]],[out[2]//4,int(kind=='Relu'),pitch,0],grid,uniforms)
                vals[op['output']]=out
                if capture:
                    self.device.queue.submit([encoder.finish()]);submitted=True
                    captures[op['output']]=np.frombuffer(self.device.queue.read_buffer(out[0]),dtype='float32').copy().reshape(out[1]);self.readbacks+=1
                    encoder=self.device.create_command_encoder();submitted=False
                for name in op['inputs']:
                    remaining[name]-=1
                    if remaining[name]==0:self.release(vals.pop(name))
            self.device.queue.submit([encoder.finish()]);submitted=True
            value=vals[self.output];result=np.frombuffer(self.device.queue.read_buffer(value[0]),dtype='float32').copy().reshape(value[1]);self.readbacks+=1
            if not np.isfinite(result).all():raise RuntimeError('WGPU输出包含非有限数值。')
            self.forwards+=1;return (result,captures) if capture else result
        finally:
            # A failed encoding is never submitted; successful work is complete after readback.
            if submitted:self.device.queue.on_submitted_work_done_sync()
            for value in vals.values():self.release(value)
            for ub in uniforms:ub.destroy()

    def report(self):return dict(self.info,gpu_buffer_bytes=self.allocated,gpu_peak_bytes=self.peak,forwards=self.forwards,readbacks=self.readbacks)

    def close(self):
        self.device.queue.on_submitted_work_done_sync()
        for buf in [*self.pool,*self.weights.values()]:buf.destroy()
        self.pool.clear();self.free.clear();self.weights.clear()
