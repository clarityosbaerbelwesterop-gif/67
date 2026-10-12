// VGPU node on WebGPU: any browser (iPad/iPhone Safari 26+, Chrome, Edge) or Node+Dawn
// adds its GPU to the forge pool. It answers the coordinator's proof-of-compute
// challenges (headcenter/backend/vgpu.py) entirely on the GPU: generate A0 and B
// from the seed (lowbias32 hash, identical in numpy/JS/WGSL), k chained products
// C = A @ B in f32 (entries in {-1,0,1}, so every value is an exact integer),
// A <- (C mod 3) - 1, then the sha256 of every row of C_k. The coordinator checks
// random rows and times the work itself.

const T = 16;

const GEN = /* wgsl */ `
struct P { n: u32, s: u32 }
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> p: P;
fn mix32(v: u32) -> u32 {
  var x = v; x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u; return x;
}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = g.x + g.y * nw.x * 256u;
  if (i >= p.n * p.n) { return; }
  out[i] = f32(mix32(mix32(i) ^ p.s) % 3u) - 1.0;
}`;

const MATMUL = /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
var<workgroup> ta: array<array<f32, ${T}>, ${T}>;
var<workgroup> tb: array<array<f32, ${T}>, ${T}>;
@compute @workgroup_size(${T}, ${T}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let row = g.y; let col = g.x; let n = p.n;
  var acc = 0.0;
  for (var t = 0u; t < n; t += ${T}u) {
    ta[l.y][l.x] = a[row * n + t + l.x];
    tb[l.y][l.x] = b[(t + l.y) * n + col];
    workgroupBarrier();
    for (var q = 0u; q < ${T}u; q++) { acc += ta[l.y][q] * tb[q][l.x]; }
    workgroupBarrier();
  }
  c[row * n + col] = acc;
}`;

const STEP = /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<storage, read> c: array<f32>;
@group(0) @binding(1) var<storage, read_write> a: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = g.x + g.y * nw.x * 256u;
  if (i >= p.n * p.n) { return; }
  let v = i32(c[i]);
  a[i] = f32(((v % 3) + 3) % 3) - 1.0;
}`;

/** The largest power-of-two n whose n x n f32 matrix fits one storage binding. */
export function maxN(device) {
  const cap = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
  let n = 256;
  while ((n * 2) * (n * 2) * 4 <= cap && n < 8192) n *= 2;
  return n;
}

function grid(total) {
  const groups = Math.ceil(total / 256);
  const x = Math.min(groups, 65535);
  return [x, Math.ceil(groups / x)];
}

/** Row digests (hex sha256 of each row's int32 bytes) of C_k for the challenge. */
export async function solve(device, seed, n, k) {
  const bytes = n * n * 4;
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const [A, B, C] = [0, 1, 2].map(() => device.createBuffer({ size: bytes, usage }));
  const uni = (vals) => {
    const buf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf, 0, new Uint32Array([...vals, 0, 0, 0].slice(0, 4)));
    return buf;
  };
  const pipe = (code) => device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
  const gen = pipe(GEN), mm = pipe(MATMUL), st = pipe(STEP);
  const enc = device.createCommandEncoder();
  const run = (p, entries, wg) => {
    const pass = enc.beginComputePass();
    pass.setPipeline(p);
    pass.setBindGroup(0, device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: entries.map((buffer, binding) => ({ binding, resource: { buffer } })) }));
    pass.dispatchWorkgroups(...wg);
    pass.end();
  };
  const g = grid(n * n);
  run(gen, [A, uni([n, seed >>> 0])], g); // A0: s = seed
  run(gen, [B, uni([n, (seed ^ 0x9e3779b9) >>> 0])], g); // B: s = seed ^ 0x9e3779b9
  const nU = uni([n]);
  for (let r = 0; r < k; r++) {
    run(mm, [A, B, C, nU], [n / T, n / T]);
    if (r + 1 < k) run(st, [C, A, nU], g);
  }
  const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  enc.copyBufferToBuffer(C, 0, read, 0, bytes);
  device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const c = new Float32Array(read.getMappedRange().slice(0));
  read.unmap();
  [A, B, C, read].forEach((b) => b.destroy());
  const rows = [];
  for (let i = 0; i < n; i++) {
    const row = Int32Array.from(c.subarray(i * n, (i + 1) * n));
    const h = new Uint8Array(await crypto.subtle.digest("SHA-256", row.buffer));
    rows.push(Array.from(h, (x) => x.toString(16).padStart(2, "0")).join(""));
  }
  return rows;
}

/** Join the pool and answer challenges every `every` seconds (once with once=true). */
export async function serve(device, { headcenter, token, label, every = 1800, once = false, log = console.log }) {
  const api = async (method, path, body) => {
    const r = await fetch(headcenter.replace(/\/$/, "") + path, {
      method, headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!r.ok) throw new Error(`${method} ${path}: HTTP ${r.status} ${await r.text()}`);
    return r.json();
  };
  const { node_id } = await api("POST", "/api/vgpu/register", { label, kind: "webgpu", claimed_tflops: 0.5, max_n: maxN(device) });
  log({ registered: node_id, max_n: maxN(device) });
  for (;;) {
    const ch = await api("GET", `/api/vgpu/challenge?node_id=${node_id}`);
    const rows = await solve(device, ch.seed, ch.n, ch.k);
    const res = await api("POST", "/api/vgpu/proof", { challenge_id: ch.challenge_id, rows });
    log({ n: ch.n, k: ch.k, ...res });
    if (once) return res;
    await new Promise((ok) => setTimeout(ok, every * 1000));
  }
}
