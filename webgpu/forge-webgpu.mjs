// forge-webgpu — runs forge / SCP-1 checkpoints on any WebGPU device
// (iPad and iPhone Safari 26+, Chrome, Edge, Node with Dawn). Pure JS + WGSL,
// no dependencies. Same equations as core_engine/train/src/model.rs:
// RMSNorm, interleaved RoPE, grouped-query causal attention, SwiGLU, tied head.

const WG = 64;

const SHADERS = {
  embed: /* wgsl */ `
struct P { n: u32, dim: u32 }
@group(0) @binding(0) var<storage, read> tok: array<u32>;
@group(0) @binding(1) var<storage, read> emb: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= p.n * p.dim) { return; }
  let t = i / p.dim; let c = i % p.dim;
  out[i] = emb[tok[t] * p.dim + c];
}`,
  rmsnorm: /* wgsl */ `
struct P { rows: u32, dim: u32, eps: f32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
var<workgroup> red: array<f32, ${WG}>;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let r = wg.x; let base = r * p.dim;
  var s = 0.0;
  for (var c = l.x; c < p.dim; c += ${WG}u) { let v = x[base + c]; s += v * v; }
  red[l.x] = s; workgroupBarrier();
  for (var k = ${WG / 2}u; k > 0u; k = k / 2u) { if (l.x < k) { red[l.x] += red[l.x + k]; } workgroupBarrier(); }
  let inv = 1.0 / sqrt(red[0] / f32(p.dim) + p.eps);
  for (var c = l.x; c < p.dim; c += ${WG}u) { out[base + c] = x[base + c] * inv * w[c]; }
}`,
  // C[coff + M×N] (+)= A[M×K] · B[N×K]ᵀ — torch.nn.Linear layout, 16×16 tiles.
  matmul_nt: /* wgsl */ `
struct P { M: u32, N: u32, K: u32, acc: u32, coff: u32 }
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
var<workgroup> As: array<array<f32, 16>, 16>;
var<workgroup> Bs: array<array<f32, 16>, 16>;
@compute @workgroup_size(16, 16) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let row = wg.y * 16u + l.y; let col = wg.x * 16u + l.x; let bn = wg.x * 16u + l.y;
  var acc = 0.0;
  for (var k0 = 0u; k0 < p.K; k0 += 16u) {
    let ka = k0 + l.x;
    As[l.y][l.x] = select(0.0, A[min(row * p.K + ka, arrayLength(&A) - 1u)], row < p.M && ka < p.K);
    Bs[l.x][l.y] = select(0.0, B[min(bn * p.K + ka, arrayLength(&B) - 1u)], bn < p.N && ka < p.K);
    workgroupBarrier();
    for (var k = 0u; k < 16u; k++) { acc += As[l.y][k] * Bs[k][l.x]; }
    workgroupBarrier();
  }
  if (row < p.M && col < p.N) { let i = p.coff + row * p.N + col; if (p.acc == 1u) { C[i] += acc; } else { C[i] = acc; } }
}`,
  rope: /* wgsl */ `
struct P { rows: u32, heads: u32, hd: u32, pos0: u32, srow0: u32 }
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> cs: array<f32>;
@group(0) @binding(2) var<storage, read> sn: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let half = p.hd / 2u; let i = g.x; if (i >= p.rows * p.heads * half) { return; }
  let t = i / (p.heads * half); let rem = i % (p.heads * half); let h = rem / half; let k = rem % half;
  let o = (p.srow0 + t) * p.heads * p.hd + h * p.hd + 2u * k;
  let pt = p.pos0 + t;
  let a = x[o]; let b = x[o + 1u]; let c = cs[pt * half + k]; let s = sn[pt * half + k];
  x[o] = a * c - b * s; x[o + 1u] = a * s + b * c;
}`,
  // One workgroup per (head, new position): causal scores against the KV
  // cache (positions 0..=p0+i), softmax, P·V.
  attention: /* wgsl */ `
struct P { T: u32, H: u32, KV: u32, hd: u32, alpha: f32, p0: u32 }
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> s: array<f32, 1024>;
var<workgroup> red: array<f32, ${WG}>;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let h = wg.x; let kvh = h / (p.H / p.KV);
  let qo = wg.y * p.H * p.hd + h * p.hd; let i = p.p0 + wg.y;
  var mx = -3.4e38;
  for (var j = l.x; j <= i; j += ${WG}u) {
    let ko = j * p.KV * p.hd + kvh * p.hd; var d = 0.0;
    for (var c = 0u; c < p.hd; c++) { d += q[qo + c] * k[ko + c]; }
    s[j] = d * p.alpha; mx = max(mx, s[j]);
  }
  red[l.x] = mx; workgroupBarrier();
  for (var r = ${WG / 2}u; r > 0u; r = r / 2u) { if (l.x < r) { red[l.x] = max(red[l.x], red[l.x + r]); } workgroupBarrier(); }
  let m = red[0]; workgroupBarrier();
  var sum = 0.0;
  for (var j = l.x; j <= i; j += ${WG}u) { s[j] = exp(s[j] - m); sum += s[j]; }
  red[l.x] = sum; workgroupBarrier();
  for (var r = ${WG / 2}u; r > 0u; r = r / 2u) { if (l.x < r) { red[l.x] += red[l.x + r]; } workgroupBarrier(); }
  let inv = 1.0 / red[0];
  for (var c = l.x; c < p.hd; c += ${WG}u) {
    var acc = 0.0;
    for (var j = 0u; j <= i; j++) { acc += s[j] * v[j * p.KV * p.hd + kvh * p.hd + c]; }
    o[qo + c] = acc * inv;
  }
}`,
  silu_mul: /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> g: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x; if (i >= p.n) { return; }
  let x = a[i]; g[i] = x / (1.0 + exp(-x)) * b[i];
}`,
};

/** Parse a safetensors file (F32 only) into a Map name → {shape, data}. */
export function parseSafetensors(buffer) {
  const view = new DataView(buffer);
  const hlen = Number(view.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, hlen)));
  const body = 8 + hlen;
  const out = new Map();
  for (const [name, t] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    if (t.dtype !== "F32") throw new Error(`${name}: only F32 tensors are supported`);
    const [b, e] = t.data_offsets;
    out.set(name, { shape: t.shape, data: new Float32Array(buffer.slice(body + b, body + e)) });
  }
  return out;
}

/** SCP byte-level BPE (tokenizer.json v2) with Python `re` split semantics. */
export class Tokenizer {
  constructor(json) {
    const WS = "\\t\\n\\v\\f\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
    const W = "\\p{L}\\p{N}_";
    this.pat = new RegExp(`'(?:[sdmt]|ll|ve|re)| ?[\\p{L}\\p{Nl}\\p{No}_]+| ?\\p{Nd}+| ?[^${WS}${W}]+|[${WS}]+(?![^${WS}])|[${WS}]+`, "gu");
    this.rank = new Map(json.merges.map(([a, b, i]) => [`${a},${b}`, i]));
    this.bytes = new Map([...Array(256).keys()].map((i) => [i, [i]]));
    for (const [a, b, i] of json.merges) this.bytes.set(i, [...this.bytes.get(a), ...this.bytes.get(b)]);
    const n = 256 + json.merges.length;
    [this.bos, this.eos, this.pad] = [n, n + 1, n + 2];
    this.vocabSize = n + 3;
  }
  encodeChunk(bytes) {
    let ids = [...bytes];
    for (;;) {
      let best = -1, br = Infinity;
      for (let i = 0; i + 1 < ids.length; i++) {
        const r = this.rank.get(`${ids[i]},${ids[i + 1]}`);
        if (r !== undefined && r < br) { br = r; best = i; }
      }
      if (best < 0) return ids;
      const [x, y] = [ids[best], ids[best + 1]];
      const out = [];
      for (let i = 0; i < ids.length;) {
        if (i + 1 < ids.length && ids[i] === x && ids[i + 1] === y) { out.push(br); i += 2; } else { out.push(ids[i]); i++; }
      }
      ids = out;
    }
  }
  encode(text) {
    const te = new TextEncoder();
    const out = [];
    for (const m of text.matchAll(this.pat)) out.push(...this.encodeChunk(te.encode(m[0])));
    return out;
  }
  decode(ids) {
    const bytes = ids.flatMap((i) => this.bytes.get(i) ?? []);
    return new TextDecoder("utf-8", { fatal: false }).decode(new Uint8Array(bytes));
  }
}

export class ForgeModel {
  /** @param {GPUDevice} device @param {object} cfg ModelConfig (forge JSON) @param {Map} weights */
  constructor(device, cfg, weights, ctx) {
    this.device = device;
    this.cfg = { ...cfg, head_dim: cfg.dim / cfg.n_heads, ffn: cfg.ffn_hidden ?? 64 * Math.ceil(Math.floor((8 * cfg.dim) / 3) / 64) };
    this.T = Math.min(ctx ?? cfg.max_seq_len, cfg.max_seq_len, 1024);
    this.pipelines = {};
    for (const [name, code] of Object.entries(SHADERS)) {
      this.pipelines[name] = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
    }
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.w = {};
    for (const [name, t] of weights) {
      const buf = device.createBuffer({ size: Math.max(16, t.data.byteLength), usage });
      device.queue.writeBuffer(buf, 0, t.data);
      this.w[name] = buf;
    }
    const { dim: d, n_heads: H, n_kv_heads: KV, head_dim: hd, ffn: f, vocab_size: V } = this.cfg;
    const T = this.T;
    const mk = (n) => device.createBuffer({ size: Math.max(16, 4 * n), usage });
    this.b = { tok: mk(T), x: mk(T * d), xn: mk(T * d), q: mk(T * H * hd), k: mk(T * KV * hd), v: mk(T * KV * hd),
      att: mk(T * H * hd), h1: mk(T * f), h3: mk(T * f), g: mk(T * f), last: mk(d), logits: mk(V) };
    // KV cache: rotated keys and values of every position fed so far, per layer.
    this.kc = Array.from({ length: cfg.n_layers }, () => mk(T * KV * hd));
    this.vc = Array.from({ length: cfg.n_layers }, () => mk(T * KV * hd));
    this.pos = 0;
    this.read = device.createBuffer({ size: 4 * V, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    // RoPE tables with the same f32 arithmetic as forge (scp_model.precompute_rope).
    const half = hd / 2, cos = new Float32Array(T * half), sin = new Float32Array(T * half);
    const theta = Math.fround(cfg.rope_theta ?? 10000);
    for (let i = 0; i < half; i++) {
      const inv = Math.fround(1 / Math.fround(Math.pow(theta, Math.fround((2 * i) / hd))));
      for (let t = 0; t < T; t++) {
        const fr = Math.fround(t * inv);
        cos[t * half + i] = Math.cos(fr);
        sin[t * half + i] = Math.sin(fr);
      }
    }
    this.cos = mk(T * half); device.queue.writeBuffer(this.cos, 0, cos);
    this.sin = mk(T * half); device.queue.writeBuffer(this.sin, 0, sin);
    this.uniforms = [];
  }

  uniform(values) {
    const data = new ArrayBuffer(Math.max(16, Math.ceil((4 * values.length) / 16) * 16));
    const dv = new DataView(data);
    values.forEach(([v, type], i) => (type === "f" ? dv.setFloat32(4 * i, v, true) : dv.setUint32(4 * i, v, true)));
    const buf = this.device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buf, 0, data);
    this.uniforms.push(buf);
    return buf;
  }

  dispatch(pass, name, buffers, uniformValues, groups) {
    const pipe = this.pipelines[name];
    const entries = [...buffers, this.uniform(uniformValues)].map((buffer, binding) => ({ binding, resource: { buffer } }));
    pass.setPipeline(pipe);
    pass.setBindGroup(0, this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries }));
    pass.dispatchWorkgroups(...groups);
  }

  /** Forget the cached positions. */
  reset() {
    this.pos = 0;
  }

  /** Logits after the last of `ids`, computed from scratch (the last T tokens are used). */
  async nextLogits(ids) {
    this.reset();
    return this.feed(ids.slice(-this.T));
  }

  /** Append `tokens` at the next positions (one block of GEMMs, keys and values
   *  go to the cache) and return the logits after the last one. */
  async feed(tokens) {
    const { dim: d, n_heads: H, n_kv_heads: KV, head_dim: hd, ffn: f, vocab_size: V, n_layers: L } = this.cfg;
    const n = tokens.length, p0 = this.pos, kvd = KV * hd;
    if (n === 0) throw new Error("feed needs at least one token");
    if (p0 + n > this.T) throw new Error(`context full: ${p0} + ${n} > ${this.T}`);
    const dev = this.device, b = this.b, w = this.w;
    dev.queue.writeBuffer(b.tok, 0, new Uint32Array(tokens));
    this.uniforms.forEach((u) => u.destroy());
    this.uniforms = [];
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    const ceil = (a, m) => Math.ceil(a / m);
    const mm = (A, B, C, M, N, K, acc, coff = 0) =>
      this.dispatch(pass, "matmul_nt", [A, B, C], [[M], [N], [K], [acc ? 1 : 0], [coff]], [ceil(N, 16), ceil(M, 16), 1]);
    this.dispatch(pass, "embed", [b.tok, w["tok_emb.weight"], b.x], [[n], [d]], [ceil(n * d, WG), 1, 1]);
    const eps = this.cfg.norm_eps ?? 1e-5;
    for (let l = 0; l < L; l++) {
      const p = (s) => w[`blocks.${l}.${s}`];
      this.dispatch(pass, "rmsnorm", [b.x, p("attn_norm.weight"), b.xn], [[n], [d], [eps, "f"]], [n, 1, 1]);
      mm(b.xn, p("attn.wq.weight"), b.q, n, H * hd, d, false);
      mm(b.xn, p("attn.wk.weight"), this.kc[l], n, kvd, d, false, p0 * kvd);
      mm(b.xn, p("attn.wv.weight"), this.vc[l], n, kvd, d, false, p0 * kvd);
      this.dispatch(pass, "rope", [b.q, this.cos, this.sin], [[n], [H], [hd], [p0], [0]], [ceil((n * H * hd) / 2, WG), 1, 1]);
      this.dispatch(pass, "rope", [this.kc[l], this.cos, this.sin], [[n], [KV], [hd], [p0], [p0]], [ceil((n * kvd) / 2, WG), 1, 1]);
      this.dispatch(pass, "attention", [b.q, this.kc[l], this.vc[l], b.att], [[n], [H], [KV], [hd], [1 / Math.sqrt(hd), "f"], [p0]], [H, n, 1]);
      mm(b.att, p("attn.wo.weight"), b.x, n, d, H * hd, true); // residual in the GEMM epilogue
      this.dispatch(pass, "rmsnorm", [b.x, p("ffn_norm.weight"), b.xn], [[n], [d], [eps, "f"]], [n, 1, 1]);
      mm(b.xn, p("ffn.w1.weight"), b.h1, n, f, d, false);
      mm(b.xn, p("ffn.w3.weight"), b.h3, n, f, d, false);
      this.dispatch(pass, "silu_mul", [b.h1, b.h3, b.g], [[n * f]], [ceil(n * f, WG), 1, 1]);
      mm(b.g, p("ffn.w2.weight"), b.x, n, d, f, true);
    }
    this.dispatch(pass, "rmsnorm", [b.x, w["norm.weight"], b.xn], [[n], [d], [eps, "f"]], [n, 1, 1]);
    pass.end();
    enc.copyBufferToBuffer(b.xn, 4 * (n - 1) * d, b.last, 0, 4 * d);
    const pass2 = enc.beginComputePass();
    this.dispatch(pass2, "matmul_nt", [b.last, w["tok_emb.weight"], b.logits], [[1], [V], [d], [0], [0]], [ceil(V, 16), 1, 1]);
    pass2.end();
    enc.copyBufferToBuffer(b.logits, 0, this.read, 0, 4 * V);
    dev.queue.submit([enc.finish()]);
    await this.read.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.read.getMappedRange().slice(0));
    this.read.unmap();
    this.pos += n;
    return out;
  }

  /** Greedy (temperature 0) or temperature/top-k sampling with the KV cache:
   *  the prompt is one block, every new token one position. The prompt is cut
   *  from the left so prompt + maxNew fits; if the context still fills up,
   *  the newest half is re-encoded. Returns new ids and tokens/s. */
  async generate(prompt, { maxNew = 64, temperature = 0, topK = 40, seed = 1, stop = null } = {}) {
    let s = seed >>> 0 || 1;
    const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
    const keep = Math.max(1, this.T - maxNew);
    const ids = prompt.length ? prompt.slice(-keep) : [0];
    const start = ids.length, t0 = performance.now();
    this.reset();
    let logits = await this.feed(ids);
    for (let i = 0; i < maxNew; i++) {
      let next = 0;
      if (temperature <= 0) {
        for (let j = 1; j < logits.length; j++) if (logits[j] > logits[next]) next = j;
      } else {
        const idx = [...logits.keys()].sort((a, b) => logits[b] - logits[a]).slice(0, Math.max(1, topK));
        const wts = idx.map((j) => Math.exp((logits[j] - logits[idx[0]]) / temperature));
        let r = rand() * wts.reduce((a, b) => a + b, 0);
        next = idx[idx.length - 1];
        for (let j = 0; j < idx.length; j++) { r -= wts[j]; if (r <= 0) { next = idx[j]; break; } }
      }
      ids.push(next);
      if (next === stop || i === maxNew - 1) break;
      if (this.pos === this.T) {
        this.reset();
        logits = await this.feed(ids.slice(-Math.floor(this.T / 2)));
      } else {
        logits = await this.feed([next]);
      }
    }
    const secs = (performance.now() - t0) / 1000;
    return { ids: ids.slice(start), tokensPerSecond: (ids.length - start) / secs };
  }
}

/** Open the best available GPU with limits large enough for the model. */
export async function openDevice(gpu = globalThis.navigator?.gpu) {
  if (!gpu) throw new Error("WebGPU is not available in this browser");
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter");
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  return { adapter, device };
}
