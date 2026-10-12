// The WebGPU node computes exactly what the coordinator checks (headcenter/backend/vgpu.py).
// Headless Linux: VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json node test/vgpu.mjs
import { execFileSync } from "node:child_process";
import { join } from "node:path";
import { create, globals } from "webgpu";
import { maxN, solve } from "../vgpu-node.mjs";

Object.assign(globalThis, globals);
globalThis.forgeGpu = create([]);
const adapter = await globalThis.forgeGpu.requestAdapter();
if (!adapter) { console.error("no WebGPU adapter (set VK_ICD_FILENAMES to a Vulkan driver)"); process.exit(2); }
const device = await adapter.requestDevice();
const root = join(import.meta.dirname, "../..");
let failed = 0;
for (const [seed, n, k] of [[1234567, 256, 3], [4294967295, 256, 2], [99, 512, 2]]) {
  const t0 = performance.now();
  const rows = await solve(device, seed, n, k);
  const ms = performance.now() - t0;
  const want = JSON.parse(execFileSync("python3", ["-c",
    `import sys,json; sys.path.insert(0, "headcenter"); from backend.vgpu import solve; print(json.dumps(solve(${seed}, ${n}, ${k})))`],
    { cwd: root, encoding: "utf8" }));
  const same = rows.length === want.length && rows.every((r, i) => r === want[i]);
  failed += !same;
  console.log(`seed ${seed} n ${n} k ${k}: ${same ? "identical" : "MISMATCH"} (${ms.toFixed(0)} ms, ${(2 * n ** 3 * k / ms / 1e9).toFixed(3)} TFLOP/s)`);
}
console.log(`max n on this device: ${maxN(device)}`);
process.exit(failed ? 1 : 0);
