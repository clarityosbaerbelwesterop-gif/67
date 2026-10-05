// Parity: WebGPU logits vs the native forge engine on the same checkpoint.
// Usage: node test/parity.mjs <checkpoint dir> (needs target/release/forge).
// Headless Linux needs a Vulkan driver, e.g. lavapipe via VK_ICD_FILENAMES.
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { create, globals } from "webgpu";
import { ForgeModel, parseSafetensors } from "../forge-webgpu.mjs";

Object.assign(globalThis, globals);
const dir = process.argv[2];
const forge = process.env.FORGE ?? join(import.meta.dirname, "../../target/release/forge");
const state = JSON.parse(readFileSync(join(dir, "state.json"), "utf8"));
const cfg = state.config.model;
const bytes = readFileSync(join(dir, "model.safetensors"));
const weights = parseSafetensors(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
// Keep the GPU object referenced for the whole run (Dawn tears down on GC).
globalThis.forgeGpu = create([]);
const adapter = await globalThis.forgeGpu.requestAdapter();
if (!adapter) { console.error("no WebGPU adapter (set VK_ICD_FILENAMES to a Vulkan driver)"); process.exit(2); }
const device = await adapter.requestDevice();
const model = new ForgeModel(device, cfg, weights, cfg.max_seq_len);
let worst = 0;
for (const prompt of [[1], [3, 1, 4, 1, 5], Array.from({ length: 40 }, (_, i) => (i * 7) % cfg.vocab_size)]) {
  const ref = JSON.parse(execFileSync(forge, ["logits", "--ckpt", dir, "--prompt-ids", prompt.join(","), "--threads", "2"], { encoding: "utf8" }).trim().split("\n").pop()).logits;
  const got = await model.nextLogits(prompt);
  const diff = Math.max(...ref.map((r, i) => Math.abs(r - got[i])));
  const scale = Math.max(...ref.map(Math.abs));
  worst = Math.max(worst, diff / scale);
  console.log(`prompt len ${prompt.length}: max |Δlogit| ${diff.toExponential(2)} (scale ${scale.toFixed(2)})`);
}
const gpuGen = (await model.generate([2, 3, 4], { maxNew: 12 })).ids;
const cpuGen = JSON.parse(execFileSync(forge, ["generate", "--ckpt", dir, "--prompt-ids", "2,3,4", "--max-new", "12", "--threads", "2"], { encoding: "utf8" }).trim().split("\n").pop()).ids;
console.log("greedy GPU", gpuGen.join(","), "\ngreedy CPU", cpuGen.join(","));
const ok = worst < 1e-4 && gpuGen.join() === cpuGen.join();
console.log(ok ? "PARITY OK" : "PARITY FAILED");
process.exit(ok ? 0 : 1);
