# Splash and oMLX: techniques for MTPLX

This report records a scan of Splash and oMLX for techniques that could speed up MTPLX on Apple Silicon.
It covers an M1 with 64 GB of memory.
No code changed, and no benchmark ran.

## Sources

- **Splash binary:** `/opt/homebrew/bin/splash`, version `1.4.0-m1`. Read through `splash serve --help` only.
- **Splash source:** `https://github.com/paperniuk/splash`, pinned to tag `1.4.0-m1`. The tag matches the installed binary. The fork is an M1/M2 (Apple7/8) port.
- **oMLX source:** the local dev tree `~/git/omlx`. Splash is an external engine there. oMLX launches the binary and talks to it over HTTP.
- **Note:** the oMLX README credits `https://github.com/incoai/splash` for the same engine. That repo was not scanned, so the two may differ.

## Method

- Splash was indexed in the codebase-memory graph, which holds about 17,000 nodes and 98,000 edges.
- Each candidate was checked with a symbol search, then read in its header.
- Each MTPLX gap was checked with a regex search over `mtplx/*.py`.

## Findings

| Technique | Where it is | In MTPLX |
|---|---|---|
| bf16 recurrent (GDN) state | Splash `runtime/ops/GDN.hpp:38-48` | No. MTPLX keeps the state in fp32 (`mtplx/kernels/gdn_blocked_prefill.py:208`). |
| GPU and Neural Engine FFN split for prefill | Splash `runtime/ops/AneFfn.hpp:1-30`, `runtime/engine/AneFfnStartup.hpp:121-124` | No. MTPLX references ANE only in power telemetry (`mtplx/commands/public.py:4702`). |
| Expert streaming with adaptive slots | Splash `runtime/model/ExpertStore.hpp:1-30` | No. Helps only when MoE experts exceed RAM. |
| Register decode tiles on Apple7/8 | Splash `runtime/ops/Linear.hpp:76-85`, `runtime/ops/MoE.hpp:188` | Not compared. MTPLX has its own M1–M4 lane. |
| oMLX MoE expert offload | oMLX `omlx/patches/moe_expert_offload.py:1-30`, `omlx/custom_kernels/moe_staging/__init__.py:1-88` | No. Same idea as Splash expert streaming. |

Already present in MTPLX: DFlash (`mtplx/nax_verify.py`), bounded prefill, idle release, verify-row policies, a thinking budget, int8 KV, an SSD cache, and per-layer embedding memmap gathering.

## Assessment for a 64 GB M1

1. **bf16 GDN state.** This may cut decode bandwidth on the hybrid models. Splash rounds the state, so output is not bit-identical to fp32. MTPLX's bit-exact contracts make this a deliberate quality decision.
2. **ANE prefill split.** This is the most promising for prefill speed. It is also the most complex: it needs a W8A8 program and a calibrated split.
3. **Expert streaming.** This ranks last on 64 GB. The models in use probably fit in RAM.

## Recommendation

Benchmark items 1 and 2 on the M1 before any code change.
Use the same model and prompt on both builds.
Skip item 3 for now.

## Open questions

- Which Splash source is authoritative: `paperniuk/splash` or `incoai/splash`?
- Does the Apple7/8 register-tile work overlap with MTPLX's M1–M4 lane? That needs a head-to-head run.
