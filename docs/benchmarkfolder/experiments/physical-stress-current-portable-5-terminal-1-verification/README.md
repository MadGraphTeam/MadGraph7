# Closed stress proof verification

`qualify.py` replays the complete continuation/ancestor/preparation closure, then copies all three capsules to a temporary sibling layout and replays with external file reads, writes and native calls blocked. Negative checks reject missing reader inventories, extra/missing declared files, changed retention/dependency/compressed bytes and a forged saved checked-stream count. Session86500 completed with exit0 and all10 checks passed.

Independent reviewer `/root/stress_unexecuted_continuation/continuation_review` separately replayed3937 physical files entirely from an in-memory reader while filesystem reads, `subprocess.Popen` and `ctypes.CDLL` were blocked. Session15924 completed with exit0. Its verified index, verifier and summary hashes match the captured packet. No native or active-main reads occurred.

The verified continuation has8 checked streams,11 physical rejections and1 controller/runtime-identity rejection. The disjoint aggregate has24 orchestrated requests,12 checked streams and22 eligible pair comparisons with12 failures, retaining all five earlier failures. These are evidence and consistency results, not independent physical truth or performance acceptance.

The QCDn4 TIRust request timed out before READY, with no completed warmup trace or final points; it did not produce a nonfinite or trap diagnostic. The EWn3 COLLIER request returned warmup statuses217,217,410 and W03 but lacked a complete runtime-map capture. Its numerical rejection and runtime-identity gap remain separate. Hardware-register flags were not sampled; runtime-reported IEEE text remains in the exact retained stdout and is exposed separately by the diagnostic companion.
