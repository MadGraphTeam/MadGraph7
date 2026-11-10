# Closed failed current-source lifecycle preparation

The parent coordinator ran the admitted preparation stages using source proposal1, index `8fc12d8160810016ed302d28fafe936830a3bd3c12cb00d1d5afdc049af2ebff`. Freeze, materialization, actual CLI generation and driver preparation passed. Build qualification failed after the first plain executable linked, because its actual link map included an input absent from the pre-link inventory. No loader query, card smoke, reducer evaluation or amplitude run followed that failure.

The five stage durations total 67.952843 seconds; the failed build stage took 23.931180 seconds. These are preparation durations, not numerical performance measurements.

The pinned Nix Fortran compiler wrapper's `nix-support/cc-ldflags` injects GCC search directories absent from the underlying compiler's `-print-search-dirs` and `-print-file-name` answers. The actual map selected `/nix/store/xc0ga87wdclrx54qjaryahkkmkmqi9qz-gcc-15.2.0-lib/lib/libgcc_s.so`. That GNU linker script selects its relative `libgcc_s.so.1` and a separate `-lgcc` archive. The original strict inventory check correctly rejected the unbound input. The successor must bind wrapper flags and transitive script inputs before linking; this packet grants no waiver and changes no previous source or result.

`payload.json.gz` contains a deduplicated UTF-8 ledger with all 144 closed report files, the exact sealed source proposal, generated text sources/cards, the actual failed link map and the implicated wrapper/script text. It binds 363 paths and 296 unique text contents. Generated binaries, archives, objects and modules are represented only by their hashes, sizes, modes and link metadata. Absolute recorded paths are provenance; the portable verifier never opens them. The complete private MadGraph source is reproducible from the recorded Git commit and materialized source inventory.

From the repository root:

```sh
source tools/tirust_env.sh
python -B docs/benchmarkfolder/experiments/subprocess-cache-lifecycle-current-failed-preparation-1/verify.py
```

The verifier also accepts a relocated copy's directory as its argument. `capture.py` records how this closed packet was produced and refuses to overwrite its payload. The original failed work and evidence directories remain unchanged.
