# Dependency fixes required by the development benchmark setup

Each issue has a standalone reproducer, a suggested upstream patch and a short
explanation. The full local dependency checkouts are not included. API adaptation
to newer dependencies is not an upstream bug and is excluded from these reports.

## SymJIT 2.25

The affected baseline is
[`f1c193d301897149de6609f706297b0c97a4f018`](https://github.com/siravan/symjit-crate/commit/f1c193d301897149de6609f706297b0c97a4f018).

- [Output reuse](SYMJIT/output-reuse/README.md): count final output stores as
  uses, preserving values also consumed by other outputs. Already applied locally.
- [Unused trailing inputs](SYMJIT/unused-trailing-inputs/README.md): place
  direct-arena outputs after every declared input. Already applied locally.
- [Compressed argument counts](SYMJIT/argument-count-overflow/README.md):
  prevent repeated-expression compression from exceeding the serialized count.
  Discovered while refreshing the pure-gluon leading-colour benchmark.

The benchmark refresh also identified incorrect compressed normal-exit
handling, an ARM SIMD argument-load error, ARM callee-saved register
corruption, inconsistent real-input location coordinates and incorrect
complex return values from compressed helpers. The complete
issue list and validation status are maintained in
[SYMJIT/README.md](SYMJIT/README.md). The first two reports above retain the
earlier verified fixes; they are not newly discovered campaign failures.
