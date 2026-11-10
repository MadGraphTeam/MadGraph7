This package retains every regular file from closed run3, including both actual
native worker traces, stdout/stderr, maps and snapshots, fixtures, parameter logs,
loader receipts, alias guards, parsed plain result and failed overall result.
Large text traces use deterministic gzip; the raw length and hash remain bound.
All fifteen original library symlink targets are retained as metadata; no ELF,
archive or actual symlink is copied. The frozen source3 controller and relevant
generated Fortran are also retained byte-for-byte.

Both attempted native workers exited zero, but the observed worker's frozen
parser rejected its first diagnostic input-equality gate. One worker passed that
parser, two completed their native calls, and fourteen were never attempted.
Neither native exit zero nor portable verification makes the diagnostic pass.
All six returned diagnostic amplitudes retain unchecked status120/accuracy-1.
No cross-reducer adjudication or coefficient qualification was reached.

`python3 -B verify.py` opens only retained package members. It reconstructs the
original source3 parser, replays the plain result and the exact observed failure,
checks all captured original runtime identities, and counts actual public
entries/returns. Original filesystem paths are metadata, not replay inputs.
The static diagnosis is a separate additive packet; this package does not alter
the rejected gate, original source, generated Fortran, executable or failed run.
