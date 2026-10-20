# Clean tensor-factor export/build retention

All eight core and five control processes were regenerated through actual MG5
command cards and built cleanly from commit `f4bffc61ed67e5d083b80ff9901768d4d5779989`.
Each subprocess had zero preexisting objects. Physical cards, point streams,
model postimages and the complete numerical dependencies remain unchanged.
No loop amplitude or Symbolica worker was evaluated during this phase.

The new live runtime inventory finds one shared complete runtime group per
backend across all13 exports. An independent reuse check verifies every new
model postimage against the retained correction and every library/module/link
signature against its prior certified counterpart. This permits reuse of the
unchanged correction, legacy references and ABI-level controls; it does not
replace fresh candidate representative/variant controls or final timing.

`manifest.json` binds the completed generation records, command cards, complete
generated/compiled manifests, parameter cards, points, correction deltas and
build logs, together with the new runtime inventory and reuse result. Earlier
exports and evidence remain untouched. Durations describe concurrent generation
and builds on CPUs7,8 and have no performance acceptance role.

Run `verify.py` for retained-file checks, or `verify.py --live` to rehash the
disposable generated source, libraries, metadata and executable files too.
Neither mode compiles or evaluates amplitudes.
