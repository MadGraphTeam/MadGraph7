# Strict metadata types promoted

The production reader now requires integer ABI and link-order schema versions.
JSON booleans and floating versions such as true/1.0 are rejected. Valid integer
metadata keeps its existing behavior. Actual production tests passed:27 tests
and37 subtests, covering malformed metadata, preservation/rollback and exports
with mocked builds. The existing qualified installed prefix also remains
available with QP support under the new reader. No native build or numerical
evaluation was performed by these checks.

The earlier tested proposal and temporary source restoration remain preserved.
Promotion occurred only after the frozen provisional timing session82678 ended;
those timing samples belong to the earlier source inventory.
