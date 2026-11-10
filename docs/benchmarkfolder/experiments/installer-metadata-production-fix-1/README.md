# Strict metadata version types

The proposed production reader requires JSON integer types for ABI and link-order
versions. This rejects `true` and `1.0`, which compare equal to integer1 in Python,
as well as other incompatible values. Valid installed metadata is unchanged.
All27 installer tests and37 subtests passed against this exact proposed source.
The tests use mocked native builds and exercise actual metadata/availability
readers, preservation, and rollback paths. They do not qualify a new native build.

The source was temporarily restored to its frozen campaign bytes after the
provisional timing admission guard rejected the changed reader. No numerical
measurement had started. This exact patch will be promoted after the frozen
shared-host attempt ends; the failure and original source identity are preserved.
