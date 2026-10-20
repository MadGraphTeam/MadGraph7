# Separate scalar-vertex extension

`oracle-dumps-ffs-v1` adds the two explicitly proved FFS maps at rank 2,
full and terminal, under the NLO compiler flags. All four streams pass the
unchanged independent exact oracle: 40 fixtures, 22,400 complex outputs and
143 strongly cancelled coefficients. The largest factored forward error is
0.665208 times `epsilon * absolute-product-scale`; the largest fraction of the
declared conservative bound is 0.047515.

The parent generator changes only its explicit proof selector:

- Initial FFV generator:
  `fec33d37a7f668355dd9073462117aa330cc81329ed7c4a069000c4b0962e570`.
- FFS-admitting generator:
  `bbf048ffe9361a041af616f95e617e8ee38de541756ad3478d9fa0e238054239`.

`ffs-selector-extension.json` records unchanged generated FFV source bytes.
The independent source diff confirms that no arithmetic-generating code changed.
The earlier FFV compiler campaigns retain their original generator snapshots.

The combined mathematical corpus contains 24 streams, 240 fixtures and
273,280 complex outputs. `validation-ffs-v1.json` binds the new campaign;
the read-only verifier discovers all four campaign aggregates. This scalar
extension does not establish rank 3/4 FFS behavior, standalone-flag FFS behavior,
new physical amplitudes or performance. The higher-rank and standalone controls
in the initial FFV campaign remain scoped to their actual vertex maps.
