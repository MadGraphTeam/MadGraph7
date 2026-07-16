# Model Asset Transformations

The source hash is the SHA-256 of the selected blob at AmpliCol revision
`643bc6f99d7b2249af0a85204768df243e612411`. The package hash is the SHA-256
after the named deterministic transformation. Files absent from this table are
copied byte-for-byte and have equal source and package hashes in
`PROVENANCE.toml`.

| Package-relative path | Transformation | Source SHA-256 | Package SHA-256 |
|---|---|---|---|
| `json/scalar_gravity/restrict_default.json` | `loader-restriction-card-v1` | `78e536276cfa98a854ea9b418a6de9437da1e21ea258fcc57ba2b20a6a691900` | `3f6727778bce0d2f5b5c9f4f8efba221b1b4753a0c87ffbed2d2af737cfe12a7` |
| `json/scalar_gravity/scalar_gravity.json` | `loader-ufo-full-v1` | `e303fd1c39eb39e7deca5b36099202074a9048da3dbf2a73baf1f43a31648173` | `b0676be537ce7b2d7bd71ca99efd10dc2c09d92c0bd2b6089703c5ec24cffda1` |
| `json/scalars/restrict_default.json` | `loader-restriction-card-v1` | `31e30e8bdbdaf4753e361269fcd0294f7d7c82956c0afe66052dd527aa4b864a` | `c6e4e1a68af965f0def35957507f0a5f51c62e8388058128cd667ca75632d73b` |
| `json/scalars/scalars.json` | `loader-ufo-full-v1` | `6085cb3b1307bfdf7f3bb709ac1df9354b0cef8b1a3f6316f673fa069c9d9c85` | `4656ad2ec58e8f460ae65a5487a8499ee2661df9a5c0b93a34527a2e186856f7` |
| `json/scalars/scalars_2p_3p.json` | `loader-json-canonical-v1` | `ba4a928f4bf28975ca861c85106559a6d2747326572751eec3448bf67a5f763a` | `0a268a9fc959036e166367155ab2eeca5c52f76927a89245f8964b9b33b2a8b5` |
| `json/sm/restrict_c_mass.json` | `loader-restriction-card-v1` | `6b191f6d0c45a6e91ea867079b1095e37fc4882df1c1380e70312890959dc7a6` | `b737d049dbf0d87f78d63f59f2cf4d00a4df0491fe3bc8383e00245bd89010be` |
| `json/sm/restrict_ckm.json` | `loader-restriction-card-v1` | `57f3ffbd0a0784d433bb9b01211299f29dde1ddd0182ee0b1e4a87dc3f2b5c73` | `a37646ed851f5a19b776551e14277d500e72b5847e7246b03d0cccfca46727f1` |
| `json/sm/restrict_default.json` | `loader-restriction-card-v1` | `9d357d17c254595dbf612b66d6b531c982eefb71432835360bbfd8b21dcf454b` | `db7492a4f219c6b54126a0f61d1cd4e1f06ea68d127808e2e3ddfdc721440433` |
| `json/sm/restrict_lepton_masses.json` | `loader-restriction-card-v1` | `a22a9de56e536cd7c4e4bd4707f5f95d3255fb0dffd49d9567ec9a659576bb76` | `7cd99a9f36acefbc52d2cb2891e487ab3900a64cdf1ac9036eee32ba4b0853af` |
| `json/sm/restrict_no_b_mass.json` | `loader-restriction-card-v1` | `2ca10cc1e02b34f226db983da9a787659257003e6d84001204389e8aef03dfa5` | `f951a911a8865eab48bb8dc29913e6e217e6cb0dd1a5f39af9371eaf2555025b` |
| `json/sm/restrict_no_masses.json` | `loader-restriction-card-v1` | `9bc6bba7b940da83484e5f43346ba202333669aaca578dfa110ed3fbc17514b8` | `98787774b557d07a53efeffb0f29f8bcb9ab7e16c95299f3a0b82057ca209fd8` |
| `json/sm/restrict_no_tau_mass.json` | `loader-restriction-card-v1` | `7560d6890d7cd26532f28865fb8f4092ffcbf49c54fa7a16e921e6b08c3f43e3` | `01c0f44c3c20f67cf3498595ebe5ec82a5813980abab4212e2ce9b2c2067b11d` |
| `json/sm/restrict_no_widths.json` | `loader-restriction-card-v1` | `9002d18be203c34418a3966f7cedc5d20844c317c55906efca32f51a38f1f183` | `79809b61aaf16b63ee6f78ffb240762cd514a046beef948e7ee82fbf9080d4a8` |
| `json/sm/restrict_zeromass_ckm.json` | `loader-restriction-card-v1` | `2173adc123ef202816d56e20772747f8a513ce9294635e099fdbe4832329944d` | `c81fcdae5e6556ac97f66d1073ab0e94132334982b8c0d0eca8f03ca6a118e63` |
| `json/sm/sm.json` | `loader-ufo-full-v1` | `684f75c88c7faf6832e9e8122970ec5fe98b9cd30bc989b7f8e1818c327fe165` | `f13311b6c66af04dde17484ce5b3b57f4e57a55f8a3729f00352105eea5a80bf` |
| `json/sm/sm_wrapped_indices.json` | `loader-ufo-full-wrapped-v1` | `b4221a0985ad5c6987c325a75012a3231995c604e44ad543d0b1ead22eb19404` | `d246c1ec3bcd2177404134571adf2f12e8faadb066bc4994f6eef8848fa1b182` |
| `ufo/scalar_gravity/parameters.py` | `scalar-gravity-static-defaults-v1` | `68de6a8ab6ecb1f358a13954be99597bb30665d1e195b7395f8faf405bd8657e` | `01b0477e0b0e352f968a6d1bfd1e9724c505adc6dd5c1d39dd08e9da97cd9f00` |
| `ufo/scalars/parameters.py` | `scalar-static-defaults-v1` | `7620698c89ec9a87eb11177f682627e9a334222d42b378c18e111b866f7c3ad0` | `7b11a9b1d84126d31f58d5c4752ea0fe0921074bfcb338ecd6ef4d2ad7da89a7` |

## Serialized Model Shape

The regenerated unrestricted JSON models have these stable counts:

| File | Particles | Vertices | Couplings | Lorentz structures |
|---|---:|---:|---:|---:|
| `sm.json` | 43 | 153 | 108 | 22 |
| `sm_wrapped_indices.json` | 43 | 153 | 108 | 22 |
| `scalars.json` | 3 | 276 | 1 | 8 |
| `scalars_2p_3p.json` | 3 | 16 | 1 | 2 |
| `scalar_gravity.json` | 4 | 11 | 14 | 8 |

The historical wrapped SM JSON contained default-restricted counts while its
serialized restriction field was unset. Regeneration makes it a complete,
unrestricted wrapped-index model, consistent with its package name and the
complete-model release requirement.
