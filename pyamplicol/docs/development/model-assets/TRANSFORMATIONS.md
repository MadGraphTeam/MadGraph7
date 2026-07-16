# Model Asset Transformations

The source hash is the SHA-256 of the selected blob at AmpliCol revision
`643bc6f99d7b2249af0a85204768df243e612411`. The package hash is the SHA-256
after the named deterministic transformation. Files absent from this table are
copied byte-for-byte and have equal source and package hashes in
`PROVENANCE.toml`.

| Package-relative path | Transformation | Source SHA-256 | Package SHA-256 |
|---|---|---|---|
| `json/scalar_gravity/restrict_default.json` | `loader-restriction-card-v1` | `78e536276cfa98a854ea9b418a6de9437da1e21ea258fcc57ba2b20a6a691900` | `78e536276cfa98a854ea9b418a6de9437da1e21ea258fcc57ba2b20a6a691900` |
| `json/scalar_gravity/scalar_gravity.json` | `loader-ufo-full-v1` | `e303fd1c39eb39e7deca5b36099202074a9048da3dbf2a73baf1f43a31648173` | `b0676be537ce7b2d7bd71ca99efd10dc2c09d92c0bd2b6089703c5ec24cffda1` |
| `json/scalars/restrict_default.json` | `loader-restriction-card-v1` | `31e30e8bdbdaf4753e361269fcd0294f7d7c82956c0afe66052dd527aa4b864a` | `ffe7759b8efcc86e6d12d3797c2e63a91c18c07de1bace6a51ee02144ecbf218` |
| `json/scalars/scalars.json` | `loader-ufo-full-v1` | `6085cb3b1307bfdf7f3bb709ac1df9354b0cef8b1a3f6316f673fa069c9d9c85` | `4656ad2ec58e8f460ae65a5487a8499ee2661df9a5c0b93a34527a2e186856f7` |
| `json/scalars/scalars_2p_3p.json` | `loader-json-canonical-v1` | `ba4a928f4bf28975ca861c85106559a6d2747326572751eec3448bf67a5f763a` | `0a268a9fc959036e166367155ab2eeca5c52f76927a89245f8964b9b33b2a8b5` |
| `json/sm/restrict_c_mass.json` | `loader-restriction-card-v1` | `6b191f6d0c45a6e91ea867079b1095e37fc4882df1c1380e70312890959dc7a6` | `014e93a779812d9af1d3886143cebc8fd911e92f5f970aee4dfffe01429651f5` |
| `json/sm/restrict_ckm.json` | `loader-restriction-card-v1` | `57f3ffbd0a0784d433bb9b01211299f29dde1ddd0182ee0b1e4a87dc3f2b5c73` | `be8bdca54507167d8ae4b33d0b0b77808052cd841d6fabd26a3d99926d915b99` |
| `json/sm/restrict_default.json` | `loader-restriction-card-v1` | `9d357d17c254595dbf612b66d6b531c982eefb71432835360bbfd8b21dcf454b` | `8fdd48a3624620c1a15bab602f43a41519038cf69d8fef6aeaaaf13a3f1846c1` |
| `json/sm/restrict_lepton_masses.json` | `loader-restriction-card-v1` | `a22a9de56e536cd7c4e4bd4707f5f95d3255fb0dffd49d9567ec9a659576bb76` | `3d9fe3095907cef03f889bd95b9630a2f836d70244f1b450cfd5d2764913531b` |
| `json/sm/restrict_no_b_mass.json` | `loader-restriction-card-v1` | `2ca10cc1e02b34f226db983da9a787659257003e6d84001204389e8aef03dfa5` | `82d2fbc06f9d0b7daaed56673912d920ca459f435e9083deed2f4b716c2f4a30` |
| `json/sm/restrict_no_masses.json` | `loader-restriction-card-v1` | `9bc6bba7b940da83484e5f43346ba202333669aaca578dfa110ed3fbc17514b8` | `f925921df1b66f010cd6f8cd348b5f4728ba37678e0733d8618cef3f2b4e610e` |
| `json/sm/restrict_no_tau_mass.json` | `loader-restriction-card-v1` | `7560d6890d7cd26532f28865fb8f4092ffcbf49c54fa7a16e921e6b08c3f43e3` | `a26603f4f3572309dc7ce2f23295d5a8aec61d2f3c42a017341245e459648478` |
| `json/sm/restrict_no_widths.json` | `loader-restriction-card-v1` | `9002d18be203c34418a3966f7cedc5d20844c317c55906efca32f51a38f1f183` | `fdc03507f82a3d98571b97ba0ab34f712dc0eb02b101f8308b96c6ab8490c08c` |
| `json/sm/restrict_zeromass_ckm.json` | `loader-restriction-card-v1` | `2173adc123ef202816d56e20772747f8a513ce9294635e099fdbe4832329944d` | `b8b4376c82f9486266283a383a0347ec42498296180d92249e2e7a772e1a7e1a` |
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
