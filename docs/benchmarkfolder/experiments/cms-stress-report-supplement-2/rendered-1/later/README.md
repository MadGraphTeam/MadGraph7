# Fourth chronology page: later matched-input QP result

This source page supplements the unchanged earlier three-page CMS diagnostic.
It uses the closed public matched-input packet
`docs/benchmarkfolder/experiments/matched-native-input-qualification-1`, index
`eacced6cc887ee586d15c4ceade8ac1f089fec0a36b84708f12086d05ba2f8a0`.
The public packet is also available at
[commit 2b4277c39e20](https://github.com/MadGraphTeam/MadGraph7/blob/2b4277c39e20e1f7011efad515e7af3a6c99f64a/docs/benchmarkfolder/experiments/matched-native-input-qualification-1/README.md).

`prepare_matched.py` authenticates all 304 public payloads, the four exact
closed receipt hashes and the scientific index/policy-note hashes. It joins
the 43 closed run payloads byte for byte. The report derivation checks all 52
paired native input/canonical-numerator/environment/metadata records, actual
direct USE, status and whole-worker gates, both bootstraps, exact returned
binary64/rational residuals and the prior/current candidate's 16+4 output-word
parity. The parent public verifier supplies the broader source/build/parser
replay; this report checker does not claim a new general source derivation.

The four table rows collapse two exact duplicate answer orders. The private
NinjaRS input-policy derivative is explicit; original legacy backend 8 is
unchanged and its historical failure remains a failure. Agreement is not an
analytic finite oracle, an algorithmic winner, normal checked-path or generic
qualification, production promotion, or precise performance acceptance.

Reproduce into a fresh output (or use `--verify-only` for the existing data):

```sh
source tools/tirust_env.sh
python3 -B .tirust-build/stress-matched-report-page-proposal-1/prepare_matched.py --public-packet docs/benchmarkfolder/experiments/matched-native-input-qualification-1 --public-index-sha256 eacced6cc887ee586d15c4ceade8ac1f089fec0a36b84708f12086d05ba2f8a0 --public-map .tirust-build/stress-matched-report-page-proposal-1/public-map.json --verify-only
```

`matched-data.json` is the final public-bound input. The separately named
`.tirust-build/stress-matched-report-preview-1.json` records the earlier explicit closed private preview
and is not used by Typst. The final mode requires an explicit public index/map;
it never discovers the latest attempt or follows paths from receipt text.

The builder-2 draft and its minimal delta in the separate sibling
`.tirust-build/stress-report-append-proposal-2/` remain unsealed working proposals
until the new four-page render exists. Their deliberately invalid pending pins
make them fail closed. The finalized 39-page builder will change only the
reviewed tail/page counts, new supplement path and exact source/render pins.
The original three-page and 38-page milestones remain unchanged.
