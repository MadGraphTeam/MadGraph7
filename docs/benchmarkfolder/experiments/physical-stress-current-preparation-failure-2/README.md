# Closed stress preparation2: absent output parent

Root session23842 exited1 after26.708035868seconds. The corrected tracked bin/madgraph dispatched the script correctly, imported loop_sm, selected dependency paths and generated g g > t t~ (3 Born,36+9 loop diagrams). The first export failed in the frozen baseline export_v4.py copy_template: it calls os.mkdir(self.dir_path), requiring the parent directory to exist. The private supervisor had omitted exports/baseline. MG catches the exception and exits0; the supervisor correctly rejected the absent output. --plain controls coloring only and is unrelated to dispatch.

All11 closed preparation files are retained byte-exact, with the two large materialized inventories gzip-compressed, plus MG5_debug, the model import debug log, and the exact exporter source. No native binary is copied. No loader/amplitude stage ran. Original supervisor2 and failed work/evidence remain unchanged.

Root authorized only an additive supervisor3 source correction: create output.parent without creating output, reject an existing output, retain fatal CLI markers as extra diagnostics, require generated files, and use fresh namespace3. No scientific input/policy change or automatic retry is made by this packet.
