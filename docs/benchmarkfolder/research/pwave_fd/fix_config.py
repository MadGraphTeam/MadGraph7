import json, shlex, sys
from pathlib import Path
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia')
import madgraph.various.tirust_install as t
d = Path(sys.argv[1])
c = json.load(open(d / 'tirust-config.json'))
c['artifact_sha256'] = t._artifact_hashes(d)
c['native_libs'] = shlex.split((d / 'tirust-native-libs.txt').read_text())
c['link_before'] = shlex.split((d / 'tirust-link-before.txt').read_text())
c['raised_powers'] = True
json.dump(c, open(d / 'tirust-config.json', 'w'), indent=1)
print(t.is_available(d), t.supports_raised_powers(d), t.supports_quad_precision(d))
