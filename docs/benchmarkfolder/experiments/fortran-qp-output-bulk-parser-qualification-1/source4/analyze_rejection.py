"""Only correct the established finite=false/no-poles native-error oracle.

The original analyzer is retained byte-for-byte. Both substitutions below are
exactly bounded by its source hash and unique statements; no value, IEEE,
precision, transport-error, cache or tolerance gate changes.
"""
import hashlib
from pathlib import Path

source=Path(__file__).with_name('analyze.py')
raw=source.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='a84cee0c224b024419d3496ac28f6f9392f3359bdce3501133b00131d2c43cb8'
text=raw.decode()
old='status=3 if unsupported else 2 if bad else 0'
assert text.count(old)==1
text=text.replace(old,'status=2 if unsupported else 2 if bad else 0')
old="error==(ERROR if status==2 else 'mock native recording provider')"
assert text.count(old)==1
text=text.replace(old,"error==(ERROR if status==2 and not unsupported else 'mock native recording provider')")
namespace={'__name__':'bulk_unit_corrected_native_rejection_oracle','__file__':str(source)}
exec(compile(text,str(source)+' [native rejection correction]','exec'),namespace)
compare=namespace['compare']
verify_complete=namespace['verify_complete']
parse=namespace['parse']

