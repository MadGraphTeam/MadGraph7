"""Load only the immutable, unmodified comparison function from its bound source."""
import ast
from collections import defaultdict
import hashlib
import itertools
import math
from pathlib import Path

SOURCE = Path(__file__).resolve().parent / 'bound-support/ninjars_report_v2.py'
text = SOURCE.read_text()
assert hashlib.sha256(text.encode()).hexdigest() == '93f942dfa74b2c3354d066e5436f4aadd7a009d0f08fc03a05190aa8f5c21c85'
tree = ast.parse(text)
node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'compare')
function_text = ast.get_source_segment(text,node)
assert hashlib.sha256(function_text.encode()).hexdigest() == '0f1e713538c509548eacc5c28ad4550a97f5c9647c2b8bf748fc4e41a5f1d430'
namespace = {'defaultdict':defaultdict,'itertools':itertools,'math':math}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),namespace)
compare = namespace['compare']
