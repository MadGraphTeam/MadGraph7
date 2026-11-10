"""Derive the private six-input replay; importing performs no native work."""
import hashlib

BASE_SHA256 = "d60574b25ec02283e643a6d059de5fe334674348e4e8b0819fe2ca948971e42b"
PRIVATE_GETTER = "ml_tirust_private_dp_diagnostics_v1"
CAPTURE = r'''    if (strcmp(stage, "evaluate") == 0) {
        unsigned char diagnostic[PRIVATE_DIAGNOSTIC_CAPACITY];
        size_t needed, copied;
        needed = ml_tirust_private_dp_diagnostics_v1(diagnostic, sizeof diagnostic);
        if (needed == 0 || needed > sizeof diagnostic) return 0;
        copied = needed - 1;
        if (diagnostic[copied] != 0) return 0;
        fprintf(out, ",\"private_diagnostic_required\":%zu,"
                "\"private_diagnostic_copy_truncated\":false,"
                "\"private_diagnostic_encoding\":\"latin1-bytes\","
                "\"private_diagnostic\":", needed);
        json_bytes(out, diagnostic, copied);
    }
'''


def prepare(base: bytes) -> bytes:
    if hashlib.sha256(base).hexdigest() != BASE_SHA256:
        raise ValueError("Unchanged first-failure replay source differs")
    text = base.decode("ascii")
    replacements = [
        ('enum { MAX_REQUESTS = 12, MAX_EVALUATIONS = 24, MAX_PATH_BYTES = 4096,',
         'enum { MAX_REQUESTS = 6, MAX_EVALUATIONS = 12, MAX_PATH_BYTES = 4096,'),
        ('       MAX_INPUT_BYTES = 32768, ERROR_CAPACITY = 4096, MAX_OUTPUT = 504 };',
         '       MAX_INPUT_BYTES = 32768, ERROR_CAPACITY = 4096, MAX_OUTPUT = 504 };\n'
         '\n/* Private bridge metadata only. This accessor never clears LAST_ERROR. */\n'
         'extern size_t ml_tirust_private_dp_diagnostics_v1(unsigned char *, size_t);\n'
         'enum { PRIVATE_DIAGNOSTIC_CAPACITY = 65536 };'),
        ('    /* Even a poisoned or unchanged output buffer is not read on failure. */',
         CAPTURE + '    /* Even a poisoned or unchanged output buffer is not read on failure. */'),
        ('    if (r->phase != 1 || r->method < 1 || r->method > 3 ||',
         '    if (r->case_id != 1 || r->phase != 1 || r->method < 1 || r->method > 3 ||'),
        ('    return 1;\nbad:\n    fclose(in);',
         '    if (*count != 6) return 0;\n'
         '    return 1;\nbad:\n    fclose(in);'),
        ('     * An empty manifest is retained as an empty journal; the controller must\n'
         '     * report all uncaptured targets explicitly rather than inventing inputs.\n',
         '     * Exactly the six captured Zgg requests are required; missing ZZgg\n'
         '     * inputs remain unavailable and are never reconstructed here.\n'),
    ]
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError("Ambiguous replay source replacement")
        text = text.replace(old, new)
    return text.encode("ascii")
