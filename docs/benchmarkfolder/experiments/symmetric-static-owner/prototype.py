#!/usr/bin/env python3
"""Reuse the sealed caller proof, append separately generated static helpers."""
import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace

import static_owner

path = static_owner.PREVIOUS / "prototype.py"
spec = importlib.util.spec_from_file_location("sealed_sharing_prototype", path)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def transform(directory, variant="folded", trace=False, count_operations=False):
    assert variant in ("lazy", "owner-only", "folded")
    assert not count_operations or (trace and variant == "folded")
    old_shared = previous.shared
    operation_helpers = None
    if variant != "lazy":
        def generate(*args, **kwargs):
            source = static_owner.generate(*args, **kwargs,
                fold_guards=variant == "folded")
            if operation_helpers is not None:
                import count_products
                rank, vertex, terminal = args[1:4]
                index = operation_helpers[(rank, terminal, vertex)]
                source = count_products.instrument(source, index,
                    len(operation_helpers), rank, terminal,
                    static_owner.shared.gather.get_number_of_coefs_for_rank(rank))
            return source
        previous.shared = SimpleNamespace(
            groups=old_shared.groups, HERE=old_shared.HERE, generate=generate)
    try:
        files, proof = previous.transform(directory, trace=False)
        if count_operations:
            operation_helpers = {
                (h["rank"], h["terminal"], h["vertex"]): h["index"]
                for h in proof["helpers"]}
            files, proof = previous.transform(directory, trace=False)
    finally:
        previous.shared = old_shared
    proof.update(variant=variant, trace=trace,
        static_owner_sha256=previous.sha(
            (Path(__file__).parent / "static_owner.py").read_text()),
        previous_prototype_sha256=previous.sha(path.read_text()),
        scope="Sealed original caller/group/alias proof, original polynomial "
              "preserved as prefix; separate lazy/static-owner/folded-guard "
              "private QP variant. Every original addition remains.")
    if not trace:
        return files, proof
    count = len(proof["helpers"])
    source = files["polynomial.f"]
    original = (Path(directory) / "polynomial.f").read_text()
    assert source.startswith(original)
    appended = source[len(original):]
    for helper in proof["helpers"]:
        name = helper["replacement"]
        index = helper["index"]
        match = re.search(r"(?m)^      SUBROUTINE " + re.escape(name)
                          + r"\(.*?(?=^      SUBROUTINE |\Z)", appended,
                          flags=re.S | re.M)
        assert match
        body = match[0]
        assert body.count("      IMPLICIT NONE") == 1
        body = body.replace("      IMPLICIT NONE",
            "      IMPLICIT NONE\n"
            f"      INTEGER*8 SHARE_COUNTS(2,{count})\n"
            "      COMMON/SHARING_COUNTS/SHARE_COUNTS", 1)
        if variant == "lazy":
            marker = "      PRODUCT_READY=.FALSE."
            assert body.count(marker) == 1
            body = body.replace(marker,
                f"      SHARE_COUNTS(1,{index})=SHARE_COUNTS(1,{index})+1\n"
                + marker, 1)
        else:
            body, hits = re.subn(r"(?m)^C[^\n]*STATIC_OWNER_FAST_PATH[^\n]*$",
                lambda m: m[0] + "\n"
                f"      SHARE_COUNTS(1,{index})=SHARE_COUNTS(1,{index})+1",
                body)
            assert hits == 1
        body = re.sub(
            r"(?m)^( +)(CALL MP_\w*(?:UPDATE|TERMINAL)_WL_\d+_1(?:_DENSE)?\()",
            rf"\1SHARE_COUNTS(2,{index})=SHARE_COUNTS(2,{index})+1\n\1\2",
            body)
        appended = appended[:match.start()] + body + appended[match.end():]
    files["polynomial.f"] = original + appended
    driver = (Path(directory) / "check_sa.f").read_text()
    assert driver.count("      IMPLICIT NONE") == 1
    driver = driver.replace("      IMPLICIT NONE",
        f"      IMPLICIT NONE\n      INTEGER*8 SHARE_COUNTS(2,{count})\n"
        "      COMMON/SHARING_COUNTS/SHARE_COUNTS\n"
        f"      DATA SHARE_COUNTS/{2*count}*0/", 1)
    marker = "      OPEN(6,FILE='/dev/null',STATUS='OLD')"
    assert driver.count(marker) == 1
    driver = driver.replace(marker, marker + "\n      SHARE_COUNTS=0", 1)
    marker = "      ELAPSED=DBLE(T1-T0)/DBLE(RATE)"
    assert driver.count(marker) == 1
    driver = driver.replace(marker, marker +
        "\n      OPEN(74,FILE='sharing_trace.dat',STATUS='REPLACE')\n"
        f"      DO I=1,{count}\n"
        "         WRITE(74,'(3I20)') I,SHARE_COUNTS(:,I)\n"
        "      ENDDO\n      CLOSE(74)", 1)
    if count_operations:
        from count_products import FIELDS
        marker = "      IMPLICIT NONE"
        driver = driver.replace(marker, marker +
            f"\n      INTEGER*8 OP_COUNTS({len(FIELDS)},{count})\n"
            "      COMMON/SHARING_OPERATIONS/OP_COUNTS\n"
            f"      DATA OP_COUNTS/{len(FIELDS)*count}*0/", 1)
        driver = driver.replace("      SHARE_COUNTS=0",
                                "      SHARE_COUNTS=0\n      OP_COUNTS=0", 1)
        marker = "      ELAPSED=DBLE(T1-T0)/DBLE(RATE)"
        driver = driver.replace(marker, marker +
            "\n      OPEN(75,FILE='sharing_operations.dat',STATUS='REPLACE')\n"
            f"      DO I=1,{count}\n"
            f"         WRITE(75,'({len(FIELDS)+1}I20)') I,OP_COUNTS(:,I)\n"
            "      ENDDO\n      CLOSE(75)", 1)
        proof["operation_counter_fields"] = list(FIELDS)
        proof["operation_counters_are_diagnostic_only"] = True
    files["check_sa.f"] = driver
    return files, proof
