#!/usr/bin/env python3
"""Isolated static product owners; no production generator modifications."""
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "symmetric-accumulation"
sys.path.insert(0, str(PREVIOUS))
import shared

LEGACY_GENERATE = shared.generate


def owner_map(groups, terminal):
    """Positive id writes, negative id reads; J/M/K is the actual traversal."""
    result = {}
    for index, group in enumerate(groups, 1):
        entries = sorted(tuple(x) for x in group)
        assert len(entries) >= 2 and len(set(entries)) == len(entries)
        assert len({x[2] for x in entries}) == 1
        if terminal:
            assert len({(x[0], x[2]) for x in entries}) == 1
        for entry in entries:
            j, m, k = entry
            assert 1 <= j <= 4 and 0 <= m <= 4 and 1 <= k <= 4
            assert entry not in result
            result[entry] = index if entry == entries[0] else -index
    return result


def generate(generator, rank, vertex="MP_FFV1L2_1", terminal=False,
             suffix="_STATIC", *, case="qcd-3", inventory_path=None,
             fold_guards=True):
    """Preserve every scatter addition and the earliest original multiply."""
    groups = shared.groups(vertex, terminal, case=case,
                           inventory_path=inventory_path)
    owners = owner_map(groups, terminal)
    source = LEGACY_GENERATE(generator, rank, vertex, terminal, suffix,
                             case=case, inventory_path=inventory_path)
    ncoef = shared.gather.get_number_of_coefs_for_rank(rank)
    name = generator.sub_prefix + ("TERMINAL" if terminal else "UPDATE")
    name += f"_WL_{rank}_1"
    extra = ",WL_ENTRY_INEXACT" if terminal else ""
    mapping = ",".join(str(owners.get((j, m, k), 0))
                       for k in range(1, 5) for m in range(5) for j in range(1, 5))
    source, count = re.subn(
        r"(?m)^  INTEGER, PARAMETER :: GROUP_MAP\(.*$",
        f"  INTEGER, PARAMETER :: GROUP_MAP(4,0:4,4)=RESHAPE((/{mapping}/),(/4,5,4/))",
        source)
    assert count == 1
    source, count = re.subn(r"(?m)^  LOGICAL PRODUCT_READY.*\n", "", source)
    assert count == 1
    assert source.count("PRODUCT_READY=.FALSE.") == 1
    source = source.replace("PRODUCT_READY=.FALSE.", "C STATIC_OWNER_FAST_PATH")
    # Discard unused predecessor/accumulator declarations inherited from gather.
    source, count = re.subn(
        r"(?m)^  (?:INTEGER N|INTEGER, PARAMETER :: PREDECESSOR.*|"
        r"LOGICAL ACTIVE.*|complex\*32 ACC)\n", "", source)
    assert count == 4
    source = source.replace("C Experimental coefficient-major gather",
                            "C Experimental static-owner scatter products", 1)

    # A platform without any requested halting query must use the public entry.
    source = source.replace("  LOGICAL HALT_FLAGS(SIZE(IEEE_ALL))",
        "  LOGICAL HALT_FLAGS(SIZE(IEEE_ALL))\n  INTEGER HALT_INDEX", 1)
    support = f"""DO HALT_INDEX=1,SIZE(IEEE_ALL)
  IF(.NOT.IEEE_SUPPORT_HALTING(IEEE_ALL(HALT_INDEX))) THEN
    CALL {name}(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE,OUT{extra})
    RETURN
  ENDIF
ENDDO
"""
    source = source.replace("CALL IEEE_GET_HALTING_MODE(IEEE_ALL,HALT_FLAGS)",
                            support + "CALL IEEE_GET_HALTING_MODE(IEEE_ALL,HALT_FLAGS)", 1)

    # The initial integer GATHER_BOUNDED scan covers exactly these unchanged A/B
    # slices. It implies finite values and the terminal WL_BOUNDED predicate.
    # Keep all dimension, rounding, INEXACT and trap guards and their fallbacks.
    if fold_guards:
        old = next(line for line in source.splitlines()
                   if line.startswith("IF(.NOT.WL_NEAREST.OR."))
        remove_at = old.index(".OR..NOT.ALL(")
        assert "WL_BOUNDED(" in old if terminal else "WL_IS_FINITE(" in old
        source = source.replace(old, old[:remove_at] + ") THEN", 1)
        guard = ("IF(WL_IS_FINITE(REAL(A(WL_K,WL_L,WL_I),KIND=16)).AND."
                 "WL_IS_FINITE(AIMAG(A(WL_K,WL_L,WL_I)))) ")
        assert source.count(guard) == 1
        source = source.replace(guard, "", 1)

    start = source.index("OUT(:,:,:)=", source.index("C STATIC_OWNER_FAST_PATH"))
    end = source.index("\nCONTAINS", start)
    iloop = "DO I=J,MIN(J,LCUT_SIZE)" if terminal else "DO I=1,LCUT_SIZE"
    # Dispatch once per B operand rather than once per coefficient. The owner
    # executes the original multiply at its original position. All users retain
    # the original additions and coefficient traversal.
    def inner(operation):
        return f"""            {iloop}
              DO L=0,{ncoef-1}
                IF(WL_SKIP_ZERO) THEN
                  IF(WL_AZERO(K,L,I)) CYCLE
                ENDIF
                NEW_POSITION = COMB_COEF_POS(L,M)
{operation}
              ENDDO
            ENDDO"""

    ordinary = inner(
        "                OUT(J,NEW_POSITION,I)=OUT(J,NEW_POSITION,I) + A(K,L,I)*UPDATER_COEF")
    owner = inner(
        "                PRODUCT_CACHE(GROUP_ID,L,I)=A(K,L,I)*UPDATER_COEF\n"
        "                OUT(J,NEW_POSITION,I)=OUT(J,NEW_POSITION,I) + PRODUCT_CACHE(GROUP_ID,L,I)")
    consumer = inner(
        "                OUT(J,NEW_POSITION,I)=OUT(J,NEW_POSITION,I) + PRODUCT_CACHE(-GROUP_ID,L,I)")
    skip = ("          WL_SKIP_ZERO=.TRUE." if fold_guards else
            "          WL_SKIP_ZERO=.FALSE.\n"
            "          IF(WL_NEAREST) WL_SKIP_ZERO=WL_IS_FINITE(REAL(UPDATER_COEF,KIND=16)).AND.WL_IS_FINITE(AIMAG(UPDATER_COEF))")
    body = f"""OUT(:,:,:)={generator.czero}
DO J=1,OUT_SIZE
  DO M=0,4
    DO K=1,IN_SIZE
          UPDATER_COEF = B(J,M,K)
          GROUP_ID=GROUP_MAP(J,M,K)
          IF (UPDATER_COEF.EQ.{generator.czero}) CYCLE
{skip}
          IF(GROUP_ID.EQ.0) THEN
{ordinary}
          ELSEIF(GROUP_ID.GT.0) THEN
{owner}
          ELSE
{consumer}
          ENDIF
    ENDDO
  ENDDO
ENDDO
"""
    return source[:start] + body + source[end:]
