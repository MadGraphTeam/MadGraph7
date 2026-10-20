"""Untimed structural counters; never part of a performance prototype."""
FIELDS = (
    "original_products_after_A_B_zero_skips",
    "unshared_multiplies",
    "owner_multiplies",
    "reused_products",
    "A_zero_skipped_product_slots",
    "B_zero_skipped_product_slots",
    "visited_B_entries",
    "eligible_products_with_pure_imaginary_B",
    "eligible_products_with_pure_real_B",
    "eligible_products_with_both_B_components_nonzero",
)


def instrument(source, index, count, rank, terminal, ncoef):
    assert source.count("  INTEGER I,J,K,L,M\n") == 1
    source = source.replace("  INTEGER I,J,K,L,M\n",
        f"  INTEGER I,J,K,L,M\n  INTEGER*8 OP_COUNTS({len(FIELDS)},{count})\n"
        "  COMMON/SHARING_OPERATIONS/OP_COUNTS\n", 1)

    def counter(field, amount="1"):
        return f"OP_COUNTS({field},{index})=OP_COUNTS({field},{index})+{amount}"

    old = "          UPDATER_COEF = B(J,M,K)"
    assert source.count(old) == 1
    source = source.replace(old, counter(7) + "\n" + old, 1)
    old = "          IF (UPDATER_COEF.EQ.CMPLX(0.0e0_16,0.0e0_16,KIND=16)) CYCLE"
    assert source.count(old) == 1
    count_zero = str(ncoef) + ("*MAX(0,MIN(J,LCUT_SIZE)-J+1)" if terminal else "*LCUT_SIZE")
    source = source.replace(old, old.removesuffix(" CYCLE") + " THEN\n"
        + counter(6, count_zero) + "\nCYCLE\nENDIF", 1)
    old = "                  IF(WL_AZERO(K,L,I)) CYCLE"
    assert source.count(old) == 3
    source = source.replace(old, "IF(WL_AZERO(K,L,I)) THEN\n"
        + counter(5) + "\nCYCLE\nENDIF")
    old = "                NEW_POSITION = COMB_COEF_POS(L,M)"
    assert source.count(old) == 3
    # Accepted operands are bounded finite. These additional equality tests
    # cannot raise new IEEE exceptions; all instrumentation is diagnostic only.
    phase = (counter(1) + "\n"
        "IF(REAL(UPDATER_COEF,KIND=16).EQ.0) THEN\n" + counter(8) + "\n"
        "ELSEIF(AIMAG(UPDATER_COEF).EQ.0) THEN\n" + counter(9) + "\n"
        "ELSE\n" + counter(10) + "\nENDIF\n")
    source = source.replace(old, phase + old)
    operations = {
        "                OUT(J,NEW_POSITION,I)=OUT(J,NEW_POSITION,I) + A(K,L,I)*UPDATER_COEF": 2,
        "                PRODUCT_CACHE(GROUP_ID,L,I)=A(K,L,I)*UPDATER_COEF": 3,
        "                OUT(J,NEW_POSITION,I)=OUT(J,NEW_POSITION,I) + PRODUCT_CACHE(-GROUP_ID,L,I)": 4,
    }
    for old, field in operations.items():
        assert source.count(old) == 1
        source = source.replace(old, counter(field) + "\n" + old, 1)
    return source
