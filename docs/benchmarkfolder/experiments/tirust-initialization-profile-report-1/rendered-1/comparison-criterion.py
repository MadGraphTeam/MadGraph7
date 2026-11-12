def compare_rows(reference, candidate, tolerance=3e-6):
    """LoopMEComparator criterion, mapped to native ANS(0:3) column order.

ANS columns are Born, finite, simple pole and double pole. The comparator uses
abs(a-b)/abs(a+b), and permits poles small compared with the finite coefficient.
Opposite nonzero inputs have an undefined relative ratio and fail unless the
pole-size exception applies. Retain undefined ratios as null, never JSON inf.
    """
    if [(r["point"], r["order"]) for r in reference] != [(r["point"], r["order"]) for r in candidate]:
        raise RuntimeError("Numerical rows do not describe identical points/orders")
    failures, residuals, worst = [], [], 0.0
    target_passed = True
    for left, right in zip(reference, candidate):
        finite_scale = max(abs(left["coefficients"][1]), abs(right["coefficients"][1]), 1e-99)
        for column, (a, b) in enumerate(zip(left["coefficients"], right["coefficients"])):
            magnitude = max(abs(a), abs(b))
            symmetric = abs(a - b) / abs(a + b) if a + b else 0.0 if not magnitude else None
            raw_relative = abs(a - b) / magnitude if magnitude else 0.0
            pole_fraction = magnitude / finite_scale if column >= 2 else None
            def accepted(threshold):
                return ((symmetric is not None and symmetric < threshold)
                        or (pole_fraction is not None and pole_fraction < threshold))
            target_passed = target_passed and accepted(1e-8)
            residual = {"point": left["point"], "order": left["order"], "column": column,
                        "reference": a, "candidate": b, "relative_error": raw_relative,
                        "symmetric_relative_error": symmetric, "pole_to_finite": pole_fraction,
                        "passed": accepted(tolerance)}
            residuals.append(residual)
            worst = max(worst, raw_relative)
            if not residual["passed"]:
                failures.append(residual)
    return {"passed": not failures, "maximum_relative_error": worst,
            "criterion": "LoopMEComparator: abs(a-b)/abs(a+b)<tolerance, or pole magnitude/finite maximum<tolerance",
            "relative_error_metric": "abs(a-b)/max(abs(a),abs(b)); zero-zero is zero",
            "ordinary_target_tolerance": 1e-8, "ordinary_target_passed": target_passed,
            "tolerance": tolerance, "failures": failures, "residuals": residuals}
