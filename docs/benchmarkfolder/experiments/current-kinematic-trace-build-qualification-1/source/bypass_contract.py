"""Narrow source admission for the current mode-zero full-sum driver path.

Inputs are already hash-verified ASCII snapshots plus the retained saved-source
statement reader. This is a literal contract for the current generated family,
not a general Fortran interpreter and not a native qualification.
"""


def require(value, message):
    if not value:
        raise ValueError(message)


def sequence(statements, wanted, message):
    matches = [i for i in range(len(statements) - len(wanted) + 1)
               if statements[i:i + len(wanted)] == list(wanted)]
    require(len(matches) == 1, message)
    return matches[0]


def validate(saved, matrix_source, commons_source, prefix="ML5_0_"):
    """Return reviewed semantic anchors; reject any changed narrow contract."""
    require(prefix == "ML5_0_", "Only the reviewed current process prefix is admitted")
    routines = saved.blocks(matrix_source)
    name = prefix + "SLOOPMATRIX_RETRY_IMPL"
    require(name in routines, "Current full-amplitude implementation absent")
    main = [s for _, s, _ in routines[name]["statements"]]
    bypass = "BYPASS_FILTERS=BYPASS_CHECK.OR.LOOPPOLEMODE.GE.0"
    require(main.count(bypass) == 1, "Explicit pole-mode bypass contract differs")
    sequence(main, ("IF(.NOT.BYPASS_FILTERS)THEN", "NTRY=NTRY+1", "ENDIF"),
             "Initialization count must not advance in the bypass path")
    opening = ("IF(BYPASS_FILTERS)THEN", "OLD_CHECKPHASE=CHECKPHASE",
               "OLD_HELDOUBLECHECKED=HELDOUBLECHECKED", "CHECKPHASE=.FALSE.",
               "HELDOUBLECHECKED=.TRUE.", "DOI=1,NCOMB",
               "OLD_GOODHEL(I)=GOODHEL(I)", "GOODHEL(I)=1", "ENDDO",
               "DOI=1,NSQUAREDSO", "DOJ=1,NLOOPGROUPS",
               "OLD_GOODAMP(I,J)=GOODAMP(I,J)", "GOODAMP(I,J)=.TRUE.",
               "ENDDO", "ENDDO", "ENDIF")
    closing = ("IF(BYPASS_FILTERS)THEN", "CHECKPHASE=OLD_CHECKPHASE",
               "HELDOUBLECHECKED=OLD_HELDOUBLECHECKED", "DOI=1,NCOMB",
               "GOODHEL(I)=OLD_GOODHEL(I)", "ENDDO", "DOI=1,NSQUAREDSO",
               "DOJ=1,NLOOPGROUPS", "GOODAMP(I,J)=OLD_GOODAMP(I,J)",
               "ENDDO", "ENDDO", "ENDIF")
    start = sequence(main, opening, "Temporary full-sum bypass differs")
    finish = sequence(main, closing, "Restored INIT/filter state differs")
    dispatch = main.index("IF(CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED))THEN")
    fixed_mode = sequence(main, ("HELPICKED=USERHEL", "IF(CTMODERUN.NE.-1)THEN",
                                "CTMODE=CTMODERUN", "ELSE", "CTMODE=1", "ENDIF"),
                          "Full-sum requested-mode dispatch differs")
    require(main.index(bypass) < start < dispatch < fixed_mode < finish,
            "Bypass/dispatch/restoration ordering differs")
    for statement in ("DATAUSERHEL/-1/", "COMMON/ML5_0_USERCHOICE/USERHEL",
                      "PARAMETER(SKIPLOOPEVAL=.FALSE.)"):
        require(statement in main, "Full-helicity/default evaluation contract differs: " + statement)
    require(not any(s.startswith("USERHEL=") for s in main),
            "Full-amplitude implementation changes the helicity request")
    sequence(main, ("ELSE", "RET_CODE_H=1", "DOI=0,NSQUAREDSO",
                    "ACCURACY(I)=-1.0D0", "ENDDO",
                    "RET_CODE_U=SET_RET_CODE_U(MLREDUCTIONLIB(I_LIB),.FALSE.,.FALSE.)",
                    "ENDIF"), "Fixed positive mode is no longer unchecked")
    final = sequence(main, ("IF(MP_DONE_ONCE)THEN", "RET_CODE_T=2", "ELSE",
                            "RET_CODE_T=1", "ENDIF"),
                     "Returned precision digit contract differs")
    require(fixed_mode < final < finish, "Return status must precede INIT restoration")

    thres = [s for _, s, _ in routines[prefix + "SLOOPMATRIX_THRES"]["statements"]]
    sequence(thres, ("USER_STAB_PREC=PREC_ASKED", "CALLML5_0_SLOOPMATRIX(P,ANS)",
                    "IF(ALWAYS_TEST_STABILITY.AND.(H.EQ.1.OR.ACCURACY(0).LT.0.0D0))THEN",
                    "BYPASS_CHECK=.TRUE.", "CALLML5_0_SLOOPMATRIX(P,ANS)",
                    "BYPASS_CHECK=.FALSE."),
             "Public threshold wrapper forced-second-call guard differs")
    force = [s for _, s, _ in routines[prefix + "FORCE_STABILITY_CHECK"]["statements"]]
    require("ALWAYS_TEST_STABILITY=ONOFF" in force,
            "Public force-stability setter no longer controls the guard")
    require("DATABYPASS_CHECK,ALWAYS_TEST_STABILITY/.FALSE.,.FALSE./" in force,
            "Stability-bypass flags lack reviewed DATA initialization")

    common_statements = [s for _, s, _ in saved.statements(commons_source)]
    function = "INTEGERFUNCTIONSET_RET_CODE_U(MLRED,DOING_QP,STABLE)"
    require(common_statements.count(function) == 1, "Return-code helper is not unique")
    begin = common_statements.index(function)
    end = common_statements.index("END", begin)
    helper = common_statements[begin:end]
    sequence(helper, ("IF(.NOT.STABLE)THEN", "SET_RET_CODE_U=0", "RETURN", "ENDIF"),
             "Unchecked fixed modes no longer return U=0")
    return {"schema": "current-pole-zero-bypass-contract-v1", "prefix": prefix,
            "pole_mode": 0, "force_stability_check": False,
            "full_helicity_userhel": -1, "temporary_all_helicities_and_groups": True,
            "postcall_init_flags": "restored; not effective-call readiness",
            "fixed_return_codes": {"1": 110, "4": 120},
            "unchecked_codes_are_not_success_certification": True,
            "source_only": True}
