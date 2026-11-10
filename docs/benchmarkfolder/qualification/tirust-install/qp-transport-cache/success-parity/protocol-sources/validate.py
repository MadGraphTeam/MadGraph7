#!/usr/bin/env python3
"""Strict frame parsing and semantic checks, independent of native execution."""
from collections import Counter
from decimal import Decimal, localcontext
import math
import sys


def decode_real(words, precision):
    if precision not in (64, 128) or len(words)!=precision//64:
        raise ValueError("Invalid payload precision")
    bits = int.from_bytes(b"".join(w.to_bytes(8, sys.byteorder) for w in words), sys.byteorder)
    fraction_bits, exponent_bits, bias = (52, 11, 1023) if precision==64 else (112, 15, 16383)
    exponent = (bits >> fraction_bits) & ((1 << exponent_bits)-1)
    fraction = bits & ((1 << fraction_bits)-1)
    if exponent==(1 << exponent_bits)-1:
        raise ValueError("Nonfinite native payload")
    mantissa = fraction if exponent==0 else fraction+(1 << fraction_bits)
    power = (1-bias if exponent==0 else exponent-bias)-fraction_bits
    with localcontext() as ctx:
        ctx.prec=100
        result = Decimal(mantissa) * Decimal(2)**power
        return -result if bits >> (precision-1) else result


def parse(text, specification):
    frames = {}
    actions = specification["actions"]
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] not in ("F", "B", "A", "C"):
            raise ValueError("Unknown or empty native frame")
        identifier = int(fields[1])
        if not 1 <= identifier <= len(actions):
            raise ValueError("Unexpected action identifier")
        record = actions[identifier-1]
        fixture = specification["fixtures"][record["fixture"]]
        if fields[0]=="F":
            if identifier!=len(frames)+1 or len(fields)!=16:
                raise ValueError("Missing, repeated or malformed action header")
            expected = [record["id"],record["precision"],fixture["n"],fixture["rank"]]
            expected += [record[k] for k in ("mode","finite","check","cache","control","budget","lastbit")]
            if list(map(int,fields[1:12]))!=expected:
                raise ValueError("Native action differs from frozen protocol")
            frames[identifier] = dict(status=int(fields[12]), raw_generation=int(fields[13]),
                generation_before=int(fields[14]),generation_after=int(fields[15]),payload=[])
        elif identifier not in frames or identifier!=len(frames):
            raise ValueError("Native records are not contiguous")
        elif fields[0] in ("B", "A"):
            if fields[0] in frames[identifier] or len(fields)!=17:
                raise ValueError("Repeated or malformed cache snapshot")
            frames[identifier][fields[0]] = list(map(int,fields[2:]))
        else:
            payload = frames[identifier]["payload"]
            if list(map(int,fields[2:4])) != [len(payload)//4+1, len(payload)%4+1]:
                raise ValueError("Coefficient payload is missing or reordered")
            if len(fields)!=4+record["precision"]//32:
                raise ValueError("Malformed coefficient payload")
            if any(len(x)!=16 or any(c not in "0123456789ABCDEF" for c in x) for x in fields[4:]):
                raise ValueError("Malformed integer hexadecimal word")
            words = [int(x,16) for x in fields[4:]]
            split = len(words)//2
            payload.append((decode_real(words[:split],record["precision"]),
                            decode_real(words[split:],record["precision"])))
    if len(frames)!=len(actions):
        raise ValueError("Incomplete native output")
    for action in actions:
        frame = frames[action["id"]]
        rank = specification["fixtures"][action["fixture"]]["rank"]
        if set(frame)!={"status","raw_generation","generation_before","generation_after","payload","B","A"}:
            raise ValueError("Incomplete native frame")
        if len(frame["payload"])!=4*math.comb(rank+4,4):
            raise ValueError("Incomplete native tensor")
    return frames


def qualify(text, specification):
    frames = parse(text,specification)
    results = []
    for action in specification["actions"]:
        f = frames[action["id"]]
        failures = []
        status = f["status"]
        if status!=0 and not (action["allow_refusal"] and status==1):
            failures.append("unexpected native status")
        before, after = f["B"], f["A"]
        delta = [a-b for a,b in zip(after,before)]
        qd = delta[9:]
        if min(before+after)<0 or after[7]>action["budget"] or after[12]>action["budget"]//2:
            failures.append("invalid cache counter or byte budget")
        if action["control"] in (1,2,3) and any(before):
            failures.append("control did not reset cache counters")
        if action["control"]==1 and f["raw_generation"]!=-1 and f["generation_before"]!=f["raw_generation"]:
            failures.append("clear replaced the context")
        if action["control"]==2:
            if f["generation_before"]!=-1 or f["generation_after"]<=f["raw_generation"]:
                failures.append("finalize did not recreate context")
        elif f["generation_before"]!=-1 and f["generation_after"]!=f["generation_before"]:
            failures.append("evaluation unexpectedly replaced context")
        if f["generation_after"]<1:
            failures.append("evaluation has no live context")
        if status!=0:
            if any(x!=0 for row in f["payload"] for x in row):
                failures.append("refused call retained nonzero output")
        else:
            if action["precision"]==64 and any(qd):
                failures.append("DP call changed QP cache")
            if action["precision"]==128:
                dp_delta = delta[:9]
                for i,j in ((4,0),(5,1),(6,2),(7,3),(8,4)):
                    dp_delta[i]-=qd[j]
                if any(dp_delta):
                    failures.append("QP call changed DP cache")
            exp = action["expectation"]
            if exp=="bypass":
                expected = [0,0,0,0,0,1] if action["precision"]==128 else [0]*6
                if any(delta[:9]) or qd!=expected:
                    failures.append("bypass modified retained entries")
            elif exp=="dp_hit":
                if sum(delta[i] for i in (0,2,4))<=0 or delta[6]!=0:
                    failures.append("DP repeat did not reuse retained entries")
            elif exp=="dp_miss":
                if sum(delta[i] for i in (1,3,5))<=0 or delta[6]<=0:
                    failures.append("DP cold call did not populate cache")
            elif exp=="dp_refused":
                if delta[8]<=0 or after[6]!=0 or after[7]!=0:
                    failures.append("zero-budget DP insertion was not refused")
            elif exp=="qp_hit":
                if qd!=[1,0,0,0,0,0]:
                    failures.append("QP repeat did not hit exactly one tensor")
            elif exp=="qp_miss":
                if qd[:3]!=[0,1,1] or qd[3]<=0 or any(qd[4:]):
                    failures.append("QP cold call did not store exactly one tensor")
            elif exp=="qp_refused":
                if qd!=[0,1,0,0,1,0] or after[11]!=0 or after[12]!=0:
                    failures.append("zero-budget QP insertion was not refused")
            for index,row in enumerate(f["payload"]):
                slot = index%4
                masked = (slot==3 and not action["finite"]) or (
                    slot==0 and action["mode"] in (2,3)) or (slot in (1,2) and action["mode"] in (1,3))
                if masked and any(x!=0 for x in row):
                    failures.append("pole selection or finite mask is nonzero")
        maximum = Decimal(0)
        reference = frames.get(action["reference"])
        if status==0 and reference and reference["status"]==0:
            threshold = Decimal("1e-10" if action["precision"]==64 else "1e-24")
            for row,truth in zip(f["payload"],reference["payload"]):
                for value,target in zip(row,truth):
                    residual=abs(value-target)/max(Decimal(1),abs(value),abs(target))
                    maximum=max(maximum,residual)
            if maximum>threshold:
                failures.append("same-precision cache/control result differs from uncached reference")
        results.append(dict(id=action["id"],name=action["name"],native_status=status,
            result="failed" if failures else ("refused" if status else "passed"),
            failures=sorted(set(failures)),maximum_scaled_reference_residual=str(maximum)))

    # Compare full versus direct-pole slots and separated pole selections on
    # the infrared triangle, whose successful full and partial paths are known.
    selections = {(a["precision"],a["finite"],a["mode"]):frames[a["id"]] for a in specification["actions"]
                  if a["name"].startswith("reference-")}
    pole_checks = []
    for precision in (64,128):
        threshold=Decimal("1e-10" if precision==64 else "1e-24")
        for finite in (1,0):
            both=selections[precision,finite,0]
            for mode in (1,2,3):
                if finite==0 and mode==3:
                    continue
                other=selections[precision,finite,mode]
                maximum=Decimal(0)
                for index,(row,truth) in enumerate(zip(other["payload"],both["payload"])):
                    slot=index%4
                    if slot==3 or (slot==0 and mode==1) or (slot in (1,2) and mode==2):
                        for x,y in zip(row,truth):
                            maximum=max(maximum,abs(x-y)/max(Decimal(1),abs(x),abs(y)))
                pole_checks.append(dict(check="pole selection",precision=precision,finite=finite,mode=mode,
                    maximum_scaled_residual=str(maximum),passed=maximum<=threshold and both["status"]==other["status"]==0))
        for mode in range(3):
            full,partial=selections[precision,1,mode],selections[precision,0,mode]
            maximum=Decimal(0)
            for index,(row,truth) in enumerate(zip(full["payload"],partial["payload"])):
                if index%4!=3:
                    for x,y in zip(row,truth):
                        maximum=max(maximum,abs(x-y)/max(Decimal(1),abs(x),abs(y)))
            pole_checks.append(dict(check="full versus direct poles",precision=precision,mode=mode,
                maximum_scaled_residual=str(maximum),passed=maximum<=threshold and full["status"]==partial["status"]==0))
    return dict(passed=all(r["result"]!="failed" for r in results) and all(p["passed"] for p in pole_checks),
        calls=len(results),status_counts=dict(Counter(r["result"] for r in results)),
        results=results,pole_checks=pole_checks)
