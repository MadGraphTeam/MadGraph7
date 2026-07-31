import os.path
import sys
import re

new_lines = []
cpptype = None

if len(sys.argv) == 1:
    exit("Use as - python3 HelAmpsDenomExpand double/double_st/DW")

if sys.argv[1] == "DW":
    cpptype = "MG_ARITHM::Double<fptype>"
elif sys.argv[1] == "double_st":
    cpptype = "double_st"
elif sys.argv[1] == "double":
    cpptype = "double"
else:
    exit("Use as - python3 HelAmpsDenomExpand double/double_st/DW")

denom_string = """
    const {cpptype} P{m}d[4] = {{ static_cast<{cpptype}>(-cxreal( {wave}{m}[0] )), static_cast<{cpptype}>(-cxreal( {wave}{m}[1] )), static_cast<{cpptype}>(-cximag( {wave}{m}[1] )), static_cast<{cpptype}>(-cximag( {wave}{m}[0] )) }};
    const {cpptype} Md{m} = static_cast<{cpptype}>(M{m});
    const fptype_sv PmM2 = static_cast<fptype_sv>(( P{m}d[0] * P{m}d[0] ) - ( P{m}d[1] * P{m}d[1] ) - ( P{m}d[2] * P{m}d[2] ) - ( P{m}d[3] * P{m}d[3] ) ) - M{m} * M{m};
    const fptype_sv iMW = M{m} * W{m};
    const cxtype_sv denden = cxmake( PmM2, iMW );
    const cxtype_sv denom = {coef}/ denden;
"""

if os.path.isfile("../HelAmps_sm_backup"):
    file = "../HelAmps_sm_backup"
else:
    file = "../HelAmps_sm.h"
    with open("../HelAmps_sm.h", "r") as f:
        with open("../HelAmps_sm_backup", "w") as fb:
            fb.write(f.read())

with open(file, "r") as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    if cpptype == "MG_ARITHM::Double<fptype>":
        if '#include "mgOnGpuConfig.h"' in line:
            new_lines.append(line)
            new_lines.append('    #include "Arithmetics/Double.h"\n')
            continue

    if "cxtype_sv denom" in line and "//" not in line:
        if "P" not in line:
            exit(f"No P found in denom line: {line.strip()}")

        idxx = idx - 1
        wave = None

        while idxx >= 0:
            pline = lines[idxx]
            if pline.strip() == " {":
                break
            if re.search(r'P\d\[4', pline):
                m_wave = re.search(r'-cxreal\(\s*([A-Z]\d)\[', pline)
                if m_wave:
                    wave = m_wave.group(1)[0]
                break
            idxx -= 1

        if wave is None:
            exit(f"Could not determine wave for line: {line.strip()}")

        mR = re.search(r'P(\d+)\[0\]', line)
        if not mR:
            exit(f"Could not find variable number in line: {line.strip()}")
        m = mR.group(1)

        coefR = re.search(r'=\s*(.*?)\s*/', line)
        if not coefR:
            exit(f"Could not find numerator in line: {line.strip()}")
        coef = coefR.group(1)

        new_lines.append("//" + line)
        new_lines.append(denom_string.format(m=m, wave=wave, coef=coef, cpptype=cpptype))
    else:
        new_lines.append(line)

with open("../HelAmps_sm.h", "w") as f:
    f.writelines(new_lines)
