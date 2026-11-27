#!/usr/bin/env bash
# build_ref.sh <V0 source dir> <build dir>: compile an MG5 2.5.5 onia V0
# reference with refdrv.F (sources copied, the reference left untouched)
set -e
SP=/tmp/claude-1134/-shared-external-madloop-onia/6136fd6c-8e41-47c7-a479-0995fb3044a9/scratchpad
export PATH=/nix/store/5km6kqfhjr16939ak9l1crzsh6x96f68-gfortran-wrapper-15.2.0/bin:$PATH
SRC=$1; DST=$2; OLO=$SP/swj/olo
rm -rf $DST; mkdir -p $DST
cp $SRC/*.f $SRC/*.f90 $SRC/*.inc $SRC/*.dat $DST/ 2>/dev/null || true
cd $DST
MK=makefile_check; [ -f $SRC/$MK ] || MK=makefile
OBJS=$(sed -n 's/^PROCESS *= *//p' $SRC/$MK | head -1)
DEFS=""
grep -q 'calcoption' constants.f90 && DEFS="$DEFS -DWITH_CALCOPTION"
grep -q 'MDL_MC' coupl.inc && DEFS="$DEFS -DHAS_MC"
grep -q 'MDL_MB' coupl.inc && DEFS="$DEFS -DHAS_MB"
grep -qi 'virt_wgts(3,2)' matrix.f 2>/dev/null && DEFS="$DEFS -DTWO_POL"
# a standalone package (no FKS layout) leaves grid_file to its own main.f;
# the driver runs in run/, one level below the grid
if grep -q 'grid_file' constants.f90 2>/dev/null && ! grep -qi 'grid_file *=' matrix.f; then
  g=$(ls *grid*.dat 2>/dev/null | head -1)
  [ -n "$g" ] && DEFS="$DEFS -DGRID_FILE='../$g'"
fi
FF="-O1 -w -ffixed-line-length-132 -I. -I$OLO"
# modules first, in dependency order
for m in constants use_func Func_PSI interpolation; do
  [ -f $m.f90 ] && gfortran $FF -c $m.f90
done
for o in $OBJS; do
  b=${o%.o}
  case $b in constants|use_func|Func_PSI|interpolation) continue;; esac
  gfortran $FF -c $b.f
done
gfortran $FF $DEFS -cpp -c $SP/swj/tools/refdrv.F -o refdrv.o
gfortran -o refdrv refdrv.o $OBJS $OLO/libavh_olo.a $OLO/libmpfun90.a
echo "built $DST ($OBJS)"
