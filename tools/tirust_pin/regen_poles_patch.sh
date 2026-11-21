#!/usr/bin/env bash
# regen_poles_patch.sh <tirust_git_dir> <rev> <old_patch> <patched_tree> <out_patch>
# Regenerate MadGraph's 0001-direct-poles-only.patch for TIRust <rev> from a tree where the
# old patch was applied (and hand-merged where hunks failed). Same file set as <old_patch>.
# Verifies that the new patch applies with zero fuzz/offset to a clean export and reproduces <patched_tree>.
set -euo pipefail
git_dir=$1; rev=$2; old=$3; B=$(realpath "$4"); out=$(realpath -m "$5")
W=$(mktemp -d); mkdir "$W/a" "$W/d"
git -C "$git_dir" archive "$rev" | tar -x -C "$W/a"
files=$(grep '^+++ b/' "$old" | sed 's|+++ b/||')
: > "$out"
cd "$W"
for f in $files; do
  if [ -e "a/$f" ]; then
    git diff --no-index --no-prefix "a/$f" "$B/$f" | sed -e '/^diff --git/d' -e '/^index /d' -e 's/^\(@@ [^@]* @@\).*/\1/' >> "$out" || true
  else
    git diff --no-index /dev/null "$B/$f" | sed -e '/^diff --git/d' -e '/^index /d' -e '/^new file mode/d' -e 's/^\(@@ [^@]* @@\).*/\1/' >> "$out" || true
  fi
done
sed -E -i "s#^\+\+\+ (b/)?${B#/}/#+++ b/#" "$out"
! grep -q "${B#/}" "$out" || { echo "absolute paths left in patch" >&2; exit 1; }
git -C "$git_dir" archive "$rev" | tar -x -C "$W/d"
( cd "$W/d" && patch -p1 --batch --forward --fuzz=0 -i "$out" ) | grep -iE "fail|offset|fuzz" && { echo "patch does not apply cleanly" >&2; exit 1; }
for f in $files; do cmp -s "$W/d/$f" "$B/$f" || { echo "DIFF $f" >&2; exit 1; }; done
echo "ok: $(grep -c '^+++ ' "$out") files, sha256 $(sha256sum "$out" | cut -c1-64)"
rm -rf "$W"
