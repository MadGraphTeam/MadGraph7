#!/usr/bin/env bash
# usage: fetch_pdfsets.sh <LHAPDF data dir> <set> [<set> ...]
# Make sure each LHAPDF set is complete in the data dir, downloading the
# missing ones from lhapdfsets.web.cern.ch. A set counts as present only when
# its <set>/<set>.info is there: a job cancelled while unpacking (every push
# cancels the running CI of its branch) left a set directory without it, which
# a directory test accepted for good, and LHAPDF then failed with "Info file
# not found for PDF set". Each set is unpacked aside and moved into place in
# one step, so an interrupted fetch leaves nothing that looks installed.
# Non-fatal: a set that cannot be fetched is reported as a warning.
set -u
DATADIR="$1"; shift
[ -d "$DATADIR" ] || exit 0
rm -rf "$DATADIR"/.fetch.*
for SET in "$@"; do
    [ -f "$DATADIR/$SET/$SET.info" ] && continue
    tmp=$(mktemp -d "$DATADIR/.fetch.XXXXXX")
    ok=0
    for url in "https://lhapdfsets.web.cern.ch/current/$SET.tar.gz" \
               "https://lhapdfsets.web.cern.ch/lhapdfsets/current/$SET.tar.gz"; do
        if curl -fsSL --retry 3 -o "$tmp/$SET.tar.gz" "$url" \
           && tar -xzf "$tmp/$SET.tar.gz" -C "$tmp" \
           && [ -f "$tmp/$SET/$SET.info" ]; then
            rm -rf "${DATADIR:?}/$SET" && mv "$tmp/$SET" "$DATADIR/$SET" && ok=1
            break
        fi
        rm -rf "${tmp:?}"/*
    done
    rm -rf "$tmp"
    [ "$ok" = 1 ] && echo "fetched $SET" || echo "::warning::could not fetch the PDF set $SET"
done
exit 0
