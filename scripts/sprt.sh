#!/usr/bin/env bash
# SPRT a candidate binary against a baseline with fastchess.
# Usage: scripts/sprt.sh <new> <old> [tc=8+0.08] [elo0=0] [elo1=5] [concurrency=nproc-1]
# Expects tools/fastchess/fastchess and tools/books/UHO_Lichess_4852_v1.epd (see PROGRESS.md).
set -euo pipefail
cd "$(dirname "$0")/.."
new=$1 old=$2
tc=${3:-8+0.08} elo0=${4:-0} elo1=${5:-5}
conc=${6:-$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))}
mkdir -p data
stamp=$(date +%Y%m%d-%H%M%S)
exec tools/fastchess/fastchess \
    -engine cmd="$new" name="$(basename "$new")" \
    -engine cmd="$old" name="$(basename "$old")" \
    -each proto=uci tc="$tc" option.Hash=16 option.Threads=1 \
    -openings file=tools/books/UHO_Lichess_4852_v1.epd format=epd order=random \
    -games 2 -rounds 50000 -repeat -recover \
    -concurrency "$conc" \
    -sprt elo0="$elo0" elo1="$elo1" alpha=0.05 beta=0.05 \
    -ratinginterval 50 \
    -pgnout file="data/sprt-$stamp.pgn"
