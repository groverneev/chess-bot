#!/usr/bin/env python3
"""Checks that every `bestmove` matches the first move of the last reported PV.

Usage: scripts/pvcheck.py <engine> [searches=600]
Runs short random-movetime searches on the bench FENs; exits nonzero on any mismatch.
"""
import random
import subprocess
import sys
from pathlib import Path

eng = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 600
bench = Path(__file__).resolve().parent.parent / "src" / "bench.rs"
fens = [l.split('"')[1] for l in bench.read_text().splitlines() if l.strip().startswith('"')]

p = subprocess.Popen([eng], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)


def send(cmd):
    p.stdin.write(cmd + "\n")
    p.stdin.flush()


send("uci")
send("setoption name Hash value 16")
send("isready")
while p.stdout.readline().strip() != "readyok":
    pass

rng = random.Random(1)
bad = 0
for _ in range(n):
    send("position fen " + rng.choice(fens))
    send(f"go movetime {rng.randint(2, 60)}")
    last = None
    while True:
        line = p.stdout.readline().strip()
        if " pv " in line:
            last = line.split(" pv ")[1].split()[0]
        if line.startswith("bestmove"):
            if last is not None and last != line.split()[1]:
                bad += 1
            break
send("quit")
p.wait()
print(f"{eng}: {bad}/{n} bestmove/PV mismatches")
sys.exit(1 if bad else 0)
