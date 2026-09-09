"""Fixed-pool strength gauntlet: run ``harness.arena`` for <agent> against each
opponent in a pool -- many games each, the arenas in parallel -- then aggregate into
per-opponent and combined Elo with 95% intervals.

The 16-game mirror arena's interval is about +-180 Elo, too wide to read a small eval
or search change. A few hundred games against a stable pool closes it to roughly +-30,
so a Texel retune / king-safety tweak / time-management change can be A/B'd instead of
shipped on faith and watched on the ladder days later.

Each opponent gets its own ``python -m harness.arena`` subprocess (fully isolated, the
harness used exactly as intended), so parallelism is just "one arena per pool member".

    uv run python -m tools.gauntlet --agent . \
        --pool versions/diag12 versions/diag15 baselines/minimax baselines/greedy \
        --games-per 150

Writes tuning/gauntlet/<timestamp>.json.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_NUM = r"([+-]?\d+(?:\.\d+)?)"
_WDL = re.compile(r"\+(\d+) =(\d+) -(\d+), score")
_ELO = re.compile(rf"Elo {_NUM}, 95% interval {_NUM} to {_NUM}")


def _elo(score: float) -> float:
    score = min(max(score, 1e-6), 1.0 - 1e-6)
    return 400.0 * math.log10(score / (1.0 - score))


def _interval(w: int, d: int, loss: int) -> tuple[float, float, float] | None:
    n = w + d + loss
    if n < 2:
        return None
    score = (w + d / 2) / n
    spread = w * (1 - score) ** 2 + d * (0.5 - score) ** 2 + loss * score**2
    margin = 1.96 * math.sqrt(spread / (n - 1) / n)
    if not (score - margin > 0.0 and score + margin < 1.0):
        return (_elo(score), float("nan"), float("nan"))
    return (_elo(score), _elo(score - margin), _elo(score + margin))


def _run_arena(agent: str, opp: str, games: int, base_ms: int, inc_ms: int) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "harness.arena",
        "--agent",
        agent,
        "--opponent",
        opp,
        "--games",
        str(games),
        "--base-ms",
        str(base_ms),
        "--increment-ms",
        str(inc_ms),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:  # a flaky opponent must not kill a multi-hour gauntlet
        return {"opponent": opp, "error": f"{type(exc).__name__}: {exc}", "returncode": -1}
    out = proc.stdout + proc.stderr
    m = _WDL.search(out)
    if not m:
        return {"opponent": opp, "error": out[-2000:], "returncode": proc.returncode}
    w, d, loss = int(m.group(1)), int(m.group(2)), int(m.group(3))
    row: dict[str, object] = {
        "opponent": opp,
        "w": w,
        "d": d,
        "l": loss,
        "returncode": proc.returncode,
    }
    e = _ELO.search(out)
    if e:
        row["elo"] = float(e.group(1))
        row["elo_lo"] = float(e.group(2))
        row["elo_hi"] = float(e.group(3))
    if proc.returncode != 0:
        row["warning"] = "harness.arena exited non-zero (an agent may have failed a game)"
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", type=Path, default=Path("."))
    p.add_argument("--pool", type=Path, nargs="+", required=True)
    p.add_argument("--games-per", type=int, default=150, help="games vs each opponent")
    p.add_argument("--base-ms", type=int, default=10_000)
    p.add_argument("--increment-ms", type=int, default=100)
    p.add_argument("--workers", type=int, default=0, help="parallel arenas (0 = one per opponent)")
    a = p.parse_args()

    agent = str(a.agent.resolve())
    pool = [str(o.resolve()) for o in a.pool]
    games = max(2, a.games_per - a.games_per % 2)
    workers = a.workers or len(pool)
    print(
        f"gauntlet: {a.agent} vs {len(pool)} opponents x {games} games "
        f"({len(pool) * games} total), {workers} arenas in parallel"
    )
    started = time.time()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(lambda o: _run_arena(agent, o, games, a.base_ms, a.increment_ms), pool))

    tot = [0, 0, 0]
    print(f"\n  {'opponent':22s} {'W/D/L':>12s} {'score':>7s} {'Elo':>7s}   95% interval")
    for row in sorted(rows, key=lambda r: r.get("elo", 1e9)):
        name = Path(str(row["opponent"])).name
        if "error" in row:
            print(f"  {name:22s}   FAILED (rc={row['returncode']}) -- see json")
            continue
        w, d, loss = int(row["w"]), int(row["d"]), int(row["l"])
        tot[0] += w
        tot[1] += d
        tot[2] += loss
        n = w + d + loss
        score = (w + d / 2) / n if n else 0.0
        lohi = ""
        if "elo_lo" in row and row["elo_lo"] == row["elo_lo"]:  # not nan
            lohi = f"[{row['elo_lo']:+.0f}, {row['elo_hi']:+.0f}]"
        elo = row.get("elo", _elo(score))
        flag = "  !!" if row.get("warning") else ""
        print(f"  {name:22s} {f'+{w} ={d} -{loss}':>12s} {score:>6.1%} {elo:>+7.0f}   {lohi}{flag}")

    n = sum(tot)
    score = (tot[0] + tot[1] / 2) / n if n else 0.0
    ci = _interval(*tot)
    ci_s = f"[{ci[1]:+.0f}, {ci[2]:+.0f}]" if ci and ci[1] == ci[1] else ""
    print(
        f"\n  {'COMBINED':22s} {f'+{tot[0]} ={tot[1]} -{tot[2]}':>12s} "
        f"{score:>6.1%} {_elo(score):>+7.0f}   {ci_s}"
    )
    print(f"\n  {n} games in {(time.time() - started) / 60:.1f} min")

    out_dir = Path("tuning/gauntlet")
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{int(time.time())}.json"
    dest.write_text(
        json.dumps(
            {
                "agent": str(a.agent),
                "base_ms": a.base_ms,
                "increment_ms": a.increment_ms,
                "games_per": games,
                "per_opponent": rows,
                "combined": {
                    "w": tot[0],
                    "d": tot[1],
                    "l": tot[2],
                    "score": score,
                    "elo": _elo(score),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  wrote {dest}")


if __name__ == "__main__":
    main()
