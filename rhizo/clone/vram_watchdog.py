"""
obj_009.1 — vram_watchdog.py : overnight VRAM safety net. Polls nvidia-smi every 30s. On SUSTAINED spill
(> THRESH_MB for CONSEC consecutive polls) it (a) logs a loud alert and (b) auto-shrinks every batch cap in
intermediate/spill_caps.json by 2 (floor 4) — ablation_v2.batch_for re-reads that file each cell, so the NEXT
cell automatically runs a smaller, spill-free batch. It never kills a running cell (avoids losing work; the
per-cell spill would be transient anyway). Heartbeat + peak logged to logs/vram_watch.log.
Stop it by creating intermediate/watchdog_stop (or when the grid is done).
"""
import time, subprocess, json, os

HERE = os.path.dirname(__file__)
EXP = os.path.join(HERE, "..")
CAPS = os.path.join(EXP, "intermediate", "spill_caps.json")
LOG = os.path.join(EXP, "logs", "vram_watch.log")
STOP = os.path.join(EXP, "intermediate", "watchdog_stop")
DONE_MARK = os.path.join(EXP, "results", "obj0091_grid.log")
THRESH_MB = 7900       # dedicated VRAM 8192; display ~300-600 -> spill onset ~7.9 GB
CONSEC = 3             # 3 x 30s = 90s sustained before mitigating (ignores transients)


def used_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=20)
    return int(out.stdout.strip().split("\n")[0])


def logln(s):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(s + "\n")


def mitigate():
    d = json.load(open(CAPS))
    for k in ("variant_batch_cap", "variant_batch_cap_k6"):
        for v in d.get(k, {}):
            d[k][v] = max(4, int(d[k][v]) - 2)
    json.dump(d, open(CAPS, "w"), indent=2)
    return d["variant_batch_cap"]


def main():
    logln(f"{time.strftime('%H:%M:%S')} watchdog START thresh={THRESH_MB}MB consec={CONSEC}")
    consec = 0; peak = 0; i = 0
    while not os.path.exists(STOP):
        try:
            mb = used_mb()
        except Exception as e:
            logln(f"{time.strftime('%H:%M:%S')} nvidia-smi error {e}"); time.sleep(30); continue
        i += 1; peak = max(peak, mb)
        hi = mb > THRESH_MB
        consec = consec + 1 if hi else 0
        if consec >= CONSEC:
            newcaps = mitigate()
            logln(f"{time.strftime('%H:%M:%S')} *** SPILL SUSTAINED {mb}MB -> caps lowered to {newcaps} ***")
            consec = 0
        elif hi:
            logln(f"{time.strftime('%H:%M:%S')} HIGH {mb}MB (consec {consec})")
        elif i % 10 == 0:
            logln(f"{time.strftime('%H:%M:%S')} heartbeat {mb}MB peak={peak}MB")
        # auto-stop when the grid finishes
        try:
            if os.path.exists(DONE_MARK) and "GRID DONE" in open(DONE_MARK, encoding="utf-8").read()[-400:]:
                logln(f"{time.strftime('%H:%M:%S')} grid done -> watchdog exit (peak={peak}MB)"); break
        except Exception:
            pass
        time.sleep(30)
    logln(f"{time.strftime('%H:%M:%S')} watchdog STOP (peak={peak}MB)")


if __name__ == "__main__":
    main()
