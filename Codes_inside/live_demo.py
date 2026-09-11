"""
live_demo.py  —  ESP32 stream -> double-blink SELECT event
==========================================================
Your collector's ACQUISITION half (reader thread + queue + writer thread), with
the experiment half (Tkinter cues, trial timeline, markers, self-check) removed,
and the trained detector wired in.

Pipeline per sample:
    serial_reader_thread  : read ESP32 line -> (ch0_raw, ch1_raw) -> queue   [real-time, light]
    worker_thread         : bandpass 0.5-10Hz -> det.push() -> (optional) log  [inference off the reader]
    det.on_select         : fires once per confirmed double-blink -> SELECT hook

Keeps: robust real-time decoupling (the timing-bug fix) + background CSV logging.
Removes: everything experiment-specific.

BEFORE trusting live: confirm the 0.5-10Hz band matches your X.npy build script
(see README / preprocessing.py). Then validate with a real double-blink on your
own face — SELECT should fire once, rest should stay silent.
"""

import csv
import os
import queue
import threading
import time
from datetime import datetime

import serial
import serial.tools.list_ports

from live_blink_detector import LiveBlinkDetector
from rule_detector import RuleBasedBlinkDetector
from preprocessing import CollectorFilter
from bci_bridge import BCIBridge

# ---- config (matches your collector + the model contract) ----
DETECTOR_MODE = "rule"            # "rule" (recommended) or "ml"
SAMPLE_RATE = 256
BAUD_RATE = 115200
ADC_MIDPOINT = 4095 / 2
ACTIVE_CHANNELS = [0, 1]          # must match training channel order
CHECKPOINT = "eog-cnn-epoch_03-val_f1_1_0000_ckpt.zip"
NORM_STATS = "norm_stats.json"
LOG_SESSIONS = True               # background CSV of every live session (cheap insurance)
SERVE_UI = True                   # run SSE bridge so the browser grid gets SELECT events
MONITOR = True                    # stream downsampled ch0 + blink pings to the browser monitor
MONITOR_DECIM = 8                 # push every Nth sample (256/8 = 32 Hz); 0 detection cost

stop_event = threading.Event()
# Small queue: with drop-oldest in the reader, this BOUNDS latency to ~2 s even
# if the worker briefly stalls (e.g. disk flush). A big queue lets lag accumulate.
data_queue = queue.Queue(maxsize=512)
bridge = BCIBridge(port=8765) if SERVE_UI else None


# ============================ SELECT / blink hooks ============================
def on_select(info):
    """
    Fires once per confirmed double-blink. Prints, and pushes SELECT to the
    browser grid (which selects the currently gaze-focused tile). This is the
    'click'. When you later retrain the model, nothing here changes.
    """
    prob = info.get("prob")          # ML gives prob; rule gives gap
    tag = f"prob={prob}" if prob is not None else f"gap={info.get('gap')}"
    print(f">>> SELECT   {tag}   t={info['time']:.2f}s")
    if bridge is not None:
        bridge.push("SELECT", **({"prob": prob} if prob is not None else {"gap": info.get("gap")}))


def on_blink(info):
    """Rule-mode only: each detected single blink -> monitor flash (not a select)."""
    if bridge is not None and MONITOR:
        bridge.push("blink")



# ============================ serial ============================
def find_esp32_port():
    ports = serial.tools.list_ports.comports()
    print("Available ports:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device} - {p.description}")
    if not ports:
        raise SystemExit("No serial ports found.")
    choice = input("Port number or name: ").strip()
    if choice.isdigit() and int(choice) < len(ports):
        return ports[int(choice)].device
    return choice or ports[0].device


def serial_reader_thread(ser):
    """Bulk-read all available bytes and split into lines. Far faster than
    readline() per line, so it keeps up with 256 Hz even under GIL contention.
    If it ever falls badly behind, it keeps only the most recent bytes."""
    print("Serial reader started")
    try:
        ser.reset_input_buffer()
    except Exception:
        pass
    buf = b""
    while not stop_event.is_set():
        try:
            n = ser.in_waiting
            if n <= 0:
                time.sleep(0.001)
                continue
            buf += ser.read(n)
            # fell way behind (e.g. a stall) -> keep only recent bytes, drop old
            if len(buf) > 65536:
                buf = buf[-8192:]
                buf = buf[buf.find(b"\n") + 1:]   # realign to a line boundary
            lines = buf.split(b"\n")
            buf = lines[-1]                        # keep the trailing partial line
            for lb in lines[:-1]:
                line = lb.decode("utf-8", "ignore").strip()
                if not line or line == "READY" or ":" in line:
                    continue
                parts = line.split(",")
                if len(parts) == 5:
                    try:
                        ts = int(parts[0])
                        raw = [float(parts[i + 1]) - ADC_MIDPOINT for i in range(4)]
                    except ValueError:
                        continue
                    ch0, ch1 = raw[ACTIVE_CHANNELS[0]], raw[ACTIVE_CHANNELS[1]]
                    try:
                        data_queue.put_nowait({"ts": ts, "ch0": ch0, "ch1": ch1})
                    except queue.Full:
                        try:
                            data_queue.get_nowait()
                            data_queue.put_nowait({"ts": ts, "ch0": ch0, "ch1": ch1})
                        except queue.Empty:
                            pass
        except Exception as e:
            if not stop_event.is_set():
                print("Serial read error:", e)
            time.sleep(0.001)
    print("Serial reader stopped")


# ============================ worker: filter -> detect -> log ============================
def worker_thread(det, pre, logger):
    print("Worker started (filter -> detect)")
    n = 0
    while not stop_event.is_set() or not data_queue.empty():
        try:
            d = data_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        # match training preprocessing (collector's exact filter, per channel, stateful)
        f0 = pre.process(0, d["ch0"])
        f1 = pre.process(1, d["ch1"])
        # feed the detector (inference/logic gated internally)
        det.push(f0, f1)
        if logger:
            logger.write(d["ts"], d["ch0"], d["ch1"], f0, f1)
        # monitor: throttled, non-blocking, and free when no browser is watching
        n += 1
        if MONITOR and bridge is not None and (n % MONITOR_DECIM == 0) and bridge.has_clients():
            bridge.push("signal", v=round(f0, 1))
    print("Worker stopped")


# ============================ optional session logger ============================
class SessionLogger:
    def __init__(self, path):
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["ts_us", "raw_ch0", "raw_ch1", "filt_ch0", "filt_ch1"])
        self.n = 0

    def write(self, ts, r0, r1, f0, f1):
        self.w.writerow([ts, f"{r0:.2f}", f"{r1:.2f}", f"{f0:.4f}", f"{f1:.4f}"])
        self.n += 1
        if self.n % 2560 == 0:      # flush every ~10 s, not every 1 s
            self.f.flush()          # frequent flushing on a synced folder stalls the worker

    def close(self):
        self.f.close()


# ============================ main ============================
def main():
    if DETECTOR_MODE == "rule":
        # Onset/offset detection + amplitude-ratio double guard. Catches abhi's
        # fast (~0.3 s) doubles AND rejects filter-ring false doubles by size.
        det = RuleBasedBlinkDetector(
            on_select=on_select,
            on_blink=on_blink,
            channel=0, threshold_sd=7.0,
            min_gap_s=0.25, max_gap_s=1.1, amp_ratio=0.45,
            fire_debounce_s=1.0, warmup_s=5.0, polarity="neg",
        )
        print("[detector] rule-based (ch0, 7*SD neg, amplitude-ratio double guard)")
    else:
        # ML: 41/50 (82%), 2 false fires. Needs both channels + norm stats.
        det = LiveBlinkDetector(
            checkpoint=CHECKPOINT,
            norm_stats=NORM_STATS,
            on_select=on_select,
            threshold=0.7,
            smooth_window=6,
            smooth_needed=4,
            refractory_s=2.0,
            hop=64,
        )
        print("[detector] ML (EOG_NET)")
    pre = CollectorFilter(fs=SAMPLE_RATE, n_channels=2)   # EXACT training filter

    logger = None
    if LOG_SESSIONS:
        os.makedirs("live_logs", exist_ok=True)
        p = os.path.join("live_logs", f"live_{datetime.now():%Y%m%d_%H%M%S}.csv")
        logger = SessionLogger(p)
        print("Logging session ->", p)

    port = find_esp32_port()
    ser = serial.Serial(port, BAUD_RATE, timeout=1)
    print("Connected", port)
    time.sleep(2)

    reader = threading.Thread(target=serial_reader_thread, args=(ser,), daemon=True)
    worker = threading.Thread(target=worker_thread, args=(det, pre, logger), daemon=True)
    if bridge is not None:
        bridge.start()
    reader.start()
    worker.start()

    print("\nLive. Double-blink to fire SELECT. Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        time.sleep(0.3)
        reader.join(timeout=1)
        worker.join(timeout=2)
        ser.close()
        if logger:
            logger.close()
        print("Stopped.")


if __name__ == "__main__":
    main()