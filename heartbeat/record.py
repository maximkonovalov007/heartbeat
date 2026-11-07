"""Record the tower's vibration data to a CSV file.
Usage:  python record.py <label> <seconds> [port]
Example: python record.py baseline 30 COM17
Needs:   pip install pyserial
"""
import sys, time, os, serial

HEADER = "t_us,f1x,f1y,f1z,f2x,f2y,f2z,f3x,f3y,f3z,errors"

label = sys.argv[1] if len(sys.argv) > 1 else "test"
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 20
port = sys.argv[3] if len(sys.argv) > 3 else "COM17"

os.makedirs("data", exist_ok=True)
path = os.path.join("data", f"{label}_{time.strftime('%H%M%S')}.csv")
ser = serial.Serial(port, 115200, timeout=1)
ser.reset_input_buffer()
rows, start = 0, None
print(f"Recording {seconds:.0f} s from {port} into {path} ... tap now")
with open(path, "w") as f:
    f.write(HEADER + "\n")
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if not line or line.startswith("#") or line.startswith("t_us"):
            continue
        parts = line.split(",")
        if len(parts) != 11:
            continue
        try:
            float(parts[0]); float(parts[9])
        except ValueError:
            continue
        if start is None:
            start = time.time()
        f.write(line + "\n")
        rows += 1
        if time.time() - start >= seconds:
            break
print(f"Done: {rows} rows ({rows/seconds:.0f} per second). Saved {path}")
