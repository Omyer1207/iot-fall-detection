import socket
import math
import statistics
import csv
import os
import time
import uuid
from datetime import datetime
from collections import deque

# =========================
# UDP тохиргоо
# =========================

UDP_IP = "0.0.0.0"
UDP_PORT = 5005

# =========================
# Data collection тохиргоо
# =========================

RECORD_SECONDS = 5.0
PRE_BUFFER_SIZE = 30

# Accelerometer data g нэгжтэй бол 0.15 - 0.30 тохиромжтой
ACC_DELTA_THRESHOLD = 0.30

# Gyro data deg/s бол 10 - 30 тохиромжтой
GYRO_THRESHOLD = 25.0

BASELINE_ALPHA = 0.02

RAW_DIR = "raw_sessions"
FEATURES_FILE = "fall_training_features_2.csv"

LABELS = [
    "walking",
    "fall",
    "stand_to_sit",
    "sit_to_stand",
    "bending"
]

os.makedirs(RAW_DIR, exist_ok=True)


# =========================
# Utility functions
# =========================

def mean(values):
    return statistics.mean(values) if values else 0


def std(values):
    return statistics.stdev(values) if len(values) >= 2 else 0


def energy(values):
    return sum(v * v for v in values) / len(values) if values else 0


def now_iso():
    return datetime.now().isoformat(timespec="milliseconds")


def safe_text(text):
    return "".join(c for c in text if c.isalnum() or c in ["_", "-"]).strip()


def update_baseline(current_baseline, acc_mag):
    if current_baseline is None:
        return acc_mag

    return (1 - BASELINE_ALPHA) * current_baseline + BASELINE_ALPHA * acc_mag


def is_motion_detected(acc_mag, gyro_mag, baseline_acc_mag):
    if baseline_acc_mag is None:
        return False

    acc_delta = abs(acc_mag - baseline_acc_mag)

    if acc_delta >= ACC_DELTA_THRESHOLD:
        return True

    if gyro_mag >= GYRO_THRESHOLD:
        return True

    return False


def parse_udp_line(line):
    """
    ESP32-с ирэх format:
    esp_ms,ax,ay,az,gx,gy,gz
    """

    parts = line.strip().split(",")

    if len(parts) != 7:
        return None

    try:
        esp_ms, ax, ay, az, gx, gy, gz = parts

        ax = float(ax)
        ay = float(ay)
        az = float(az)

        gx = float(gx)
        gy = float(gy)
        gz = float(gz)

        acc_mag = math.sqrt(ax * ax + ay * ay + az * az)
        gyro_mag = math.sqrt(gx * gx + gy * gy + gz * gz)

        return {
            "pc_time": time.time(),
            "pc_time_iso": now_iso(),
            "esp_ms": esp_ms,

            "ax": ax,
            "ay": ay,
            "az": az,

            "gx": gx,
            "gy": gy,
            "gz": gz,

            "acc_mag": acc_mag,
            "gyro_mag": gyro_mag
        }

    except ValueError:
        return None


def choose_label():
    print("\n🏷 Label сонгоно уу:")
    print("0. хадгалахгүй / skip")

    for i, label in enumerate(LABELS, start=1):
        print(f"{i}. {label}")

    while True:
        choice = input("Label number сонгоно уу: ").strip()

        if choice == "0":
            return None

        if choice.isdigit():
            index = int(choice)

            if 1 <= index <= len(LABELS):
                return LABELS[index - 1]

        print("❌ 0-5 хооронд сонгоно уу.")


def extract_features(rows):
    b_ax = [x["ax"] for x in rows]
    b_ay = [x["ay"] for x in rows]
    b_az = [x["az"] for x in rows]

    b_gx = [x["gx"] for x in rows]
    b_gy = [x["gy"] for x in rows]
    b_gz = [x["gz"] for x in rows]

    b_amag = [x["acc_mag"] for x in rows]
    b_gmag = [x["gyro_mag"] for x in rows]

    features = {
        "ax_mean": mean(b_ax),
        "ay_mean": mean(b_ay),
        "az_mean": mean(b_az),

        "gx_mean": mean(b_gx),
        "gy_mean": mean(b_gy),
        "gz_mean": mean(b_gz),

        "ax_std": std(b_ax),
        "ay_std": std(b_ay),
        "az_std": std(b_az),

        "gx_std": std(b_gx),
        "gy_std": std(b_gy),
        "gz_std": std(b_gz),

        "ax_min": min(b_ax),
        "ay_min": min(b_ay),
        "az_min": min(b_az),

        "gx_min": min(b_gx),
        "gy_min": min(b_gy),
        "gz_min": min(b_gz),

        "ax_max": max(b_ax),
        "ay_max": max(b_ay),
        "az_max": max(b_az),

        "gx_max": max(b_gx),
        "gy_max": max(b_gy),
        "gz_max": max(b_gz),

        "acc_mag_mean": mean(b_amag),
        "acc_mag_std": std(b_amag),
        "acc_mag_min": min(b_amag),
        "acc_mag_max": max(b_amag),

        "gyro_mag_mean": mean(b_gmag),
        "gyro_mag_std": std(b_gmag),
        "gyro_mag_min": min(b_gmag),
        "gyro_mag_max": max(b_gmag),

        "acc_energy": energy(b_amag),
        "gyro_energy": energy(b_gmag),

        "sample_count": len(rows)
    }

    return features


def save_raw_data(session_id, label, rows):
    safe_label = safe_text(label)
    filename = f"{session_id}_{safe_label}.csv"
    path = os.path.join(RAW_DIR, filename)

    fieldnames = [
        "session_id",
        "label",
        "pc_time",
        "pc_time_iso",
        "esp_ms",
        "ax",
        "ay",
        "az",
        "gx",
        "gy",
        "gz",
        "acc_mag",
        "gyro_mag"
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            out = {
                "session_id": session_id,
                "label": label,
                **row
            }
            writer.writerow(out)

    return path


def append_features(session_id, label, start_time, end_time, rows):
    features = extract_features(rows)

    row = {
        "session_id": session_id,
        "label": label,
        "start_time": start_time,
        "end_time": end_time,
        **features
    }

    file_exists = os.path.exists(FEATURES_FILE)

    with open(FEATURES_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# =========================
# Main app
# =========================

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(0.5)

print("==============================================")
print("📡 Fall Detection Data Collector")
print(f"UDP Port: {UDP_PORT}")
print(f"Record duration: {RECORD_SECONDS} seconds")
print(f"ACC_DELTA_THRESHOLD: {ACC_DELTA_THRESHOLD}")
print(f"GYRO_THRESHOLD: {GYRO_THRESHOLD}")
print("==============================================")
print("🟢 Хөдөлгөөн хүлээж байна...")
print("CTRL + C дарж зогсооно.\n")

baseline_acc_mag = None
pre_buffer = deque(maxlen=PRE_BUFFER_SIZE)

is_recording = False
record_start_time = None
record_start_iso = None
record_rows = []

last_status_print = 0

try:
    while True:
        try:
            data, addr = sock.recvfrom(1024)
        except socket.timeout:
            continue

        line = data.decode(errors="ignore").strip()
        sample = parse_udp_line(line)

        if sample is None:
            continue

        current_time = time.time()

        acc_mag = sample["acc_mag"]
        gyro_mag = sample["gyro_mag"]

        # =========================
        # IDLE state
        # =========================

        if not is_recording:
            pre_buffer.append(sample)

            motion = is_motion_detected(
                acc_mag=acc_mag,
                gyro_mag=gyro_mag,
                baseline_acc_mag=baseline_acc_mag
            )

            if not motion:
                baseline_acc_mag = update_baseline(
                    current_baseline=baseline_acc_mag,
                    acc_mag=acc_mag
                )

                if current_time - last_status_print >= 2.0:
                    print(
                        f"🟢 Idle | acc_mag={acc_mag:.3f} | "
                        f"baseline={baseline_acc_mag:.3f} | "
                        f"gyro_mag={gyro_mag:.3f}"
                    )
                    last_status_print = current_time

                continue

            # Хөдөлгөөн илэрсэн
            print("\n🎯 ХӨДӨЛГӨӨН ИЛЭРЛЭЭ!")
            print("⏺ 5 секунд data цуглуулж эхэллээ...")

            is_recording = True
            record_start_time = current_time
            record_start_iso = now_iso()

            # Хөдөлгөөн эхлэхээс өмнөх бага зэрэг data-г хамт авна
            record_rows = list(pre_buffer)
            record_rows.append(sample)

            last_status_print = current_time

            continue

        # =========================
        # RECORDING state
        # =========================

        record_rows.append(sample)

        elapsed = current_time - record_start_time

        if elapsed < RECORD_SECONDS:
            if current_time - last_status_print >= 1.0:
                remaining = RECORD_SECONDS - elapsed
                print(f"⏳ Data цуглуулж байна... үлдсэн {remaining:.1f} сек")
                last_status_print = current_time

            continue

        # =========================
        # RECORD FINISHED
        # =========================

        record_end_iso = now_iso()

        print("\n✅ Data цуглуулалт дууслаа.")
        print(f"📊 Нийт sample count: {len(record_rows)}")

        label = choose_label()

        if label is None:
            print("\n🗑 Data хадгалсангүй. Энэ session skip хийгдлээ.")
        else:
            session_id = str(uuid.uuid4())[:8]

            raw_path = save_raw_data(
                session_id=session_id,
                label=label,
                rows=record_rows
            )

            append_features(
                session_id=session_id,
                label=label,
                start_time=record_start_iso,
                end_time=record_end_iso,
                rows=record_rows
            )

            print("\n💾 Data хадгалагдлаа!")
            print(f"Raw data: {raw_path}")
            print(f"Feature data: {FEATURES_FILE}")

        print("==============================================")
        print("🟢 Дараагийн хөдөлгөөн хүлээж байна...\n")

        # =========================
        # State reset
        # =========================

        is_recording = False
        record_start_time = None
        record_start_iso = None
        record_rows = []
        pre_buffer.clear()

        # Дараагийн idle baseline-г одоогийн байрлалаас дахин эхлүүлнэ
        baseline_acc_mag = acc_mag
        last_status_print = current_time

except KeyboardInterrupt:
    print("\n🛑 Програм зогслоо.")