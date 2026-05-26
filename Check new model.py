import socket
import math
import statistics
import pandas as pd
import joblib
import warnings
import time
from datetime import datetime
from collections import deque

warnings.filterwarnings("ignore")

# =========================
# UDP тохиргоо
# =========================

UDP_IP = "0.0.0.0"
UDP_PORT = 5005

MODEL_FILE = "fall_detection_model.pkl"

# =========================
# Predict тохиргоо
# =========================

RECORD_SECONDS = 5.0
PRE_BUFFER_SIZE = 30

ACC_DELTA_THRESHOLD = 0.30
GYRO_THRESHOLD = 25.0
BASELINE_ALPHA = 0.02

FALL_HOLD_TIME = 5.0

LABELS = [
    "walking",
    "fall",
    "stand_to_sit",
    "sit_to_stand",
    "bending"
]

DEFAULT_FEATURE_COLUMNS = [
    "ax_mean", "ay_mean", "az_mean",
    "gx_mean", "gy_mean", "gz_mean",

    "ax_std", "ay_std", "az_std",
    "gx_std", "gy_std", "gz_std",

    "ax_min", "ay_min", "az_min",
    "gx_min", "gy_min", "gz_min",

    "ax_max", "ay_max", "az_max",
    "gx_max", "gy_max", "gz_max",

    "acc_mag_mean", "acc_mag_std",
    "acc_mag_min", "acc_mag_max",

    "gyro_mag_mean", "gyro_mag_std",
    "gyro_mag_min", "gyro_mag_max",

    "acc_energy", "gyro_energy",

    "sample_count"
]


def mean(values):
    return statistics.mean(values) if values else 0


def std(values):
    return statistics.stdev(values) if len(values) >= 2 else 0


def energy(values):
    return sum(v * v for v in values) / len(values) if values else 0


def now_iso():
    return datetime.now().isoformat(timespec="milliseconds")


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


print("==============================================")
print("🔮 Fall Detection Real-time Prediction")
print("==============================================")
print(f"⏳ Model уншиж байна: {MODEL_FILE}")

try:
    package = joblib.load(MODEL_FILE)

    if isinstance(package, dict):
        model = package["model"]
        FEATURE_COLUMNS = package.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
        LABELS = package.get("labels", LABELS)
    else:
        model = package
        FEATURE_COLUMNS = DEFAULT_FEATURE_COLUMNS

    print("✅ Model амжилттай уншлаа.")

except Exception as e:
    print(f"❌ Model уншихад алдаа гарлаа: {e}")
    exit()


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(0.5)

print(f"📡 UDP data хүлээж байна. Port: {UDP_PORT}")
print(f"Record duration: {RECORD_SECONDS} seconds")
print(f"ACC_DELTA_THRESHOLD: {ACC_DELTA_THRESHOLD}")
print(f"GYRO_THRESHOLD: {GYRO_THRESHOLD}")
print("==============================================")
print("🟢 Хөдөлгөөн хүлээж байна...\n")

baseline_acc_mag = None
pre_buffer = deque(maxlen=PRE_BUFFER_SIZE)

is_recording = False
record_start_time = None
record_rows = []

last_status_print = 0
last_fall_time = 0

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

        if current_time - last_fall_time < FALL_HOLD_TIME:
            if current_time - last_status_print >= 1.0:
                print("⚠️ Уналтын дараах cooldown үе...")
                last_status_print = current_time
            continue

        if not is_recording:
            pre_buffer.append(sample)

            motion = is_motion_detected(
                acc_mag=acc_mag,
                gyro_mag=gyro_mag,
                baseline_acc_mag=baseline_acc_mag
            )

            if not motion:
                baseline_acc_mag = update_baseline(baseline_acc_mag, acc_mag)

                if current_time - last_status_print >= 2.0:
                    print(
                        f"🟢 Idle | acc_mag={acc_mag:.3f} | "
                        f"baseline={baseline_acc_mag:.3f} | gyro_mag={gyro_mag:.3f}"
                    )
                    last_status_print = current_time

                continue

            print("\n🎯 Хөдөлгөөн илэрлээ!")
            print("⏺ Prediction хийхийн тулд 5 секунд data цуглуулж байна...")

            is_recording = True
            record_start_time = current_time
            record_rows = list(pre_buffer)
            record_rows.append(sample)

            continue

        record_rows.append(sample)

        elapsed = current_time - record_start_time

        if elapsed < RECORD_SECONDS:
            if current_time - last_status_print >= 1.0:
                remaining = RECORD_SECONDS - elapsed
                print(f"⏳ Data цуглуулж байна... үлдсэн {remaining:.1f} сек")
                last_status_print = current_time

            continue

        print("\n✅ 5 секунд data цуглуулалт дууслаа.")
        print(f"📊 Sample count: {len(record_rows)}")

        features = extract_features(record_rows)
        df_features = pd.DataFrame([features])

        df_features = df_features[FEATURE_COLUMNS]

        prediction = model.predict(df_features)[0]

        confidence_text = ""

        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(df_features)[0]
                classes = model.classes_
                max_index = proba.argmax()
                confidence = proba[max_index]
                confidence_text = f" | confidence={confidence:.2f}"
            except Exception:
                confidence_text = ""

        if prediction == "fall":
            print("🚨🚨 УНАЛТ ИЛЭРЛЭЭ!!! FALL DETECTED 🚨🚨")
            proba = model.predict_proba(df_features)[0]
            classes = model.classes_
            max_index = proba.argmax()
            confidence = proba[max_index]
            confidence_text = f" | confidence={confidence:.2f}"
            last_fall_time = current_time
        else:
            print(f"👉 Илэрсэн хөдөлгөөн: {str(prediction).upper()}{confidence_text}")

        print("==============================================")
        print("🟢 Дараагийн хөдөлгөөн хүлээж байна...\n")

        is_recording = False
        record_start_time = None
        record_rows = []
        pre_buffer.clear()

        baseline_acc_mag = acc_mag

except KeyboardInterrupt:
    print("\n🛑 Програм зогслоо.")