import asyncio
import json
import math
import socket
import statistics
import time
import threading
import warnings
from collections import deque

import numpy as np
import paho.mqtt.client as mqtt
from bleak import BleakScanner
from flask import Flask, render_template_string, Response

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

THINGY_MAC = "E0:8C:FE:B2:B1:06"

MQTT_BROKER = "172.16.158.210"
MQTT_PORT = 1883
UDP_IP = "0.0.0.0"
UDP_PORT = 5005
WEB_PORT = 5006

MODEL_FILE = "fall_detection_model.pkl"

RECORD_SECONDS = 5.0
PRE_BUFFER_SIZE = 30

ACC_DELTA_THRESHOLD = 0.30
GYRO_THRESHOLD = 25.0
BASELINE_ALPHA = 0.02
FALL_HOLD_TIME = 5.0
ACTIVITY_HOLD_TIME = 4.0

ACTIVITY_LABELS = [
    "walking",
    "fall",
    "stand_to_sit",
    "sit_to_stand",
    "bending",
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
    "sample_count",
]

ROOM_W = 4.5
ROOM_H = 5.0

BASE_NODE_POS = {
    "node2": (0.0, 0.0),
    "node1": (3.4, 0.0),
    "node3": (0.6176, 4.1534),
}

FLIP_X = True
FLIP_Y = False


def apply_flip_positions(base_positions):
    flipped = {}

    for node, (x, y) in base_positions.items():
        if FLIP_X:
            x = ROOM_W - x
        if FLIP_Y:
            y = ROOM_H - y

        flipped[node] = (x, y)

    return flipped


NODE_POS = apply_flip_positions(BASE_NODE_POS)

# ─────────────────────────────────────────────────────────────
# NONLINEAR RSSI → DISTANCE CALIBRATION TABLE
# ─────────────────────────────────────────────────────────────

DISTANCE_TABLE_BY_NODE = {
    "node1": [
        (-45, 0.3),
        (-52, 1.0),
        (-60, 2.0),
        (-64, 3.0),
        (-70, 4.0),
        (-74, 5.0),
    ],
    "node2": [
        (-45, 0.3),
        (-52, 1.0),
        (-60, 2.0),
        (-64, 3.0),
        (-70, 4.0),
        (-74, 5.0),
    ],
    "node3": [
        (-56, 0.3),
        (-60, 1.0),
        (-64, 2.0),
        (-70, 3.0),
        (-76, 4.0),
        (-79, 5.0),
    ],
}

CURVE_EXP_BY_NODE = {
    "node1": 1.25,
    "node2": 1.25,
    "node3": 1.15,
}

WINDOW = 5
MEDIAN_LAST_N = 3
BLE_PUBLISH_INTERVAL = 0.2

WEB_POLL_INTERVAL_MS = 150

# ─────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────

latest_rssi = {n: None for n in NODE_POS}
raw_rssi_latest = {n: None for n in NODE_POS}
node_last_seen = {n: None for n in NODE_POS}
history = {n: deque(maxlen=WINDOW) for n in NODE_POS}

latest_position = {
    "x": None,
    "y": None,
    "x_raw": None,
    "y_raw": None,
}

position_trail = deque(maxlen=80)

activity_state = {
    "label": "idle",
    "confidence": None,
    "status": "starting",
    "is_recording": False,
    "sample_count": 0,
    "acc_mag": None,
    "gyro_mag": None,
    "baseline_acc_mag": None,
    "last_event_time": None,
    "last_sample_time": None,
    "error": None,
}

last_ble_publish = 0
mqtt_connected = False

state_lock = threading.Lock()
log_lock = threading.Lock()

web_logs = deque(maxlen=200)
log_seq = 0

web_log_colors = {
    "info": "#4a9",
    "warn": "#f90",
    "err": "#f55",
    "mqtt": "#00e5ff",
    "ble": "#a8ff3e",
    "pos": "#ffe033",
}


def wlog(text, kind="info"):
    global log_seq

    color = web_log_colors.get(kind, "#4a9")

    with log_lock:
        log_seq += 1
        web_logs.append({
            "id": log_seq,
            "text": str(text),
            "color": color,
            "ts": time.time(),
        })

    print(text)


# ─────────────────────────────────────────────────────────────
# KALMAN FILTER
# ─────────────────────────────────────────────────────────────

class KalmanFilter:
    def __init__(self, process_noise=0.12, measurement_noise=3.0, estimate_error=1.0):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.estimate_error = estimate_error
        self.estimate = None

    def update(self, measurement):
        measurement = float(measurement)

        if self.estimate is None:
            self.estimate = measurement
            return self.estimate

        self.estimate_error += self.process_noise

        kalman_gain = self.estimate_error / (
            self.estimate_error + self.measurement_noise
        )

        self.estimate = self.estimate + kalman_gain * (
            measurement - self.estimate
        )

        self.estimate_error = (1.0 - kalman_gain) * self.estimate_error

        return self.estimate


rssi_filters = {
    node: KalmanFilter(process_noise=0.12, measurement_noise=3.0)
    for node in NODE_POS
}

x_filter = KalmanFilter(process_noise=0.25, measurement_noise=0.4)
y_filter = KalmanFilter(process_noise=0.25, measurement_noise=0.4)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def extract_rssi_from_payload(data):
    if not isinstance(data, dict):
        return None

    for key in ["rssi_raw", "rssi", "rssi_filtered"]:
        if key in data and data[key] is not None:
            try:
                return float(data[key])
            except Exception:
                return None

    return None


def nonlinear_interpolate_distance(node, rssi):
    table = DISTANCE_TABLE_BY_NODE[node]
    curve_exp = CURVE_EXP_BY_NODE.get(node, 1.0)

    if rssi >= table[0][0]:
        return table[0][1]

    if rssi <= table[-1][0]:
        return table[-1][1]

    for i in range(len(table) - 1):
        rssi1, dist1 = table[i]
        rssi2, dist2 = table[i + 1]

        if rssi1 >= rssi >= rssi2:
            ratio = (rssi1 - rssi) / (rssi1 - rssi2)
            curved_ratio = ratio ** curve_exp
            dist = dist1 + curved_ratio * (dist2 - dist1)
            return max(0.1, dist)

    return table[-1][1]


def rssi_to_distance(node, rssi):
    if rssi is None:
        return None

    return nonlinear_interpolate_distance(node, rssi)


def mean(values):
    return statistics.mean(values) if values else 0


def std(values):
    return statistics.stdev(values) if len(values) >= 2 else 0


def energy(values):
    return sum(v * v for v in values) / len(values) if values else 0


def update_activity_state(**kwargs):
    with state_lock:
        activity_state.update(kwargs)


def update_baseline(current_baseline, acc_mag):
    if current_baseline is None:
        return acc_mag

    return (1 - BASELINE_ALPHA) * current_baseline + BASELINE_ALPHA * acc_mag


def is_motion_detected(acc_mag, gyro_mag, baseline_acc_mag):
    if baseline_acc_mag is None:
        return False

    acc_delta = abs(acc_mag - baseline_acc_mag)
    return acc_delta >= ACC_DELTA_THRESHOLD or gyro_mag >= GYRO_THRESHOLD


def parse_udp_line(line):
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
            "esp_ms": esp_ms,
            "ax": ax,
            "ay": ay,
            "az": az,
            "gx": gx,
            "gy": gy,
            "gz": gz,
            "acc_mag": acc_mag,
            "gyro_mag": gyro_mag,
        }
    except ValueError:
        return None


def extract_activity_features(rows):
    b_ax = [x["ax"] for x in rows]
    b_ay = [x["ay"] for x in rows]
    b_az = [x["az"] for x in rows]
    b_gx = [x["gx"] for x in rows]
    b_gy = [x["gy"] for x in rows]
    b_gz = [x["gz"] for x in rows]
    b_amag = [x["acc_mag"] for x in rows]
    b_gmag = [x["gyro_mag"] for x in rows]

    return {
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
        "sample_count": len(rows),
    }


def trilaterate(nodes, dists):
    ref = "node2"
    others = [n for n in nodes if n != ref]

    x0, y0 = nodes[ref]
    r0 = dists[ref]

    A_rows = []
    b_rows = []

    for name in others:
        xi, yi = nodes[name]
        ri = dists[name]

        a1 = 2 * (xi - x0)
        a2 = 2 * (yi - y0)

        b = (
            r0 ** 2
            - ri ** 2
            + xi ** 2
            - x0 ** 2
            + yi ** 2
            - y0 ** 2
        )

        weight = 1.0 / (ri + 0.01)

        A_rows.append([a1 * weight, a2 * weight])
        b_rows.append(b * weight)

    A = np.array(A_rows, dtype=float)
    b = np.array(b_rows, dtype=float)

    try:
        cond = np.linalg.cond(A)
    except Exception:
        return None, None

    if cond > 1e8:
        wlog(f"⚠️ Ill-conditioned matrix cond={cond:.1e}", "warn")
        return None, None

    try:
        pos, *_ = np.linalg.lstsq(A, b, rcond=None)
        return float(pos[0]), float(pos[1])
    except Exception as e:
        wlog(f"❌ Trilateration failed: {e}", "err")
        return None, None


def get_missing_nodes():
    return [n for n in latest_rssi if latest_rssi[n] is None]


def process_rssi_reading(node, rssi_raw, log_kind="mqtt"):
    if node not in NODE_POS:
        wlog(f"⚠️ Unknown node: {node}", "warn")
        return

    if rssi_raw is None:
        wlog(f"⚠️ No RSSI value from {node}", "warn")
        return

    rssi_raw = clamp(float(rssi_raw), -100, -20)
    log_parts = {}

    with state_lock:
        now = time.time()

        raw_rssi_latest[node] = rssi_raw
        node_last_seen[node] = now

        history[node].append(rssi_raw)

        recent_values = list(history[node])[-MEDIAN_LAST_N:]
        recent_values = sorted(recent_values)
        median_rssi = recent_values[len(recent_values) // 2]

        rssi_filtered = rssi_filters[node].update(median_rssi)
        rssi_filtered = clamp(rssi_filtered, -100, -20)

        latest_rssi[node] = rssi_filtered

        ready = all(latest_rssi[n] is not None for n in latest_rssi)

        if ready:
            dists = {
                n: rssi_to_distance(n, latest_rssi[n])
                for n in latest_rssi
            }

            x_raw, y_raw = trilaterate(NODE_POS, dists)

            if x_raw is not None and y_raw is not None:
                x = x_filter.update(x_raw)
                y = y_filter.update(y_raw)

                x = clamp(x, 0.0, ROOM_W)
                y = clamp(y, 0.0, ROOM_H)

                latest_position["x"] = x
                latest_position["y"] = y
                latest_position["x_raw"] = x_raw
                latest_position["y_raw"] = y_raw

                position_trail.append((x, y))

                log_parts["pos"] = (x, y, x_raw, y_raw, dists)
        else:
            log_parts["waiting"] = get_missing_nodes()

        log_parts["rssi"] = (node, rssi_raw, rssi_filtered)

    node_name, raw_value, filtered_value = log_parts["rssi"]

    wlog(
        f"{node_name}: raw={raw_value:.1f} dBm | kalman={filtered_value:.2f} dBm",
        log_kind,
    )

    if "waiting" in log_parts:
        missing = log_parts["waiting"]
        wlog(f"⏳ Waiting for nodes: {missing}", "warn")

    elif "pos" in log_parts:
        x, y, x_raw, y_raw, dists = log_parts["pos"]

        dist_text = " | ".join(
            f"{n}: {dists[n]:.2f}m"
            for n in dists
        )

        wlog(
            f"📍 X={x:.3f}m Y={y:.3f}m | RAW X={x_raw:.3f}, Y={y_raw:.3f} | {dist_text}",
            "pos",
        )


# ─────────────────────────────────────────────────────────────
# MQTT
# ─────────────────────────────────────────────────────────────

def create_mqtt_client():
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    except Exception:
        return mqtt.Client()


mqttc = create_mqtt_client()


def on_connect(client, userdata, flags, rc):
    global mqtt_connected

    if rc == 0:
        mqtt_connected = True
        wlog(f"✅ MQTT connected to {MQTT_BROKER}:{MQTT_PORT}", "mqtt")

        for node in NODE_POS:
            topic = f"nordic/{node}/rssi"
            client.subscribe(topic)
            wlog(f"📡 Subscribed: {topic}", "mqtt")
    else:
        mqtt_connected = False
        wlog(f"❌ MQTT connect failed rc={rc}", "err")


def on_disconnect(client, userdata, rc):
    global mqtt_connected

    mqtt_connected = False

    if rc != 0:
        wlog(f"⚠️ MQTT disconnected unexpectedly rc={rc}. Reconnecting...", "warn")
    else:
        wlog("MQTT disconnected", "mqtt")


def on_message(client, userdata, msg):
    try:
        payload_text = msg.payload.decode("utf-8", errors="replace")
        data = json.loads(payload_text)
    except Exception as e:
        wlog(f"❌ Invalid JSON: {msg.topic} | {e}", "err")
        return

    try:
        parts = msg.topic.split("/")
        node = parts[1]
    except Exception:
        wlog(f"❌ Invalid topic: {msg.topic}", "err")
        return

    rssi_raw = extract_rssi_from_payload(data)

    if rssi_raw is None:
        wlog(f"⚠️ No RSSI in payload from {node}: {data}", "warn")
        return

    process_rssi_reading(node, rssi_raw, "mqtt")


# ─────────────────────────────────────────────────────────────
# LOCAL BLE SCANNER: LAPTOP = NODE3
# ─────────────────────────────────────────────────────────────

def detection_callback(device, adv):
    global last_ble_publish

    if device.address.upper() != THINGY_MAC.upper():
        return

    now = time.time()

    if now - last_ble_publish < BLE_PUBLISH_INTERVAL:
        return

    rssi = getattr(adv, "rssi", None)

    if rssi is None:
        return

    payload = {
        "mac": device.address,
        "rssi_raw": rssi,
        "timestamp": now,
        "source": "local_laptop_node3",
    }

    process_rssi_reading("node3", rssi, "ble")

    result = mqttc.publish("nordic/node3/rssi", json.dumps(payload))

    last_ble_publish = now

    wlog(f"node3 BLE raw RSSI: {rssi} dBm | publish rc={result.rc}", "ble")


def ble_loop():
    async def _run():
        scanner = BleakScanner(detection_callback)

        await scanner.start()

        wlog("✅ Local BLE scanning started as node3", "ble")

        try:
            while True:
                await asyncio.sleep(1)
        finally:
            await scanner.stop()

    try:
        asyncio.run(_run())
    except Exception as e:
        wlog(f"❌ BLE scanner error: {e}", "err")


# ─────────────────────────────────────────────────────────────
# UDP ACTIVITY / FALL DETECTOR
# ─────────────────────────────────────────────────────────────

def activity_loop():
    warnings.filterwarnings("ignore")

    try:
        import pandas as pd
    except Exception as e:
        update_activity_state(status="disabled", error=f"Missing dependency: {e}")
        wlog(f"❌ Activity detector disabled. Missing dependency: {e}", "err")
        return

    try:
        try:
            import joblib
            package = joblib.load(MODEL_FILE)
        except ImportError:
            import pickle

            with open(MODEL_FILE, "rb") as model_file:
                package = pickle.load(model_file)

        if isinstance(package, dict):
            model = package["model"]
            feature_columns = package.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
        else:
            model = package
            feature_columns = DEFAULT_FEATURE_COLUMNS

        update_activity_state(status="model_loaded", error=None)
        wlog(f"✅ Activity model loaded: {MODEL_FILE}", "info")
    except Exception as e:
        update_activity_state(status="disabled", error=f"Model load failed: {e}")
        wlog(f"❌ Activity model load failed: {e}", "err")
        return

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))
        sock.settimeout(0.5)
    except Exception as e:
        update_activity_state(status="disabled", error=f"UDP bind failed: {e}")
        wlog(f"❌ Activity UDP bind failed on {UDP_IP}:{UDP_PORT}: {e}", "err")
        return

    wlog(f"📡 Activity UDP listening on {UDP_IP}:{UDP_PORT}", "info")

    baseline_acc_mag = None
    pre_buffer = deque(maxlen=PRE_BUFFER_SIZE)
    is_recording = False
    record_start_time = None
    record_rows = []
    last_status_print = 0
    last_fall_time = 0
    last_non_idle_label = None
    last_non_idle_confidence = None
    last_non_idle_time = 0

    update_activity_state(status="idle", label="idle", is_recording=False)

    while True:
        try:
            data, _addr = sock.recvfrom(1024)
        except socket.timeout:
            continue
        except Exception as e:
            update_activity_state(status="error", error=f"UDP receive failed: {e}")
            wlog(f"❌ Activity UDP receive failed: {e}", "err")
            time.sleep(1)
            continue

        sample = parse_udp_line(data.decode(errors="ignore").strip())

        if sample is None:
            continue

        current_time = time.time()
        acc_mag = sample["acc_mag"]
        gyro_mag = sample["gyro_mag"]

        update_activity_state(
            acc_mag=acc_mag,
            gyro_mag=gyro_mag,
            baseline_acc_mag=baseline_acc_mag,
            last_sample_time=current_time,
        )

        if current_time - last_fall_time < FALL_HOLD_TIME:
            update_activity_state(status="fall_cooldown", label="fall")
            continue

        if not is_recording:
            pre_buffer.append(sample)

            motion = is_motion_detected(
                acc_mag=acc_mag,
                gyro_mag=gyro_mag,
                baseline_acc_mag=baseline_acc_mag,
            )

            if not motion:
                baseline_acc_mag = update_baseline(baseline_acc_mag, acc_mag)

                if (
                    last_non_idle_label is not None
                    and current_time - last_non_idle_time < ACTIVITY_HOLD_TIME
                ):
                    update_activity_state(
                        status="detected_hold",
                        label=last_non_idle_label,
                        confidence=last_non_idle_confidence,
                        is_recording=False,
                        sample_count=0,
                        baseline_acc_mag=baseline_acc_mag,
                    )
                    continue

                update_activity_state(
                    status="idle",
                    label="idle",
                    confidence=None,
                    is_recording=False,
                    sample_count=0,
                    baseline_acc_mag=baseline_acc_mag,
                )

                if current_time - last_status_print >= 2.0:
                    wlog(
                        f"Activity idle | acc={acc_mag:.3f} | baseline={baseline_acc_mag:.3f} | gyro={gyro_mag:.3f}",
                        "info",
                    )
                    last_status_print = current_time

                continue

            wlog("🎯 Activity motion detected. Collecting prediction window...", "info")
            is_recording = True
            record_start_time = current_time
            record_rows = list(pre_buffer)
            record_rows.append(sample)
            update_activity_state(
                status="recording",
                label="motion",
                confidence=None,
                is_recording=True,
                sample_count=len(record_rows),
            )
            continue

        record_rows.append(sample)
        elapsed = current_time - record_start_time
        update_activity_state(
            status="recording",
            is_recording=True,
            sample_count=len(record_rows),
        )

        if elapsed < RECORD_SECONDS:
            if current_time - last_status_print >= 1.0:
                remaining = RECORD_SECONDS - elapsed
                wlog(f"Activity collecting... {remaining:.1f}s left", "info")
                last_status_print = current_time
            continue

        try:
            features = extract_activity_features(record_rows)
            df_features = pd.DataFrame([features])
            df_features = df_features[feature_columns]

            prediction = str(model.predict(df_features)[0])
            confidence = None

            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(df_features)[0]
                    confidence = float(proba.max())
                except Exception:
                    confidence = None

            if prediction == "fall":
                last_fall_time = current_time
                wlog(f"🚨 FALL DETECTED | confidence={confidence}", "err")
            else:
                last_non_idle_label = prediction
                last_non_idle_confidence = confidence
                last_non_idle_time = current_time
                conf_text = "" if confidence is None else f" | confidence={confidence:.2f}"
                wlog(f"Activity detected: {prediction}{conf_text}", "info")

            update_activity_state(
                status="detected",
                label=prediction,
                confidence=confidence,
                is_recording=False,
                sample_count=len(record_rows),
                last_event_time=current_time,
                error=None,
            )
        except Exception as e:
            update_activity_state(status="error", error=f"Predict failed: {e}")
            wlog(f"❌ Activity prediction failed: {e}", "err")

        is_recording = False
        record_start_time = None
        record_rows = []
        pre_buffer.clear()
        baseline_acc_mag = acc_mag


# ─────────────────────────────────────────────────────────────
# FLASK WEB SERVER
# ─────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/favicon.ico")
@app.route("/apple-touch-icon.png")
@app.route("/apple-touch-icon-precomposed.png")
def browser_icon_probe():
    return Response(status=204)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BLE RSSI Trilateration</title>

<style>
  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  body {
    background: #070b16;
    color: #e0e6f0;
    font-family: 'Courier New', monospace;
    min-height: 100vh;
    padding: 20px;
  }

  h1 {
    color: #00e5ff;
    font-size: 20px;
    letter-spacing: 3px;
    margin-bottom: 4px;
  }

  .subtitle {
    color: #556;
    font-size: 11px;
    margin-bottom: 20px;
  }

  .layout {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    align-items: flex-start;
  }

  canvas {
    background: #0a0e1a;
    border: 1px solid #1a2035;
    border-radius: 12px;
    display: block;
    width: min(760px, 100%);
    height: auto;
  }

  .right-panel {
    flex: 1;
    min-width: 280px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  .section-label {
    font-size: 10px;
    color: #556;
    letter-spacing: 2px;
    font-weight: bold;
    margin-bottom: 8px;
  }

  .card {
    background: #0d1526;
    border: 1px solid #1a2035;
    border-radius: 10px;
    padding: 14px 16px;
  }

  .card.node1 {
    border-left: 3px solid #00e5ff;
  }

  .card.node2 {
    border-left: 3px solid #ff6d3a;
  }

  .card.node3 {
    border-left: 3px solid #a8ff3e;
  }

  .card.tag {
    border-left: 3px solid #ffe033;
    border-color: #ffe03344;
  }

  .node-name {
    font-weight: bold;
    font-size: 13px;
    margin-bottom: 8px;
  }

  .node1 .node-name {
    color: #00e5ff;
  }

  .node2 .node-name {
    color: #ff6d3a;
  }

  .node3 .node-name {
    color: #a8ff3e;
  }

  .tag .node-name {
    color: #ffe033;
  }

  .row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
    font-size: 11px;
    gap: 12px;
  }

  .row .lbl {
    color: #778;
  }

  .row .val {
    color: #fff;
    text-align: right;
  }

  .row .val.hi {
    font-weight: bold;
  }

  .row .val.node1 {
    color: #00e5ff;
  }

  .row .val.node2 {
    color: #ff6d3a;
  }

  .row .val.node3 {
    color: #a8ff3e;
  }

  .row .val.tag {
    color: #ffe033;
  }

  .waiting {
    color: #667;
    font-size: 12px;
  }

  .status-bar {
    display: flex;
    gap: 10px;
    align-items: center;
    margin-bottom: 20px;
    flex-wrap: wrap;
  }

  .pill {
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: bold;
    border: 1px solid #1a2035;
    color: #667;
    background: #0d1526;
  }

  .pill.on {
    border-color: #00e5ff;
    color: #00e5ff;
    background: #00e5ff11;
  }

  .pill.warn {
    border-color: #f90;
    color: #f90;
    background: #ff990011;
  }

  .pill.err {
    border-color: #f55;
    color: #f55;
    background: #ff555511;
  }

  .log-box {
    background: #050810;
    border: 1px solid #1a2035;
    border-radius: 10px;
    padding: 10px 14px;
    height: 160px;
    overflow-y: auto;
    font-size: 11px;
    line-height: 1.35;
  }

  .log-box div {
    margin-bottom: 2px;
  }

  .footer {
    margin-top: 18px;
    color: #334;
    font-size: 10px;
    text-align: center;
  }
</style>
</head>

<body>
<h1>BLE RSSI TRILATERATION</h1>
<div class="subtitle">Nonlinear calibration · Kalman filter · Real-time position · HTTP polling</div>

<div class="status-bar">
  <div class="pill warn" id="pill-server">SERVER ...</div>
  <div class="pill warn" id="pill-mqtt">MQTT ...</div>
  <div class="pill warn" id="pill-nodes">WAITING NODES</div>
  <div class="pill warn" id="pill-ble">BLE ...</div>
  <div class="pill warn" id="pill-activity">ACTIVITY ...</div>
</div>

<div class="layout">
  <canvas id="room" width="760" height="780"></canvas>

  <div class="right-panel">
    <div>
      <div class="section-label">ANCHOR NODES</div>

      <div style="display:flex; flex-direction:column; gap:10px;">
        <div class="card node1">
          <div class="node-name">node1</div>
          <div id="info-node1" class="waiting">waiting…</div>
        </div>

        <div class="card node2">
          <div class="node-name">node2</div>
          <div id="info-node2" class="waiting">waiting…</div>
        </div>

        <div class="card node3">
          <div class="node-name">node3</div>
          <div id="info-node3" class="waiting">waiting…</div>
        </div>
      </div>
    </div>

    <div>
      <div class="section-label">TAG POSITION</div>

      <div class="card tag">
        <div class="node-name">TAG</div>
        <div id="info-tag" class="waiting">Waiting for all node RSSI values…</div>
      </div>
    </div>

    <div>
      <div class="section-label">ACTIVITY</div>

      <div class="card tag">
        <div class="node-name">MOTION</div>
        <div id="info-activity" class="waiting">Waiting for UDP data…</div>
      </div>
    </div>

    <div>
      <div class="section-label">ROOM CONFIG</div>

      <div class="card">
        <div class="row">
          <span class="lbl">Room W</span>
          <span class="val">{{ room_w }} m</span>
        </div>

        <div class="row">
          <span class="lbl">Room H</span>
          <span class="val">{{ room_h }} m</span>
        </div>

        <hr style="border-color:#1a2035; margin:8px 0;">

        {% for node, pos in node_pos.items() %}
        <div class="row">
          <span class="lbl">{{ node }}</span>
          <span class="val {{ node }}">({{ "%.3f"|format(pos[0]) }}, {{ "%.3f"|format(pos[1]) }})</span>
        </div>
        {% endfor %}

        <hr style="border-color:#1a2035; margin:8px 0;">

        <div class="row">
          <span class="lbl">FLIP_X</span>
          <span class="val">{{ flip_x }}</span>
        </div>

        <div class="row">
          <span class="lbl">FLIP_Y</span>
          <span class="val">{{ flip_y }}</span>
        </div>

        <div class="row">
          <span class="lbl">Polling</span>
          <span class="val">{{ poll_ms }} ms</span>
        </div>

        <div class="row">
          <span class="lbl">Web Port</span>
          <span class="val">{{ web_port }}</span>
        </div>

        <div class="row">
          <span class="lbl">UDP Port</span>
          <span class="val">{{ udp_port }}</span>
        </div>

        <div class="row">
          <span class="lbl">Activity Hold</span>
          <span class="val">{{ activity_hold }} s</span>
        </div>
      </div>
    </div>

    <div>
      <div class="section-label">SYSTEM LOG</div>
      <div class="log-box" id="logbox"></div>
    </div>
  </div>
</div>

<div class="footer">BLE RSSI Trilateration Dashboard · Flask / MQTT / BLE / Polling</div>

<script>
const ROOM_W = {{ room_w }};
const ROOM_H = {{ room_h }};
const NODE_POS = {{ node_pos_json|safe }};
const DIST_TABLE = {{ dist_table_json|safe }};
const CURVE_EXP = {{ curve_exp_json|safe }};
const POLL_MS = {{ poll_ms }};

const NODE_COLORS = {
  node1: "#00e5ff",
  node2: "#ff6d3a",
  node3: "#a8ff3e",
};

const ACTIVITY_COLORS = {
  idle: "#667788",
  motion: "#ffe033",
  recording: "#ffe033",
  walking: "#00e5ff",
  fall: "#ff5555",
  stand_to_sit: "#a8ff3e",
  sit_to_stand: "#a8ff3e",
  bending: "#ff9900",
};

function activityColor(activity) {
  const label = activity?.label || activity?.status || "idle";
  return ACTIVITY_COLORS[label] || ACTIVITY_COLORS[activity?.status] || "#ffe033";
}

function activityText(activity) {
  if (!activity) {
    return "waiting";
  }

  if (activity.status === "disabled" || activity.status === "error") {
    return activity.status;
  }

  if (activity.is_recording) {
    return "recording";
  }

  return activity.label || activity.status || "idle";
}

const canvas = document.getElementById("room");
const ctx = canvas.getContext("2d");

let trailData = [];
let logLines = [];
let lastLogId = 0;

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function nonlinearDist(node, rssi) {
  if (rssi === null || rssi === undefined) {
    return null;
  }

  const table = DIST_TABLE[node];
  const exp = CURVE_EXP[node] ?? 1.0;

  if (rssi >= table[0][0]) {
    return table[0][1];
  }

  if (rssi <= table[table.length - 1][0]) {
    return table[table.length - 1][1];
  }

  for (let i = 0; i < table.length - 1; i++) {
    const [r1, d1] = table[i];
    const [r2, d2] = table[i + 1];

    if (r1 >= rssi && rssi >= r2) {
      const ratio = (r1 - rssi) / (r1 - r2);
      return Math.max(0.1, d1 + Math.pow(ratio, exp) * (d2 - d1));
    }
  }

  return table[table.length - 1][1];
}

function drawRoom(data) {
  const W = canvas.width;
  const H = canvas.height;
  const PAD = 50;

  const scaleX = (W - PAD * 2) / ROOM_W;
  const scaleY = (H - PAD * 2) / ROOM_H;

  const toScreen = (rx, ry) => {
    return [
      PAD + rx * scaleX,
      H - PAD - ry * scaleY,
    ];
  };

  ctx.clearRect(0, 0, W, H);

  const grad = ctx.createLinearGradient(0, 0, W, H);
  grad.addColorStop(0, "#0a0e1a");
  grad.addColorStop(1, "#0d1526");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, H);

  ctx.strokeStyle = "rgba(255,255,255,0.05)";
  ctx.lineWidth = 1;

  for (let x = 0; x <= ROOM_W; x += 0.5) {
    const [sx] = toScreen(x, 0);
    ctx.beginPath();
    ctx.moveTo(sx, PAD);
    ctx.lineTo(sx, H - PAD);
    ctx.stroke();
  }

  for (let y = 0; y <= ROOM_H; y += 0.5) {
    const [, sy] = toScreen(0, y);
    ctx.beginPath();
    ctx.moveTo(PAD, sy);
    ctx.lineTo(W - PAD, sy);
    ctx.stroke();
  }

  const [bx1, by1] = toScreen(0, 0);
  const [bx2, by2] = toScreen(ROOM_W, ROOM_H);

  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.lineWidth = 2;
  ctx.strokeRect(bx1, by2, bx2 - bx1, by1 - by2);

  ctx.fillStyle = "rgba(255,255,255,0.35)";
  ctx.font = "10px 'Courier New'";

  for (let x = 0; x <= ROOM_W; x += 1) {
    const [sx, sy] = toScreen(x, 0);
    ctx.fillText(x + "m", sx - 8, sy + 16);
  }

  for (let y = 1; y <= ROOM_H; y += 1) {
    const [sx, sy] = toScreen(0, y);
    ctx.fillText(y + "m", sx - 34, sy + 4);
  }

  ctx.fillStyle = "#00e5ff";
  ctx.font = "bold 12px 'Courier New'";
  ctx.fillText("BLE RSSI Trilateration - Nonlinear Calibration", PAD, 22);

  for (const [node, [rx, ry]] of Object.entries(NODE_POS)) {
    const [cx, cy] = toScreen(rx, ry);
    const color = NODE_COLORS[node];
    const nodeData = data ? data.nodes[node] : null;

    if (nodeData && nodeData.rssi_filtered !== null) {
      const dist = nonlinearDist(node, nodeData.rssi_filtered);
      const radius = dist * scaleX;

      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.strokeStyle = color + "44";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    const grd = ctx.createRadialGradient(cx, cy, 0, cx, cy, 28);
    grd.addColorStop(0, color + "55");
    grd.addColorStop(1, "transparent");

    ctx.fillStyle = grd;
    ctx.beginPath();
    ctx.arc(cx, cy, 28, 0, Math.PI * 2);
    ctx.fill();

    ctx.save();
    ctx.translate(cx, cy);

    ctx.beginPath();
    ctx.moveTo(0, -12);
    ctx.lineTo(10, 8);
    ctx.lineTo(-10, 8);
    ctx.closePath();

    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 12;
    ctx.fill();

    ctx.restore();

    ctx.fillStyle = "#fff";
    ctx.font = "bold 11px 'Courier New'";
    ctx.fillText(node, cx + 14, cy - 8);

    if (nodeData && nodeData.rssi_raw !== null) {
      const dist = nonlinearDist(node, nodeData.rssi_filtered);

      ctx.fillStyle = color;
      ctx.font = "10px 'Courier New'";
      ctx.fillText("raw: " + nodeData.rssi_raw.toFixed(1) + " dBm", cx + 14, cy + 6);
      ctx.fillText("kalman: " + nodeData.rssi_filtered.toFixed(2) + " dBm", cx + 14, cy + 18);
      ctx.fillText("dist: " + dist.toFixed(2) + " m", cx + 14, cy + 30);

      if (nodeData.age_sec !== null) {
        ctx.fillStyle = "#778";
        ctx.fillText("age: " + nodeData.age_sec.toFixed(1) + "s", cx + 14, cy + 42);
      }
    } else {
      ctx.fillStyle = "#667";
      ctx.font = "10px 'Courier New'";
      ctx.fillText("waiting", cx + 14, cy + 6);
    }
  }

  if (trailData.length > 1) {
    ctx.beginPath();

    const [tx0, ty0] = toScreen(trailData[0][0], trailData[0][1]);
    ctx.moveTo(tx0, ty0);

    for (let i = 1; i < trailData.length; i++) {
      const [tx, ty] = toScreen(trailData[i][0], trailData[i][1]);
      ctx.lineTo(tx, ty);
    }

    ctx.strokeStyle = "rgba(255,220,50,0.45)";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  if (data && data.position.x !== null) {
    const {x, y, x_raw, y_raw} = data.position;

    const [px, py] = toScreen(x, y);
    const [prx, pry] = toScreen(x_raw, y_raw);

    ctx.beginPath();
    ctx.moveTo(prx - 7, pry - 7);
    ctx.lineTo(prx + 7, pry + 7);
    ctx.moveTo(prx + 7, pry - 7);
    ctx.lineTo(prx - 7, pry + 7);
    ctx.strokeStyle = "rgba(255,255,255,0.4)";
    ctx.lineWidth = 2;
    ctx.stroke();

    const tg = ctx.createRadialGradient(px, py, 0, px, py, 22);
    tg.addColorStop(0, "rgba(255,220,50,0.5)");
    tg.addColorStop(1, "transparent");

    ctx.fillStyle = tg;
    ctx.beginPath();
    ctx.arc(px, py, 22, 0, Math.PI * 2);
    ctx.fill();

    ctx.beginPath();
    ctx.arc(px, py, 9, 0, Math.PI * 2);
    ctx.fillStyle = "#ffe033";
    ctx.shadowColor = "#ffe033";
    ctx.shadowBlur = 20;
    ctx.fill();
    ctx.shadowBlur = 0;

    ctx.fillStyle = "#ffe033";
    ctx.font = "bold 12px 'Courier New'";
    ctx.fillText("TAG", px + 13, py - 6);

    ctx.font = "10px 'Courier New'";
    ctx.fillText("X=" + x.toFixed(2) + "m", px + 13, py + 7);
    ctx.fillText("Y=" + y.toFixed(2) + "m", px + 13, py + 19);

    const activity = data.activity || {};
    const actText = activityText(activity).toUpperCase();
    const actColor = activityColor(activity);

    ctx.fillStyle = "rgba(0,0,0,0.72)";
    ctx.beginPath();

    if (ctx.roundRect) {
      ctx.roundRect(px + 13, py + 28, 128, 24, 6);
    } else {
      ctx.rect(px + 13, py + 28, 128, 24);
    }

    ctx.fill();
    ctx.strokeStyle = actColor;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = actColor;
    ctx.font = "bold 11px 'Courier New'";
    ctx.fillText(actText, px + 22, py + 44);

    ctx.fillStyle = "rgba(0,0,0,0.5)";
    ctx.beginPath();

    if (ctx.roundRect) {
      ctx.roundRect(PAD, H - PAD - 80, 300, 74, 6);
    } else {
      ctx.rect(PAD, H - PAD - 80, 300, 74);
    }

    ctx.fill();

    ctx.fillStyle = "#ffe033";
    ctx.font = "11px 'Courier New'";
    ctx.fillText("Filtered: X=" + x.toFixed(3) + "m, Y=" + y.toFixed(3) + "m", PAD + 10, H - PAD - 58);

    ctx.fillStyle = "#aaa";
    ctx.fillText("Raw:      X=" + x_raw.toFixed(3) + "m, Y=" + y_raw.toFixed(3) + "m", PAD + 10, H - PAD - 42);

    ctx.fillStyle = "#667";
    ctx.fillText("FLIP_X={{ flip_x }}, FLIP_Y={{ flip_y }}", PAD + 10, H - PAD - 26);
    ctx.fillText("node3 curve=" + CURVE_EXP["node3"], PAD + 10, H - PAD - 10);
  } else {
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    ctx.beginPath();

    if (ctx.roundRect) {
      ctx.roundRect(PAD, H - PAD - 46, 300, 40, 6);
    } else {
      ctx.rect(PAD, H - PAD - 46, 300, 40);
    }

    ctx.fill();

    ctx.fillStyle = "#667";
    ctx.font = "11px 'Courier New'";
    ctx.fillText("Waiting for all node RSSI values...", PAD + 10, H - PAD - 26);
    ctx.fillText("FLIP_X={{ flip_x }}, FLIP_Y={{ flip_y }}", PAD + 10, H - PAD - 10);
  }
}

function updateCards(data) {
  const nodes = ["node1", "node2", "node3"];

  for (const node of nodes) {
    const el = document.getElementById("info-" + node);
    const nd = data.nodes[node];

    if (!nd || nd.rssi_raw === null) {
      el.className = "waiting";
      el.innerHTML = "waiting…";
    } else {
      const dist = nonlinearDist(node, nd.rssi_filtered);
      const ageText = nd.age_sec === null ? "-" : nd.age_sec.toFixed(1) + " s";

      el.className = "";
      el.innerHTML = `
        <div class="row">
          <span class="lbl">raw RSSI</span>
          <span class="val">${nd.rssi_raw.toFixed(1)} dBm</span>
        </div>

        <div class="row">
          <span class="lbl">Kalman</span>
          <span class="val ${node}">${nd.rssi_filtered.toFixed(2)} dBm</span>
        </div>

        <div class="row">
          <span class="lbl">dist</span>
          <span class="val hi">${dist.toFixed(3)} m</span>
        </div>

        <div class="row">
          <span class="lbl">age</span>
          <span class="val">${ageText}</span>
        </div>
      `;
    }
  }

  const tagEl = document.getElementById("info-tag");

  if (data.position.x === null) {
    tagEl.className = "waiting";
    tagEl.innerHTML = "Waiting for all node RSSI values…";
  } else {
    const p = data.position;

    tagEl.className = "";
    tagEl.innerHTML = `
      <div class="row">
        <span class="lbl">X filtered</span>
        <span class="val tag hi">${p.x.toFixed(3)} m</span>
      </div>

      <div class="row">
        <span class="lbl">Y filtered</span>
        <span class="val tag hi">${p.y.toFixed(3)} m</span>
      </div>

      <hr style="border-color:#1a2035; margin:6px 0;">

      <div class="row">
        <span class="lbl">X raw</span>
        <span class="val" style="color:#aaa">${p.x_raw.toFixed(3)} m</span>
      </div>

      <div class="row">
        <span class="lbl">Y raw</span>
        <span class="val" style="color:#aaa">${p.y_raw.toFixed(3)} m</span>
      </div>
    `;
  }

  const activity = data.activity || {};
  const activityEl = document.getElementById("info-activity");
  const actText = activityText(activity);
  const actColor = activityColor(activity);
  const confidenceText = activity.confidence === null || activity.confidence === undefined
    ? "-"
    : (activity.confidence * 100).toFixed(0) + "%";
  const sampleAge = activity.last_sample_age_sec === null || activity.last_sample_age_sec === undefined
    ? "-"
    : activity.last_sample_age_sec.toFixed(1) + " s";

  activityEl.className = "";
  activityEl.innerHTML = `
    <div class="row">
      <span class="lbl">state</span>
      <span class="val hi" style="color:${actColor}">${actText.toUpperCase()}</span>
    </div>

    <div class="row">
      <span class="lbl">status</span>
      <span class="val">${activity.status || "-"}</span>
    </div>

    <div class="row">
      <span class="lbl">confidence</span>
      <span class="val">${confidenceText}</span>
    </div>

    <div class="row">
      <span class="lbl">samples</span>
      <span class="val">${activity.sample_count ?? 0}</span>
    </div>

    <div class="row">
      <span class="lbl">udp age</span>
      <span class="val">${sampleAge}</span>
    </div>
  `;

  const pillServer = document.getElementById("pill-server");
  pillServer.className = "pill on";
  pillServer.textContent = "SERVER LIVE";

  const pillMqtt = document.getElementById("pill-mqtt");

  if (data.mqtt_connected) {
    pillMqtt.className = "pill on";
    pillMqtt.textContent = "MQTT LIVE";
  } else {
    pillMqtt.className = "pill warn";
    pillMqtt.textContent = "MQTT WAITING";
  }

  const allReady = Object.values(data.nodes).every(n => n && n.rssi_raw !== null);
  const pillNodes = document.getElementById("pill-nodes");

  if (allReady) {
    pillNodes.className = "pill on";
    pillNodes.textContent = "ALL NODES READY";
  } else {
    const missing = Object.entries(data.nodes)
      .filter(([_, n]) => !n || n.rssi_raw === null)
      .map(([name, _]) => name)
      .join(", ");

    pillNodes.className = "pill warn";
    pillNodes.textContent = "WAITING: " + missing;
  }

  const pillBle = document.getElementById("pill-ble");
  const n3 = data.nodes["node3"];

  if (n3 && n3.rssi_raw !== null && n3.age_sec !== null && n3.age_sec < 5) {
    pillBle.className = "pill on";
    pillBle.textContent = "BLE LIVE";
  } else if (n3 && n3.rssi_raw !== null) {
    pillBle.className = "pill warn";
    pillBle.textContent = "BLE STALE";
  } else {
    pillBle.className = "pill warn";
    pillBle.textContent = "BLE WAITING";
  }

  const pillActivity = document.getElementById("pill-activity");

  if (activity.status === "disabled" || activity.status === "error") {
    pillActivity.className = "pill err";
    pillActivity.textContent = "ACTIVITY " + activity.status.toUpperCase();
  } else if (activity.label === "fall") {
    pillActivity.className = "pill err";
    pillActivity.textContent = "FALL DETECTED";
  } else if (activity.last_sample_age_sec !== null && activity.last_sample_age_sec !== undefined && activity.last_sample_age_sec < 3) {
    pillActivity.className = "pill on";
    pillActivity.textContent = "ACTIVITY " + actText.toUpperCase();
  } else {
    pillActivity.className = "pill warn";
    pillActivity.textContent = "UDP WAITING";
  }
}

function renderLogs(logs) {
  const newEntries = logs.filter(l => l.id > lastLogId);

  for (const l of newEntries) {
    logLines.push(l);

    if (l.id > lastLogId) {
      lastLogId = l.id;
    }
  }

  if (logLines.length > 200) {
    logLines = logLines.slice(-200);
  }

  const lb = document.getElementById("logbox");

  lb.innerHTML = logLines
  .map(l => `<div style="color:${l.color}">${escapeHtml(l.text)}</div>`)
  .join("");

  lb.scrollTop = lb.scrollHeight;
}

async function poll() {
  try {
    const resp = await fetch("/state?t=" + Date.now(), {
      cache: "no-store",
    });

    if (!resp.ok) {
      throw new Error("HTTP " + resp.status);
    }

    const data = await resp.json();

    trailData = data.trail || [];

    drawRoom(data);
    updateCards(data);
    renderLogs(data.logs || []);
  } catch (e) {
    const pillServer = document.getElementById("pill-server");
    pillServer.className = "pill err";
    pillServer.textContent = "SERVER ERROR";

    const lb = document.getElementById("logbox");
    lb.innerHTML =
  `<div style="color:#f55">Poll error: ${escapeHtml(e.message || e)}</div>` +
  lb.innerHTML;

    console.error("Poll error:", e);
  }

  setTimeout(poll, POLL_MS);
}

drawRoom(null);
poll();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(
        HTML_PAGE,
        room_w=ROOM_W,
        room_h=ROOM_H,
        node_pos=NODE_POS,
        node_pos_json=json.dumps(NODE_POS),
        dist_table_json=json.dumps(DISTANCE_TABLE_BY_NODE),
        curve_exp_json=json.dumps(CURVE_EXP_BY_NODE),
        flip_x=str(FLIP_X),
        flip_y=str(FLIP_Y),
        poll_ms=WEB_POLL_INTERVAL_MS,
        web_port=WEB_PORT,
        udp_port=UDP_PORT,
        activity_hold=ACTIVITY_HOLD_TIME,
    )


@app.route("/state")
def state():
    now = time.time()

    with state_lock:
        nodes_out = {}

        for node in NODE_POS:
            last_seen = node_last_seen[node]

            if last_seen is None:
                age_sec = None
            else:
                age_sec = now - last_seen

            nodes_out[node] = {
                "rssi_raw": raw_rssi_latest[node],
                "rssi_filtered": latest_rssi[node],
                "dist": rssi_to_distance(node, latest_rssi[node]),
                "age_sec": age_sec,
                "last_seen": last_seen,
            }

        pos = dict(latest_position)
        trail = list(position_trail)
        activity = dict(activity_state)

        if activity.get("last_sample_time") is None:
            activity["last_sample_age_sec"] = None
        else:
            activity["last_sample_age_sec"] = now - activity["last_sample_time"]

        if activity.get("last_event_time") is None:
            activity["last_event_age_sec"] = None
        else:
            activity["last_event_age_sec"] = now - activity["last_event_time"]

    with log_lock:
        logs = list(web_logs)

    response = {
        "server_time": now,
        "mqtt_connected": mqtt_connected,
        "nodes": nodes_out,
        "position": pos,
        "activity": activity,
        "trail": trail,
        "logs": logs,
    }

    return Response(
        json.dumps(response, allow_nan=False),
        mimetype="application/json",
    )


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("📍 Active node positions:")

    for node, pos in NODE_POS.items():
        print(f"   {node}: x={pos[0]:.3f}, y={pos[1]:.3f}")

    print("\n📏 Nonlinear RSSI → distance table:")

    for node, table in DISTANCE_TABLE_BY_NODE.items():
        print(f"   {node}: {table}, curve={CURVE_EXP_BY_NODE[node]}")

    mqttc.on_connect = on_connect
    mqttc.on_disconnect = on_disconnect
    mqttc.on_message = on_message

    mqttc.reconnect_delay_set(min_delay=1, max_delay=10)

    wlog(f"MQTT connecting to {MQTT_BROKER}:{MQTT_PORT}", "mqtt")

    try:
        mqttc.connect_async(MQTT_BROKER, MQTT_PORT, 60)
        mqttc.loop_start()
    except Exception as e:
        wlog(f"❌ MQTT start failed: {e}", "err")

    threading.Thread(target=ble_loop, daemon=True).start()
    wlog("BLE scanner thread started. Laptop is used as node3.", "ble")

    threading.Thread(target=activity_loop, daemon=True).start()
    wlog(f"Activity detector thread started. UDP port is {UDP_PORT}.", "info")

    print("\n🌐 Web dashboard:")
    print(f"   Local:   http://localhost:{WEB_PORT}")
    print(f"   Network: http://<YOUR_LAPTOP_IP>:{WEB_PORT}")
    print(f"   Activity UDP: {UDP_IP}:{UDP_PORT}")
    print("\n🚀 BLE Trilateration Web Server started\n")

    app.run(
        host="0.0.0.0",
        port=WEB_PORT,
        debug=False,
        threaded=True,
    )