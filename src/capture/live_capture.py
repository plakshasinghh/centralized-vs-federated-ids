from scapy.all import sniff, IP, TCP, UDP, ICMP, ARP
import math
import time
from collections import defaultdict, deque
import pandas as pd

# ------------------ GLOBAL STATE ----------------------------
active_flows = defaultdict(lambda: {
    'start_time': None,
    'last_packet_time': None,
    'fwd_packets': 0,
    'bwd_packets': 0,
    'fwd_bytes': 0,
    'bwd_bytes': 0,
    'fwd_syn_count': 0,
    'bwd_syn_count': 0,
    'fwd_ack_count': 0,
    'bwd_ack_count': 0,
    'fwd_packet_lengths': deque(),
    'bwd_packet_lengths': deque()
})

# ------------------ HELPER FUNCTIONS -------------------------
def calculate_entropy(data):
    if not data:
        return 0
    entropy = 0
    for x in range(256):
        p_x = float(data.count(x)) / len(data)
        if p_x > 0:
            entropy -= p_x * math.log2(p_x)
    return entropy

def extract_packet_level_features(packet):
    features = {
        "timestamp": packet.time,
        "src_ip": None, "dst_ip": None,
        "protocol": "OTHER", "protocol_num": 0,
        "src_port": 0, "dst_port": 0,
        "length": len(packet),
        "payload_length": 0,
        "ip_version": 0, "ip_ihl": 0, "ip_ttl": 0,
        "ip_flags_DF": 0, "ip_options_count": 0,
        "tcp_flags_SYN": 0, "tcp_flags_ACK": 0, "tcp_flags_FIN": 0,
        "tcp_flags_RST": 0, "tcp_flags_PSH": 0, "tcp_flags_URG": 0,
        "tcp_window_size": 0, "tcp_sequence_number": 0,
        "tcp_acknowledgement_number": 0, "tcp_data_offset": 0,
        "tcp_options_count": 0,
        "udp_checksum": 0,
        "icmp_type": -1, "icmp_code": -1,
        "payload_entropy": 0.0,
        "has_http_sig": 0, "has_tls_sig": 0,
    }

    if IP in packet:
        features["src_ip"] = packet[IP].src
        features["dst_ip"] = packet[IP].dst
        features["ip_version"] = packet[IP].version
        features["ip_ihl"] = packet[IP].ihl
        features["ip_ttl"] = packet[IP].ttl
        features["ip_flags_DF"] = 1 if 'DF' in str(packet[IP].flags) else 0
        features["ip_options_count"] = len(packet[IP].options) if packet[IP].options else 0
        features["payload_length"] = len(packet[IP].payload)

        if TCP in packet:
            features["protocol"] = "TCP"
            features["protocol_num"] = 6
            features["src_port"] = packet[TCP].sport
            features["dst_port"] = packet[TCP].dport
            flags = str(packet[TCP].flags)
            features["tcp_flags_SYN"] = int('S' in flags)
            features["tcp_flags_ACK"] = int('A' in flags)
            if packet[TCP].payload:
                features["payload_length"] = len(packet[TCP].payload)

        elif UDP in packet:
            features["protocol"] = "UDP"
            features["protocol_num"] = 17
            features["src_port"] = packet[UDP].sport
            features["dst_port"] = packet[UDP].dport
            if packet[UDP].payload:
                features["payload_length"] = len(packet[UDP].payload)

        elif ICMP in packet:
            features["protocol"] = "ICMP"
            features["protocol_num"] = 1
            features["icmp_type"] = packet[ICMP].type
            features["icmp_code"] = packet[ICMP].code
            if packet[ICMP].payload:
                features["payload_length"] = len(packet[ICMP].payload)

        raw_payload = bytes(packet[IP].payload)
        if raw_payload:
            features["payload_entropy"] = calculate_entropy(raw_payload)

    elif ARP in packet:
        features["protocol"] = "ARP"
        features["protocol_num"] = 2

    return features

def update_flow_features(features, packet):
    if not (features["src_ip"] and features["dst_ip"] and features["protocol_num"] in [1,6,17]):
        return features

    direction = "fwd" if features["src_ip"] < features["dst_ip"] else "bwd"
    flow_key = (features["src_ip"], features["dst_ip"], features["src_port"], features["dst_port"], features["protocol_num"]) \
        if direction=="fwd" else (features["dst_ip"], features["src_ip"], features["dst_port"], features["src_port"], features["protocol_num"])

    current_flow = active_flows[flow_key]
    if current_flow['start_time'] is None:
        current_flow['start_time'] = packet.time
    current_flow['last_packet_time'] = packet.time

    packet_len = features["length"]

    if direction=="fwd":
        current_flow['fwd_packets'] +=1
        current_flow['fwd_bytes'] += packet_len
        current_flow['fwd_packet_lengths'].append(packet_len)
        if features.get("tcp_flags_SYN"): current_flow['fwd_syn_count'] +=1
        if features.get("tcp_flags_ACK"): current_flow['fwd_ack_count'] +=1
    else:
        current_flow['bwd_packets'] +=1
        current_flow['bwd_bytes'] += packet_len
        current_flow['bwd_packet_lengths'].append(packet_len)
        if features.get("tcp_flags_SYN"): current_flow['bwd_syn_count'] +=1
        if features.get("tcp_flags_ACK"): current_flow['bwd_ack_count'] +=1

    features["flow_duration"] = current_flow['last_packet_time'] - current_flow['start_time']
    features["total_fwd_packets"] = current_flow['fwd_packets']
    features["total_bwd_packets"] = current_flow['bwd_packets']
    features["total_fwd_bytes"] = current_flow['fwd_bytes']
    features["total_bwd_bytes"] = current_flow['bwd_bytes']

    return features

def extract_features(packet):
    pkt_features = extract_packet_level_features(packet)
    pkt_features = update_flow_features(pkt_features, packet)
    return pkt_features

# ------------------ MAIN EXECUTION ---------------------------
def live_sniff(duration=30):
    print(f"🔍 Sniffing live traffic for {duration} seconds...")
    captured = []

    def process(pkt):
        features = extract_features(pkt)
        captured.append(features)
        print(features)

    sniff(prn=process, filter="ip", store=0, timeout=duration)
    df = pd.DataFrame(captured)
    df.to_csv("live_packet_features.csv", index=False)
    print("✅ Live sniffing features saved to live_packet_features.csv")

if __name__=="__main__":
    # Just do live sniffing; no need for .pcap file
    live_sniff(duration=15)  # capture 15 seconds of live traffic

ML.PY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
import joblib
import warnings
warnings.filterwarnings("ignore")

# ===============================
# STEP 1: LOAD DATA
# ===============================
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded dataset with shape: {df.shape}")
    print("🧾 Columns:", list(df.columns))
    return df


# ===============================
# STEP 2: PREPROCESSING
# ===============================
def preprocess_data(df, target_column):
    df = df.dropna()

    # Encode categorical columns (IP, protocol, etc.)
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return train_test_split(X, y, test_size=0.2, random_state=42)


# ===============================
# STEP 3: MODEL DEFINITIONS
# ===============================
def build_linear_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])

def build_ensemble_model():
    return RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

def build_clustering_model(n_clusters=3):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=42))
    ])


# ===============================
# STEP 4: TRAIN + EVALUATE
# ===============================
def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = {}

    # --- Linear Model ---
    print("\n🔹 Training Linear Model (Ridge Regression)")
    linear_model = build_linear_model()
    linear_model.fit(X_train, y_train)
    y_pred_lin = linear_model.predict(X_test)
    results["linear"] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_lin)),
        "R2": r2_score(y_test, y_pred_lin)
    }

    # --- Ensemble Model ---
    print("\n🔹 Training Ensemble Model (Random Forest)")
    ensemble_model = build_ensemble_model()
    ensemble_model.fit(X_train, y_train)
    y_pred_rf = ensemble_model.predict(X_test)
    results["ensemble"] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        "R2": r2_score(y_test, y_pred_rf)
    }

    # --- Clustering Model ---
    print("\n🔹 Running Clustering (KMeans)")
    clustering_model = build_clustering_model(n_clusters=3)
    X_scaled = clustering_model.named_steps['scaler'].fit_transform(X_train)
    clustering_model.named_steps['kmeans'].fit(X_scaled)
    cluster_labels = clustering_model.named_steps['kmeans'].labels_
    results["clustering"] = {
        "Silhouette Score": silhouette_score(X_scaled, cluster_labels)
    }

    # --- Save Models ---
    joblib.dump(linear_model, "ridge_model.pkl")
    joblib.dump(ensemble_model, "random_forest_model.pkl")
    joblib.dump(clustering_model, "kmeans_model.pkl")

    print("\n✅ Models saved as .pkl files.")
    return results, y_pred_lin, y_pred_rf


# ===============================
# STEP 5: VISUALIZATIONS
# ===============================
def visualize_data(df, target_col):
    print("\n📊 Generating Data Visualizations...")

    # 1️⃣ Correlation Heatmap
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # 2️⃣ Distribution of Target
    plt.figure(figsize=(8,5))
    sns.histplot(df[target_col], bins=30, kde=True, color='teal')
    plt.title(f"Distribution of Target: {target_col}")
    plt.tight_layout()
    plt.show()

    # 3️⃣ Pair Plot (subset)
    subset_cols = df.select_dtypes(include=[np.number]).columns[:5].tolist()
    sns.pairplot(df[subset_cols])
    plt.suptitle("Pairwise Feature Relationships (subset)", y=1.02)
    plt.show()


def visualize_model_results(y_test, y_pred_lin, y_pred_rf):
    # 4️⃣ Actual vs Predicted (Linear)
    plt.figure(figsize=(10,5))
    plt.scatter(y_test, y_pred_lin, alpha=0.6, label='Ridge Predictions', color='blue')
    plt.scatter(y_test, y_pred_rf, alpha=0.6, label='RandomForest Predictions', color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_feature_importance(model, X_train):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10,5))
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), X_train.columns[indices], rotation=90)
        plt.title("Feature Importance (Random Forest)")
        plt.tight_layout()
        plt.show()


def visualize_clusters(X_train, clustering_model):
    # 5️⃣ KMeans Cluster Visualization
    X_scaled = clustering_model.named_steps['scaler'].transform(X_train)
    labels = clustering_model.named_steps['kmeans'].labels_
    plt.figure(figsize=(8,6))
    plt.scatter(X_scaled[:,0], X_scaled[:,1], c=labels, cmap='viridis', s=20)
    plt.title("KMeans Clusters (first two features)")
    plt.tight_layout()
    plt.show()


# ===============================
# MAIN EXECUTION
# ===============================
if __name__ == "__main__":
    df = load_dataset("./live_packet_features.csv")

    # Choose your target column (adjust based on your dataset)
    target_col = "payload_length"
    X_train, X_test, y_train, y_test = preprocess_data(df, target_col)

    visualize_data(df, target_col)

    results, y_pred_lin, y_pred_rf = train_and_evaluate(X_train, X_test, y_train, y_test)
    print("\n📈 Evaluation Results:")
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics}")

    visualize_model_results(y_test, y_pred_lin, y_pred_rf)
    visualize_feature_importance(build_ensemble_model().fit(X_train, y_train), X_train)
    visualize_clusters(X_train, build_clustering_model(n_clusters=3).fit(X_train))












MAIN2.PY
from fastapi import FastAPI
import logging
from scapy.all import sniff, IP, TCP, UDP, ICMP, ARP
import threading
import csv
import math
from collections import defaultdict, deque
import time

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

# ------------------ GLOBAL STATE ----------------------------
sniffing_active = False
last_packet_features = None  # store last captured packet's features

active_flows = defaultdict(lambda: {
    'start_time': None,
    'last_packet_time': None,
    'fwd_packets': 0,
    'bwd_packets': 0,
    'fwd_bytes': 0,
    'bwd_bytes': 0,
    'fwd_syn_count': 0,
    'bwd_syn_count': 0,
    'fwd_ack_count': 0,
    'bwd_ack_count': 0,
    'fwd_packet_lengths': deque(),
    'bwd_packet_lengths': deque()
})

dataset_file = "packets_dataset.csv"

# ------------------ CSV INIT -------------------------------
header_fields = [
    "timestamp", "src_ip", "dst_ip", "protocol", "protocol_num", "src_port", "dst_port",
    "length", "payload_length", "ip_version", "ip_ihl", "ip_ttl", "ip_flags_DF", "ip_options_count",
    "tcp_flags_SYN", "tcp_flags_ACK", "tcp_flags_FIN", "tcp_flags_RST", "tcp_flags_PSH", "tcp_flags_URG",
    "tcp_window_size", "tcp_sequence_number", "tcp_acknowledgement_number", "tcp_data_offset", "tcp_options_count",
    "udp_checksum", "icmp_type", "icmp_code",
    "payload_entropy", "has_http_sig", "has_tls_sig",
    "flow_duration", "total_fwd_packets", "total_bwd_packets", "total_fwd_bytes", "total_bwd_bytes",
    "fwd_packets_per_sec", "bwd_packets_per_sec", "mean_fwd_packet_len", "mean_bwd_packet_len",
    "fwd_syn_count", "bwd_syn_count", "fwd_ack_count", "bwd_ack_count"
]

# Initialize CSV
with open(dataset_file, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header_fields)
    writer.writeheader()

# ------------------ HELPER FUNCTIONS ------------------------
def calculate_entropy(data):
    if not data:
        return 0
    entropy = 0
    for x in range(256):
        p_x = float(data.count(x)) / len(data)
        if p_x > 0:
            entropy -= p_x * math.log2(p_x)
    return entropy

def extract_packet_level_features(packet):
    features = {k: 0 for k in header_fields}
    features["timestamp"] = packet.time

    if IP in packet:
        ip = packet[IP]
        features.update({
            "src_ip": ip.src,
            "dst_ip": ip.dst,
            "protocol_num": ip.proto,
            "ip_version": ip.version,
            "ip_ihl": ip.ihl,
            "ip_ttl": ip.ttl,
            "ip_flags_DF": 1 if 'DF' in str(ip.flags) else 0,
            "ip_options_count": len(ip.options) if ip.options else 0,
            "length": len(packet),
            "payload_length": len(ip.payload) if ip.payload else 0
        })

        # TCP Layer
        if TCP in packet:
            tcp = packet[TCP]
            features.update({
                "protocol": "TCP",
                "src_port": tcp.sport,
                "dst_port": tcp.dport,
                "tcp_flags_SYN": int('S' in str(tcp.flags)),
                "tcp_flags_ACK": int('A' in str(tcp.flags)),
                "tcp_flags_FIN": int('F' in str(tcp.flags)),
                "tcp_flags_RST": int('R' in str(tcp.flags)),
                "tcp_flags_PSH": int('P' in str(tcp.flags)),
                "tcp_flags_URG": int('U' in str(tcp.flags)),
                "tcp_window_size": tcp.window,
                "tcp_sequence_number": tcp.seq,
                "tcp_acknowledgement_number": tcp.ack,
                "tcp_data_offset": tcp.dataofs,
                "tcp_options_count": len(tcp.options) if tcp.options else 0
            })
        # UDP Layer
        elif UDP in packet:
            udp = packet[UDP]
            features.update({
                "protocol": "UDP",
                "src_port": udp.sport,
                "dst_port": udp.dport,
                "udp_checksum": udp.chksum
            })
        # ICMP Layer
        elif ICMP in packet:
            icmp = packet[ICMP]
            features.update({
                "protocol": "ICMP",
                "icmp_type": icmp.type,
                "icmp_code": icmp.code
            })

        # Payload Entropy & Signatures
        raw_payload = bytes(ip.payload)
        if raw_payload:
            features["payload_entropy"] = calculate_entropy(raw_payload)
            payload_str = raw_payload.decode('latin-1', errors='ignore')
            features["has_http_sig"] = int(any(sig in payload_str for sig in ['GET ', 'POST ', 'HTTP/', 'Host:']))
            features["has_tls_sig"] = int(len(raw_payload) > 5 and raw_payload[0] == 0x16 and raw_payload[1] == 0x03)

    elif ARP in packet:
        features["protocol"] = "ARP"
        features["protocol_num"] = 2

    return features

def update_flow_features(features, packet):
    if not (features["src_ip"] and features["dst_ip"] and features["protocol_num"] in [1,6,17]):
        return features

    direction = "fwd" if features["src_ip"] < features["dst_ip"] else "bwd"
    flow_key = (
        features["src_ip"], features["dst_ip"], features["src_port"], features["dst_port"], features["protocol_num"]
    ) if direction=="fwd" else (
        features["dst_ip"], features["src_ip"], features["dst_port"], features["src_port"], features["protocol_num"]
    )

    flow = active_flows[flow_key]
    if flow['start_time'] is None:
        flow['start_time'] = packet.time
    flow['last_packet_time'] = packet.time

    packet_len = features["length"]

    if direction=="fwd":
        flow['fwd_packets'] +=1
        flow['fwd_bytes'] += packet_len
        flow['fwd_packet_lengths'].append(packet_len)
        flow['fwd_syn_count'] += features.get("tcp_flags_SYN",0)
        flow['fwd_ack_count'] += features.get("tcp_flags_ACK",0)
    else:
        flow['bwd_packets'] +=1
        flow['bwd_bytes'] += packet_len
        flow['bwd_packet_lengths'].append(packet_len)
        flow['bwd_syn_count'] += features.get("tcp_flags_SYN",0)
        flow['bwd_ack_count'] += features.get("tcp_flags_ACK",0)

    # Flow statistics
    features["flow_duration"] = flow['last_packet_time'] - flow['start_time']
    features["total_fwd_packets"] = flow['fwd_packets']
    features["total_bwd_packets"] = flow['bwd_packets']
    features["total_fwd_bytes"] = flow['fwd_bytes']
    features["total_bwd_bytes"] = flow['bwd_bytes']
    if features["flow_duration"]>0:
        features["fwd_packets_per_sec"] = flow['fwd_packets']/features["flow_duration"]
        features["bwd_packets_per_sec"] = flow['bwd_packets']/features["flow_duration"]
    if flow['fwd_packet_lengths']:
        features["mean_fwd_packet_len"] = sum(flow['fwd_packet_lengths'])/len(flow['fwd_packet_lengths'])
    if flow['bwd_packet_lengths']:
        features["mean_bwd_packet_len"] = sum(flow['bwd_packet_lengths'])/len(flow['bwd_packet_lengths'])
    features["fwd_syn_count"] = flow['fwd_syn_count']
    features["bwd_syn_count"] = flow['bwd_syn_count']
    features["fwd_ack_count"] = flow['fwd_ack_count']
    features["bwd_ack_count"] = flow['bwd_ack_count']

    return features

def extract_features(packet):
    pkt_features = extract_packet_level_features(packet)
    pkt_features = update_flow_features(pkt_features, packet)
    return pkt_features

# ------------------ PACKET HANDLER ---------------------------
def packet_handler(packet):
    global last_packet_features
    features = extract_features(packet)
    last_packet_features = features
    with open(dataset_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header_fields)
        writer.writerow(features)
    logger.info(f"Captured packet: {features['src_ip']} -> {features['dst_ip']} Proto:{features['protocol']}")

def stop_filter(packet):
    return not sniffing_active

def start_sniffing_thread():
    global sniffing_active
    sniffing_active = True
    sniff(prn=packet_handler, store=0, stop_filter=stop_filter)
    logger.info("Sniffing ended.")

# ------------------ API ENDPOINTS ----------------------------
@app.get("/")
def read_root():
    return {"message": "Packet Monitor API Running"}

@app.get("/start_sniffing")
def start_sniffing_endpoint():
    global sniffing_active
    if not sniffing_active:
        thread = threading.Thread(target=start_sniffing_thread, daemon=True)
        thread.start()
        return {"message": "Packet sniffing started."}
    else:
        return {"message": "Packet sniffing already running."}

@app.get("/stop_sniffing")
def stop_sniffing():
    global sniffing_active
    sniffing_active = False
    return {"message": "Packet sniffing will stop shortly."}

@app.get("/status")
def get_status():
    return {"status": "Monitoring" if sniffing_active else "Not Monitoring"}

@app.get("/last_packet")
def get_last_packet():
    if last_packet_features:
        return last_packet_features
    else:
        return {"summary": "No packet captured yet."}


























MAIN.PY
from fastapi import FastAPI
import logging
from scapy.all import sniff
from scapy.layers.inet import IP, TCP, UDP
import threading
import csv
import time

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

# Global variable to store sniffing state
sniffing_active = False

def packet_callback(packet):
    if IP in packet:
        src = packet[IP].src
        dst = packet[IP].dst
        proto = packet[IP].proto
        logger.info(f"Packet: {src} -> {dst} (Protocol: {proto})")
        logger.info(f"Packet details: {packet.summary()}")

def stop_filter(packet):
    return not sniffing_active  # Stop sniffing if sniffing_active becomes False

def start_sniffing():
    global sniffing_active
    sniffing_active = True
    sniff(prn=packet_callback, store=0, stop_filter=stop_filter)
    logger.info("Sniffing ended.")

@app.get("/last_packet")
def get_last_packet():
    print("last packet function called ")
    global sniffing_active
    sniffing_active = False
    logger.info("Packet sniffing stop requested.")

    last_packet = sniff(count=1)
    # Check if a packet was captured
    if last_packet:
        # Print the summary of the first (and only) packet captured
        print("Summary of the last packet received:")
        print(last_packet[0].summary())
        # print("IP:", last_packet[0][IP])
        # print("Source IP:", last_packet[0][IP].src)
    else:
        print("No packet was captured.")

@app.get("/start_sniffing")
def start_sniffing_endpoint():
    if not sniffing_active:
        thread = threading.Thread(target=start_sniffing, daemon=True)
        thread.start()
        logger.info("Started packet sniffing thread.")
        return {"message": "Packet sniffing started."}
    else:
        return {"message": "Packet sniffing is already running."}

@app.get("/stop_sniffing")
def stop_sniffing():
    global sniffing_active
    sniffing_active = False
    logger.info("Packet sniffing stop requested.")
    return {"message": "Packet sniffing will stop shortly."}

@app.get("/")
def read_root():
    logger.info("Root endpoint was accessed")
    return {"message": "Packet Monitor API Running"}

@app.get("/status")
def get_status():
    return {"status": "Monitoring" if sniffing_active else "Not Monitoring"}

dataset_file = "packets_dataset.csv"

# Initialize CSV file with headers
with open(dataset_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "src_ip", "dst_ip", "protocol", "src_port", "dst_port", "length", "flags", "payload"])

def packet_callback(packet):
    timestamp = time.time()
    src = packet[IP].src if IP in packet else None
    dst = packet[IP].dst if IP in packet else None
    proto = packet[IP].proto if IP in packet else None
    length = len(packet)

    src_port, dst_port, flags = None, None, None
    if TCP in packet:
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport
        flags = packet[TCP].flags
    elif UDP in packet:
        src_port = packet[UDP].sport
        dst_port = packet[UDP].dport

    # Limit payload size (avoid huge binary dumps)
    payload = bytes(packet[IP].payload)[:50] if IP in packet else b""

    # Append row to CSV
    with open(dataset_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, src, dst, proto, src_port, dst_port, length, flags, payload])

    logger.info(f"Captured packet {src}:{src_port} -> {dst}:{dst_port} Proto:{proto} Len:{length}")





COMBINER.PY
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------- CONFIG --------
FOLDER_PATH = "./archive"    # your folder path
OUTPUT_FILE = "final_balanced_dataset.csv"

def find_label_column(columns):
    """Find a column that looks like a label column."""
    for col in columns:
        if 'label' in col.lower().strip():
            return col
    return None

# -------- STEP 1: READ ALL CSVs --------
df_list = []
for file in os.listdir(FOLDER_PATH):
    if file.endswith(".csv"):
        path = os.path.join(FOLDER_PATH, file)
        try:
            temp = pd.read_csv(path)
            label_col = find_label_column(temp.columns)
            if not label_col:
                print(f"⚠️ Skipping {file} (no label column found)")
                continue

            temp.rename(columns={label_col: "Label"}, inplace=True)
            df_list.append(temp)
            print(f"✅ Loaded {file} with shape {temp.shape} (Label col: '{label_col}')")

        except Exception as e:
            print(f"❌ Error reading {file}: {e}")

# -------- STEP 2: COMBINE ALL --------
if not df_list:
    raise ValueError("No CSVs with label column found — please check your files!")

df = pd.concat(df_list, ignore_index=True)
print(f"\nCombined dataset shape: {df.shape}")

# -------- STEP 3: SPLIT BENIGN vs ATTACK --------
benign_df = df[df["Label"].str.strip().str.upper() == "BENIGN"]
attack_df = df[df["Label"].str.strip().str.upper() != "BENIGN"]

print(f"Benign samples: {len(benign_df)}")
print(f"Attack samples: {len(attack_df)}")

# -------- STEP 4: BALANCE DATA --------
min_class_count = min(len(benign_df), len(attack_df))

benign_sampled = benign_df.sample(n=min_class_count, random_state=42)
attack_sampled = attack_df.sample(n=min_class_count, random_state=42)

balanced_df = pd.concat([benign_sampled, attack_sampled], ignore_index=True).sample(frac=1, random_state=42)
print(f"\nFinal balanced dataset shape: {balanced_df.shape}")
print(balanced_df["Label"].value_counts())

# -------- STEP 5: SAVE --------
balanced_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Saved final dataset to {OUTPUT_FILE}")

# -------- STEP 6: PLOT --------
plt.figure(figsize=(8,5))
sns.countplot(x="Label", data=balanced_df, palette="coolwarm")
plt.title("Label Distribution in Final Balanced Dataset")
plt.xlabel("Label")
plt.ylabel("Count")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
