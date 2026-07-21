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
