#!/usr/bin/env python3
"""Minimal script to poll Medoc and print raw + parsed response."""

import sys
from psycopy.config import MedocConfig
from psycopy.medoc import MedocClient

IP = "10.196.94.38"   # change if your Medoc IP differs
PORT = 20121
TIMEOUT = 5.0

config = MedocConfig(medoc_ip=IP, medoc_port=PORT, medoc_timeout=TIMEOUT)
client = MedocClient(config)

try:
    print(f"Connecting to {IP}:{PORT} ...")
    client.connect()
    print("Connected.\n")

    print("Polling status ...")
    status = client.poll_status()
    print("\n--- Parsed status ---")
    for k, v in status.items():
        print(f"  {k}: {v}")

    raw = status.get("raw_bytes")
    if raw:
        print(f"\n--- Raw bytes ({len(raw)} bytes) ---")
        print("  hex:", raw.hex())
        print("  repr:", repr(raw))

except Exception as exc:
    print(f"ERROR: {exc}")
    sys.exit(1)
finally:
    client.disconnect()
    print("\nDisconnected.")
