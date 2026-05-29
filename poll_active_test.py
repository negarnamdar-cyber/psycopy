#!/usr/bin/env python3
"""Poll Medoc while the unified program is running to diagnose incomplete responses."""

import time
from psycopy.config import MedocConfig
from psycopy.medoc import MedocClient

IP = "10.196.94.38"
PORT = 20121
TIMEOUT = 5.0

config = MedocConfig(medoc_ip=IP, medoc_port=PORT, medoc_timeout=TIMEOUT)
client = MedocClient(config)

try:
    print(f"Connecting to {IP}:{PORT} ...")
    client.connect()
    print("Connected.\n")

    print("Starting unified program (192) ...")
    client.send_unified_program()
    print("Program started.\n")

    print("Polling every 5 seconds for 30 seconds ...\n")
    for i in range(6):
        print(f"--- Poll {i + 1} ---")
        try:
            status = client.poll_status()
            raw = status.get("raw_bytes", b"")
            print(f"  raw_len: {len(raw)}")
            print(f"  raw_hex: {raw.hex()}")
            print(f"  temp:    {status.get('temperature_celsius')}")
            print(f"  state:   {status.get('device_state')}")
            print(f"  test:    {status.get('test_state')}")
            print(f"  rc:      {status.get('response_code')}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
        time.sleep(5)

    print("\nStopping program ...")
    client.stop_unified_program()
    print("Stopped.")

except Exception as exc:
    print(f"FATAL ERROR: {exc}")
finally:
    client.disconnect()
    print("\nDisconnected.")
