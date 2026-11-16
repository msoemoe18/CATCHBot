import json
import socket

from output import DEFAULT_HOST, DEFAULT_PORT

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((DEFAULT_HOST, DEFAULT_PORT))
sock.settimeout(1.0)
print(f"Listening on {DEFAULT_HOST}:{DEFAULT_PORT} (Ctrl+C to stop)")

try:
    while True:
        try:
            data, _ = sock.recvfrom(4096)
        except socket.timeout:
            continue
        print(json.loads(data))
except KeyboardInterrupt:
    print("Stopped")
finally:
    sock.close()