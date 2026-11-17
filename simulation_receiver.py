
import json
import socket
import threading
import time
import math


class SimulationReceiver:
    def __init__(self, host="127.0.0.1", port=5006):
        self.host = host
        self.port = port
        self.latest_target = None
        self.last_received = 0
        # Live object tracking data
        self.latest_position = None
        self.last_track_received = 0
        self.running = False
        self.sock = None
        self.thread = None

    def start(self):
        self.sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM
        )

        self.sock.bind((self.host, self.port))
        self.sock.settimeout(0.2)

        self.running = True

        self.thread = threading.Thread(
            target=self.receive,
            daemon=True
        )
        self.thread.start()

    def receive(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(4096)
                message = json.loads(data.decode("utf-8"))

                print("UDP message received:", message, flush=True)

                if message.get("type") == "landing":
                    x = message.get("x_land")

                    if isinstance(x, (int, float)) and math.isfinite(x):
                        self.latest_target = float(x)
                        self.last_received = time.monotonic()


                elif message.get("type") == "track":
                    x = message.get("x")
                    y = message.get("y")

                    if (
                        isinstance(x, (int, float))
                        and isinstance(y, (int, float))
                        and math.isfinite(x)
                        and math.isfinite(y)
                    ):
                        self.latest_position = (
                            float(x),
                            float(y)
                        )
                        self.last_track_received = time.monotonic()


            except socket.timeout:
                continue
            except OSError:
                break
            except (ValueError, TypeError):
                continue

    def stop(self):
        self.running = False

        if self.sock is not None:
            self.sock.close()
            self.sock = None

        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
