import json
import socket
import time

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5005


class Publisher:
    """Sends small JSON messages over UDP so the vision loop never waits on the frontend."""

    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT, units="px"):
        self.address = (host, port)
        self.units = units
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, message):
        message["units"] = self.units
        message["sent_at"] = time.time()
        self.sock.sendto(json.dumps(message).encode("utf-8"), self.address)

    def send_track(self, x, y, vx, vy):
        self.send({"type": "track", "x": float(x), "y": float(y),
                   "vx": float(vx), "vy": float(vy)})

    def send_landing(self, x_land, time_to_land, frames_used):
        self.send({"type": "landing", "x_land": float(x_land),
                   "time_to_land": float(time_to_land), "frames_used": int(frames_used)})

    def close(self):
        self.sock.close()


if __name__ == "__main__":
    # Sends fake landing predictions so the receiver can be tested without a camera.
    pub = Publisher()
    for i in range(20):
        pub.send_landing(x_land=300 + 5 * i, time_to_land=0.5 - 0.02 * i, frames_used=4 + i)
        print("sent message", i)
        time.sleep(0.1)
    pub.close()
    