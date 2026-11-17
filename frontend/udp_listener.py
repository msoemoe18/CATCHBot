
import json
import socket
import threading
from queue import Queue, Full


class UDPListener:
    def __init__(self, host="127.0.0.1", port=5005):
        self.host = host
        self.port = port
        self.messages = Queue(maxsize=100)
        self.running = False
        self.sock = None
        self.thread = None

    def start(self):
        if self.running:
            return

        self.sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM
        )

        self.sock.bind((self.host, self.port))
        self.sock.settimeout(0.2)

        self.running = True

        self.thread = threading.Thread(
            target=self._receive,
            daemon=True
        )
        self.thread.start()

    def _receive(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(4096)
                message = json.loads(
                    data.decode("utf-8")
                )

                try:
                    self.messages.put_nowait(message)
                except Full:
                    # Discard old data when the queue is full
                    try:
                        self.messages.get_nowait()
                    except Exception:
                        pass
                    self.messages.put_nowait(message)

            except socket.timeout:
                continue
            except (OSError, ValueError):
                if not self.running:
                    break

    def stop(self):
        self.running = False

        if self.sock is not None:
            self.sock.close()
            self.sock = None

        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
