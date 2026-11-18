
import time
from queue import Empty
from udp_listener import UDPListener

listener = UDPListener()

try:
    listener.start()
    print("CATCHBot frontend listening on UDP port 5005")

    while True:
        try:
            message = listener.messages.get(timeout=1.0)
            print("Received:", message)
        except Empty:
            continue

except KeyboardInterrupt:
    print("\nStopping listener...")

finally:
    listener.stop()
