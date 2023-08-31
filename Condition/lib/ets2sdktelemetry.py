import threading

from lib.ets2telemetry import Ets2Telemetry
from lib.sharedmemory import SharedMemory


class Ets2SdkTelemetry:
    def __init__(self):
        self.ets2telemetry = None
        self.interval = 25e-3
        self.shared_memory = SharedMemory()

        self.threading_event = threading.Event()
        self.elapsed()

    def elapsed(self):
        raw_data = self.shared_memory.update()
        self.ets2telemetry = Ets2Telemetry(raw_data)

        if not self.threading_event.isSet():
            threading.Timer(self.interval, self.elapsed).start()
