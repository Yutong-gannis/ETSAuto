import threading


class ReadIOThread(threading.Thread):
    def __init__(self, args, demo_files, queue, eventStop):
        super(ReadIOThread, self).__init__()
