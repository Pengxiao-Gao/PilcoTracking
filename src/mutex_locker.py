from threading import Lock


class Mutex_Locker() :
    def __init__(self) :
        self.mutexState = Lock()
        self.mutexPath = Lock()
        self.mutexCtrl = Lock()