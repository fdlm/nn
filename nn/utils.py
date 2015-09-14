import time


class Colors:

    @staticmethod
    def red(string):
        return '\033[0;31m' + string + '\033[0m'

    @staticmethod
    def green(string):
        return '\033[0;32m' + string + '\033[0m'

    @staticmethod
    def yellow(string):
        return '\033[0;33m' + string + '\033[0m'

    @staticmethod
    def blue(string):
        return '\033[0;34m' + string + '\033[0m'

    @staticmethod
    def magenta(string):
        return '\033[0;35m' + string + '\033[0m'

    @staticmethod
    def cyan(string):
        return '\033[0;36m' + string + '\033[0m'

    @staticmethod
    def white(string):
        return '\033[0;37m' + string + '\033[0m'


class Timer:

    def __init__(self):
        self.times = dict()
        self.paused = set()
        self.running = set()

    def start(self, name):
        if name in self.paused:
            self.paused.remove(name)
            add = self.times[name]
        else:
            add = 0

        self.times[name] = time.time() - add
        self.running.add(name)

    def stop(self, name):
        if name in self.paused:
            self.paused.remove(name)
        else:
            self.times[name] = time.time() - self.times[name]
            self.running.remove(name)

    def pause(self, name):
        self.times[name] = time.time() - self.times[name]
        self.paused.add(name)
        self.running.remove(name)

    def __getitem__(self, name):
        if name in self.running:
            return time.time() - self.times[name]
        return self.times[name]

