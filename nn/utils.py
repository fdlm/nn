import time


class Colors:
    """
    Color formatting for console output. Will probably work only under
    Linux/MacOs. Each function adds color escape sequences to a string.
    """

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
    """
    Simple timer class to time different sections of the program. One timer
    object can have multiple timings with different names.
    """

    def __init__(self):
        self.times = dict()
        self.paused = set()
        self.running = set()

    def start(self, name):
        """
        (Re-)start a timer. If the timer was paused, it will continue where it
        left off. If not, timer will start at 0.
        :param name: name of the timer to start
        """
        if name in self.paused:
            self.paused.remove(name)
            add = self.times[name]
        else:
            add = 0

        self.times[name] = time.time() - add
        self.running.add(name)

    def stop(self, name):
        """
        Stop a timer.
        :param name: name of the timer to stop
        """
        if name in self.paused:
            self.paused.remove(name)
        else:
            self.times[name] = time.time() - self.times[name]
            self.running.remove(name)

    def pause(self, name):
        """
        Pause a timer. Timer can be continued using start()
        :param name: name of the timer to pause.
        """
        self.times[name] = time.time() - self.times[name]
        self.paused.add(name)
        self.running.remove(name)

    def __getitem__(self, name):
        """
        Get the current value of a timer.
        :param name: name of the timer to get the current value of
        :return:     time in seconds
        """
        if name in self.running:
            return time.time() - self.times[name]
        return self.times[name]

