import time

from prettytable import PrettyTable

from config import Config


class Logger:
    _filename = f"log/{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))}.log"
    _file = None
    if not Config.TEST:
        _file = open(_filename, 'w')

    _prefix = '-->'
    _iteration = '-' * 34
    _error = '!' * 30

    @staticmethod
    def iteration_start(it, content=None):
        Logger.print("")
        if content is None:
            Logger.print(f"{Logger._iteration} [[#{it} iteration]] {Logger._iteration}")
        else:
            Logger.print(f"{Logger._iteration} [[#{it} iteration ({content})]] {Logger._iteration}")

    @staticmethod
    def stage(name, content=""):
        Logger.print(f"{Logger._prefix} [{name}] {content}")

    @staticmethod
    def print(*content):
        content = [str(x) for x in content]
        content = ' '.join(content)

        print(content)
        if not Config.TEST:
            Logger._file.write(content + '\n')
            Logger._file.flush()

    @staticmethod
    def error(e):
        Logger.print(f"{Logger._error} [[ ERROR OCCUR ]] {Logger._error}")
        Logger.print(e)

    @staticmethod
    def show_status():
        table = PrettyTable(["Type", "OfFilter", "FilterHeight", "FilterWidth", "StrideHeight", "StrideWidth", "Acc"])

        table.add_row(['current'] + Config.states + [Config.acc])
        table.add_row(['best'] + Config.best_states + [Config.best_acc])
        table.add_row(['worst'] + Config.worst_states + [Config.worst_acc])

        Logger.print(str(table))
        Logger.print("")

