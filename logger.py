import time

from prettytable import PrettyTable

from config import Config


class Logger:
    _filename = f"log/{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))}.log"
    _file = open(_filename, 'w')

    _prefix = '-->'
    _iteration = '-' * 20

    @staticmethod
    def iteration_start(it, content = None):
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
        Logger._file.write(content + '\n')
        Logger._file.flush()

    @staticmethod
    def show_params():
        table = PrettyTable(["Type", "ofFilter", "FilterHeight", "FilterWidth", "StrideHeight", "StrideWidth"])

        table.add_row(['current', Config.of_filter,
                       Config.filter_height, Config.filter_width,
                       Config.stride_height, Config.stride_width])

        table.add_row(['best', Config.best_of_filter,
                       Config.best_filter_height, Config.best_filter_width,
                       Config.best_stride_height, Config.best_stride_width])

        table.add_row(['worst', Config.worst_of_filter,
                       Config.worst_filter_height, Config.worst_filter_width,
                       Config.worst_stride_height, Config.worst_stride_width])

        Logger.print(str(table))
        Logger.print("")
