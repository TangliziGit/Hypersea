from prettytable import PrettyTable

from config import Config


class Logger:
    prefix = '-->'
    iteration = '-' * 20

    @staticmethod
    def iteration_start(it):
        print("")
        print(f"{Logger.iteration} [[#{it} iteration]] {Logger.iteration}")

    @staticmethod
    def stage(name, content=""):
        print(f"{Logger.prefix} [{name}] {content}")

    @staticmethod
    def print(*content):
        print(*content)

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

        print(table)
        print("")
