import time

from prettytable import PrettyTable

from config import Config


class Logger:
    _q_tables = None
    _filename = f"log/{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))}.log"
    _file = None
    if not Config.TEST:
        _file = open(_filename, 'w')

    _prefix = '-->'
    _iteration = '-' * 25
    _error = '!' * 30

    @staticmethod
    def set_q_tables(q_tables):
        Logger._q_tables = q_tables

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
        if not Config.TEST:
            Logger._file.write(content + '\n')
            Logger._file.flush()

    @staticmethod
    def error(e):
        Logger.print(f"{Logger._error} [[ ERROR OCCUR ]] {Logger._error}")
        Logger.print(e)

    @staticmethod
    def show_status(is_stable=False):
        if is_stable:
            Logger.print(f"{'#' * 30} STABLE STATUS {'#' * 30}")

        # show param status
        table = PrettyTable(["Type", "OfFilter", "FilterHeight", "FilterWidth", "StrideHeight", "StrideWidth", "Acc"])

        table.add_row(['current', Config.of_filter,
                       Config.filter_height, Config.filter_width,
                       Config.stride_height, Config.stride_width,
                       f"{Config.last_accuracy} (last)"])

        table.add_row(['best', Config.best_of_filter,
                       Config.best_filter_height, Config.best_filter_width,
                       Config.best_stride_height, Config.best_stride_width,
                       Config.best_acc])

        table.add_row(['worst', Config.worst_of_filter,
                       Config.worst_filter_height, Config.worst_filter_width,
                       Config.worst_stride_height, Config.worst_stride_width,
                       Config.worst_acc])

        Logger.print(str(table))
        Logger.print("")

        # show q table status
        def get_q_table_str(name, df):
            Logger.stage("QTable", f"{name} (0 for Sub, 1 for Add)")
            qtable = PrettyTable(['Status'] + list(df.columns.array))
            for row_name, row in zip(df.index.values, df.values):
                qtable.add_row([row_name] + list(row))
            return str(qtable)

        Logger.print(get_q_table_str("OfFilter", Logger._q_tables.RL_table_of_filter.q_table))
        Logger.print(get_q_table_str("FilterHeight", Logger._q_tables.RL_table_filter_height.q_table))
        Logger.print(get_q_table_str("FilterWidth", Logger._q_tables.RL_table_filter_width.q_table))
        Logger.print(get_q_table_str("StrideHeight", Logger._q_tables.RL_table_stride_height.q_table))
        Logger.print(get_q_table_str("StrideWidth", Logger._q_tables.RL_table_stride_width.q_table))
        Logger.print("")

