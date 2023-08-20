from collections import defaultdict
import logging
import numpy as np

class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value
        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)


# set up a custom logger
def get_logger():
    # 创建一个日志记录对象，logging.getLogger()返回一个全局的日志记录器实例
    logger = logging.getLogger()
    # 清空日志记录器的所有处理器，确保日志记录器没有现有的处理器
    logger.handlers = []
    # 创建一个输出到标准输出流的处理器
    ch = logging.StreamHandler()
    # 创建一个日志消息格式化器，定义了包括日志级别、时间戳、记录器名和消息内容在内的日志消息的显示格式
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    # 将上述定义的日志消息格式化器应用到之前创建的处理器ch
    ch.setFormatter(formatter)
    # 将处理器ch添加到日志记录器logger中，使得日志消息可以通过处理器输出到控制台
    logger.addHandler(ch)
    # 记录的日志级别为debug
    logger.setLevel('DEBUG')

    return logger

