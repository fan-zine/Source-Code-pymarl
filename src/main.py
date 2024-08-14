import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run import run

# 自定义设置，指示日志如何处理标准输出和标准错误输出
SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
# 日志记录器，用于记录程序运行期间的信息
logger = get_logger()

# Experiment是Sacred框架的核心类
# TODO 细看一下Scared-Experiment
ex = Experiment("pymarl")
# 将之前创建的日志记录器分配给实验对象的日志记录器，这样实验的日志将由该记录器记录
ex.logger = logger
# 设置输出文件格式，避免有些实时输出（进度条等）不适合文件输出的形式
ex.captured_out_filter = apply_backspaces_and_linefeeds

# 设置结果路径，通过将当前文件的绝对路径的两级目录与"results"目录连接而形成
results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


# @ex.main一个装饰器，将下面的my_main标记为实验的主函数，也就是说执行实验时，my_main函数将会被调用
@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    # 深拷贝，在修改配置时不会影响原始的_config
    config = config_copy(_config)
    # 设置随机数生成器的种子，以确保每次运行时生成的随机数序列是一样的，可以使得实验结果在不同运行之间具有可重复性
    np.random.seed(config["seed"])
    # 与上面类似，设置PyTorch的随机数生成器的种子
    th.manual_seed(config["seed"])
    # 将配置中的环境参数的种子设置为相同的种子值，以确保环境的随机性在不同运行之间是一致的
    config['env_args']['seed'] = config["seed"]

    # run the framework
    # 重要！是实验运行的入口，运行整个框架
    run(_run, config, _log)

# 从命令行参数中提取指定参数名的值，并使用这个值来找到配置文件并加载它
def _get_config(params, arg_name, subfolder):
    config_name = None
    # 遍历params列表，找到与指定参数名相匹配的命令行参数，一旦找到了匹配，就会提取参数值，并从params列表中删除这个参数
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    # 使用提取的参数值构建配置文件的路径，将其加载并解析为字典
    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        # 返回解析后的配置字典
        return config_dict

# 递归的把字典u合并到字典d
def recursive_dict_update(d, u):
    for k, v in u.items():
        # 如果v是一个字典，则递归调用recursive_dict_update将v合并到d[k]
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        # 如果v不是字典，直接将其赋值给d[k]
        else:
            d[k] = v
    return d


# 深拷贝复制配置文件
def config_copy(config):
    # 如果config是一个字典，这个分支会创建一个新的字典，并对字典中的每个键值对递归调用config_copy复制
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    # 如果config是一个列表，这个分支会创建一个新的列表，并对列表中的每个元素调用config_copy复制
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    # 如果是原子类型，就用deepcopy函数创建这个原子类型的完全独立拷贝
    else:
        return deepcopy(config)


# 确保代码仅在直接运行这个脚本时执行，而不是被其他脚本导入时执行
if __name__ == '__main__':
    # 深拷贝了命令行参数sys.argv
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    # 打开当前脚本所在目录的config子目录下的config.yaml配置文件，获取默认配置信息
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            # 把default.yaml配置文件的内容存储在config_dict变量中
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            # 如果加载配置文件时出现错误，会引发断言错误
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    # 调用_get_config来获取环境配置，这个配置从命令行参数中提取，--env-config是用于指定配置文件路径的命令行参数
    env_config = _get_config(params, "--env-config", "envs")
    # 调用_get_config来获取算法配置，这个配置从命令行参数中提取
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    # 下面两行将环境配置和算法配置递归地更新到config_dict中
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # now add all the config to sacred
    # 现在config_dict包含了默认配置、环境配置和算法配置，把更新后的配置添加到ex中(Sacred实验对象)
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    # 运行Sacred实验时，实验的配置、运行日志和结果数据需要被保存起来，以便后续的分析和复现
    # file_obs_path存储了完整的观察者路径
    file_obs_path = os.path.join(results_path, "sacred")
    # 创建一个FileStorageObserver实例，并将其添加到ex.observers列表中，观察者会监听实验的不同阶段，将相关信息保存到指定的目录中
    # 在这里我们使用FileStorageObserver来保存实验信息和结果到之前创建的file_obs_path目录中
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    # 通过这行来运行Sacred实验，会调用之前通过装饰器注册的函数
    ex.run_commandline(params)

