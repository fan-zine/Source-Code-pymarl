import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


# 负责初始化、配置、运行训练过程，并在训练完成后进行清理和退出
def run(_run, _config, _log):

    # check args sanity
    # 检查参数的合理性
    _config = args_sanity_check(_config, _log)

    # 如果 _config 包含键值对 {"use_cuda": True, "batch_size_run": 32}，那么 args 对象将具有属性 args.use_cuda 和 args.batch_size_run，分别对应配置中的 use_cuda 和 batch_size_run 参数的值
    args = SN(**_config)
    # 设置args的计算设备为CUDA或CPU
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    # 初始化日志记录器
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    # 把_config中的内容格式化成一个可打印的字符串 indent=4：每个键值对之间的缩进空格数为4 width=1：每行文本的最大宽度为1，也就是输出的字符串尽可能保持单行显示，不会自动换行
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    # 生成一个唯一的标识符unique_token，args.name：从字典中提取出的参数，用于表示实验的名称，会得到一个类似"qmix__2023-07-31_13-45-59"的唯一标识符，用于标记当前实验的结果文件夹
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    # 如果默认配置里开启了tensorboard
    if args.use_tensorboard:
        # tb_logs_direc表示Tensorboard日志文件的根目录：用当前脚本所在目录+results+tb_logs文件夹连接起来，形成tensorboard日志文件夹的完整路径
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        # tb_exp_direc表示实验的tensorboard日志文件夹的路径，用tb_logs_direc和unique_token进行组合，奖唯一标识符添加到tensorboard日志文件夹的路径中
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        # 设置tensorboard日志记录，表示tb日志应该存储在这个路径下
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    # 用于把日志信息与Sacred集成起来，把实验的日志信息记录到Sacred数据库中，实验运行期间产生的所有日志信息、输出等都会被传递给Sacred
    logger.setup_sacred(_run)

    # Run and train
    # 重要！开始运行训练过程
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    # threading.enumerate是一个Python内置函数，返回当前所有活动线程的列表
    for t in threading.enumerate():
        # 如果不是主线程
        if t.name != "MainThread":
            # 打印出当前线程的名称和是否为守护线程
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            # 这行代码试图等待当前线程终止，join方法会阻塞当前线程，直到被调用的线程终止
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    # 确保框架真正退出
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    # 循环会运行args.test_nepisode次，每次都会调用runner.run(test_mode=True)，从而在测试模型下执行一次评估
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    # 如果args.save_replay为真，表示需要保存回放数据
    if args.save_replay:
        runner.save_replay()

    # 关闭评估过程中打开的环境
    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    # 创建一个名为runner的对象，根据传入的参数arg.runner来选择合适的runner类
    # runner的主要作用是运行环境以产生训练样本
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    # 通过runner实例的get_env_info方法获取环境信息
    env_info = runner.get_env_info()
    # 从env_info中提取agent数量、agent可执行的动作的数量和agent的状态的形状(例如在迷宫游戏中，这个状态可能表示每个格子里的内容，如墙、空地等)
    # 把环境信息分配给args对象的属性，可以确保在整个训练过程中使用正确的参数和配置
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    # scheme是用于配置智能体与环境之间信息交互格式的数据结构，它指定了如何组织和传递智能体的状态、观察、动作、奖励等信息
    scheme = {
        # state:agent的状态 vshape:对应一个元组，表示状态的形状(维度)
        "state": {"vshape": env_info["state_shape"]},
        # obs:agent的观察 vshape：对应一个元组，表示观察信息的形状 group：多智能体环境下指定观察信息所属的智能体组
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        # actions：智能体采取的动作 vshape：对应一个元组，表示动作的形状 dtype：指定了动作的数据类型，这里的th.long表示动作是整数类型
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        # avail_actions：表示智能体可采取的动作集合 vshape：表示可用动作的形状 group：指定可用动作集合所属的智能体组
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        # reward： 表示智能体获得的奖励 vshape：对应一个元组，表示奖励的形状，通常是一个标量值
        "reward": {"vshape": (1,)},
        # terminated： 表示智能体是否完成任务 vshape：表示终止信号的形状 dtype：表示终止信号是一个bool类型
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    # 组信息
    groups = {
        "agents": args.n_agents
    }
    # 预处理信息
    preprocess = {
        # actions_onehot：意味着将动作进行one-hot编码的预处理
        # [OneHot(out_dim=args.n_actions)]：out_dim表示one-hot编码的向量维度
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    # 创建一个用于存储训练数据的缓冲区，使得智能体可以从中随机采样来进行训练
    # args.buffer_size 表示回放缓冲区的最大容量
    # env_info["episode_limit"] + 1 表示每个episode的最大长度
    # ReplayBuffer的父类是EpisodeBatch，用于存储episode的样本
    # ReplayBuffer类对象用于存储所有的off-policy样本
    # EpisodeBatch与ReplayBuffer中的样本都是以episode为单位存储的
    # EpisodeBatch中数据的维度是[batch_size, max_seq_length, *shape]，batch_size表示此时batch中有多少episode
    # ReplayBuffer中数据的维度是[buffer_size, max_seq_length, *shape]，buffer_size表示此时buffer中有多少个episode的有效样本，max_seq_length表示一个episode的最大长度
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    # 从mac_REGISTRY中选择并实例化一个多智能体控制器MAC
    # args.mac是一个参数，表示希望使用的多智能体控制器的类型，如qmix、coma等，使用这个作为key来选择适当的构造函数
    # mac的主要作用是控制智能体
    # 接收观测作为输入，输出智能体各个动作的隐藏层值
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    # 为训练环境设置所需的数据格式、分组、预处理，以及多智能体控制器，使得训练环境准备好接收数据并进行训练
    # 用于以episode为单位存储环境运行所产生的样本
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    # 用于创建一个学习器learner，可以根据环境的反馈和MAC的决策来更新agent的决策以及可能的值函数
    # learner对象需要学习的参数包括各个智能体的局部Q网络参数mac.parmeters()，以及混合网络参数learner.mixer.parameters()，两者共同组成了learner.params，然后用优化器learner.optimiser进行优化
    # 最终mac集成进了learner模块，learner的更新中有着QMIX模块和RNN智能体模块，最终的学习也在学习器里面进行，所以对于Q值网络的模拟如Qtran的修改，都在此模块进行，重中之重
    # 该对象的主要作用是依据特定算法对智能体参数进行训练更新
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    # 如果使用CUDA，将learner移动到GPU
    if args.use_cuda:
        learner.cuda()

    # 从checkpoint加载模型
    # 首先检查是否已经指定了一个检查点路径，如果没有指定，就没有必要继续后面的操作
    # 如果有已经保存的模型，就读取此模型，接着训练
    if args.checkpoint_path != "":

        # 创建一个空列表，用于存储检查点路径中的时间步信息
        timesteps = []
        # 初始化一个变量，用于存储要加载的模型的时间步
        timestep_to_load = 0

        # 检查指定的检查点路径是否存在，如果路径不存在，将在日志中记录警告信息并返回
        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        # 遍历检查点路径中的所有文件和文件夹
        for name in os.listdir(args.checkpoint_path):
            # 将当前文件名或文件夹名与检查点路径连接起来，得到完整的路径
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            # 检查是否是一个文件夹，并且文件夹名称是数字，因为在模型存储时，以时间步作为文件夹名称，所以这里只选择数字文件夹作为候选
            if os.path.isdir(full_name) and name.isdigit():
                # 如果满足条件，就将文件夹名（时间步）转换为整数并添加到时间步列表中
                timesteps.append(int(name))

        # 如果未指定加载时间步（args.load_step为0），则选择最大的时间步作为要加载的模型时间步
        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        # 如果指定了加载时间步，则选择与指定时间步最接近的时间步
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        # 构建要加载模型的路径
        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        # 在日志中记录加载模型的路径
        logger.console_logger.info("Loading model from {}".format(model_path))
        # 使用学习器(learner)从指定路径加载模型
        learner.load_models(model_path)
        # 更新运行器(runner)的时间步为加载的时间步
        runner.t_env = timestep_to_load

        # 如果指定了评估或保存回放数据，执行下面的操作
        if args.evaluate or args.save_replay:
            # 执行评估逻辑，传入参数和运行器
            evaluate_sequential(args, runner)
            return

    # start training
    # 开始训练，初始化episode计数器，用于跟踪训练过程中已经完成 的轮数
    # TODO https://blog.csdn.net/m0_62313824/article/details/134840516?spm=1001.2014.3001.5502
    episode = 0
    # 初始化上一次测试评估的时间步数，设置为一个负数，确保在开始时进行初始评估
    last_test_T = -args.test_interval - 1
    # 初始化上一次记录训练统计信息的时间步数，设置为 0，确保在开始时记录初始统计信息
    last_log_T = 0
    # 初始化上一次保存模型的时间步数，设置为 0，确保在开始时进行初始模型保存
    model_save_time = 0

    # 记录训练开始的时间
    start_time = time.time()
    # 初始化last_time为训练开始的时间
    last_time = start_time

    # 打印一条日志，指示开始进行一共 args.t_max 个时间步的训练
    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        # 运行一整个episode，将获得的episode数据存储在episode_batch中
        # 利用当前智能体mac在环境中运行，产生一个episode的样本数据episode_batch，存储在runner.batch中
        episode_batch = runner.run(test_mode=False)
        # 将episode_batch插入回放缓冲区，以便后续的样本采样和训练
        # 将EpisodeBatch变量ep_batch中的样本全部存储到buffer中
        buffer.insert_episode_batch(episode_batch)

        # 检查回放缓冲区是否包含足够的样本，可以进行批量采样
        if buffer.can_sample(args.batch_size):
            # 从回放缓冲区中采样出batch_size个episode的样本用于训练
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            # 计算采样批次中填充的最大时间步数
            max_ep_t = episode_sample.max_t_filled()
            # 截断批次数据，只保留填充的时间步数
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            # 使用采样的数据对学习器进行训练
            # episode_sample:表示当前用于训练的样本，t_env:表示当前环境运行的总时间步数，episode:表示当前环境运行的总episode数，该方法利用特定算法对learner.params进行更新
            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        # 计算需要执行的测试轮数，确保每个测试轮次都至少包含一个批次大小
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        # 检查是否达到了执行测试的时间间隔，如果达到了测试时间间隔，会进行一次测试评估，评估一系列测试轮次，并记录测试结果，打印出当前训练进度信息，如时间步数、剩余时间估计等
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            # time_left是一个用于估计剩余训练时间的函数
            # time_str是一个用于将时间转换为易读的字符串格式
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            # 更新last_time
            last_time = time.time()

            # 更新 last_test_T 为当前时间步数，以便在下一次检查测试时间间隔时使用
            last_test_T = runner.t_env
            # 执行测试评估。根据 n_test_runs 的值，可能会多次运行测试轮次，每轮都会通过
            for _ in range(n_test_runs):
                # 在测试模式下运行一整个 episode。这个操作有助于评估模型在当前训练状态下的性能
                runner.run(test_mode=True)

        # args.save_model：这是一个设置，指示是否要在训练过程中保存模型
        # (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0)：这个条件检查是否达到了保存模型的时间间隔
        # model_save_time 是上一次保存模型的时间步数，runner.t_env 是当前的时间步数，args.save_model_interval 是设置的保存模型的时间间隔
        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            # 更新 model_save_time 为当前时间步数，以便在下一次检查保存模型时间间隔时使用
            model_save_time = runner.t_env
            # 构建模型保存路径，这个路径包括了模型保存的文件夹和文件名，args.local_results_path 是保存结果的路径，args.unique_token 是一个唯一标识符，str(runner.t_env) 是当前时间步数
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            # 如果路径不存在的话就创建保存路径
            os.makedirs(save_path, exist_ok=True)
            # 将保存模型的路径打印到日志中，以便在训练过程中可以查看模型的保存情况
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            # 通过调用 learner 对象的方法来保存模型的权重和参数
            learner.save_models(save_path)

        # 增加当前的 episode 计数器
        episode += args.batch_size_run

        # 这个条件检查是否达到了打印训练统计信息的时间间隔
        if (runner.t_env - last_log_T) >= args.log_interval:
            # 调用 logger 对象的方法来记录训练的统计信息。这个方法会记录指定的键值对，其中 "episode" 是键，episode 是当前的 episode 计数器，runner.t_env 是当前的时间步数
            logger.log_stat("episode", episode, runner.t_env)
            # 调用 logger 对象的方法来打印最近的训练统计信息，以便在训练过程中可以查看训练的进展情况
            logger.print_recent_stats()
            # 更新 last_log_T 为当前时间步数，以便在下一次检查打印统计信息时间间隔时使用
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    # 检查是否可用CUDA，如果config["use_cuda"]为true且没有可用的cuda设备，则将其设置为false
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    # test_nepisode：用来配置测试过程中有n个episode  每个episode中会运行一次测试，以评估模型在当前状态下的性能表现，多个episode取平均能更准确
    # batch_size_run：指定了每次训练使用的批次大小  通常会把训练数据分成多个批次，每个批次包含一定数量的训练样例，模型会根据这些批次进行权重更新，以进行训练
    # 确保测试的episode次数是batch_size的整数倍，以便能够完整运行整个batch
    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
