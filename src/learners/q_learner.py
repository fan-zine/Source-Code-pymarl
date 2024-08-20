import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop


# 如果想修改QMIX之类的网络结构如Qtran，就在此处修改
class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    # batch:当前用于训练的样本，t_env：当前环境运行的总时间步数，episode_num:当前环境运行的总episode数，方法利用特定算法对learner.params进行更新
    # 这些episode是已经在环境里运行完一次之后，产生的数据，sample出一组数据，然后放到这里来训练
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        # batch是一个字典，reward键对应着一个二维数组或tensor，这个数组的每一行代表一个episode的奖励，每一列代表时间步上的奖励值
        # 切片操作排除了每行最后的一个元素，因为RL通常需要当前时间步的奖励来计算预期的未来奖励，而最后一个时间步之后没有后续奖励，所以不需要考虑
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        # terminated的取值通常为0(未终止)或1(已终止)
        terminated = batch["terminated"][:, :-1].float()
        # filled通常用于指示episode中哪些时间步是有数据的，哪些是填充的，在处理不同长度的序列时非常重要，以避免使用无效的数据
        mask = batch["filled"][:, :-1].float()
        # 将mask数组中除了第一列之外的所有列与(1 - terminated[:, :-1])相乘。当terminated为1时，1 - terminated为0，这将使得相应的mask值变为0，从而在后续的计算中忽略该时间步
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        # 表示在每个时间步上可选的动作集合，在部分可观测环境或有约束的条件中尤为重要
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        # 初始化多智能体控制器的隐藏层
        self.mac.init_hidden(batch.batch_size)
        # 遍历序列的最大长度
        for t in range(batch.max_seq_length):
            # 调用mac对象的前向传播方法，在给定的时间步t上执行模型的计算
            agent_outs = self.mac.forward(batch, t=t)
            # mac_out是一个列表，用于存储每个时间步的输出结果
            mac_out.append(agent_outs)
        # 原本保存在mac_out中的各个时间步的输出会被拼接成一个新的张量，其中新的维度代表时间步
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        # 最后得到的tensor，包含了每个agent在每个时间步上采取的动作对应的q值
        # 从模型的输出中提取出agent实际采取的动作对应的Q值，以便进一步计算损失函数或更新策略
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # mac_out是行为网络在每个时间步上产生的输出，通常包含了所有Agent在当前时间步上可能采取的所有动作的Q值估计，主要用于选择Agent当前时间步上的动作以及计算当前网络的损失函数
        # target_mac_out是目标网络在每个时间步上产生的输出，用于计算目标Q值，用于评估当前网络性能的标准，反映了采取某个动作后预期的长期回报

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        # 如果某个动作在某个时间步上不可用，那么该动作的Q值会被设置为一个非常小的数，这样做的目的是确保在计算目标Q值时，Agent不会选择不可用的动作
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            # 创建mac_out的一个副本，将其从计算图中分离出来，以避免梯度计算。这是因为我们只关心当前网络的最大Q值动作，而不是要更新它的参数
            mac_out_detach = mac_out.clone().detach()
            # 将不可用动作的Q值设置为一个非常小的数，以确保它们不会被选为最大Q值的动作
            mac_out_detach[avail_actions == 0] = -9999999
            # 第3个维度是动作维度
            # 这句的意思是从当前网络的输出中找到第一个时间步之外的每个时间步的最大Q值动作索引
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            # 使用cur_max_actions来从target_mac_out中收集对应的最大Q值
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            # 如果使用的是标准Q学习，则直接从target_mac_out中找到每个时间步的最大Q值
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            # 调用混合器的前向传播方法，入参为 单个智能体选择的动作对应的Q值 和 当前批次的状态数据
            # 混合器接收每个agent的Q值和状态作为输入，并输出一个全局的Q值，这个Q值是考虑到所有Agent的贡献后计算出来的
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            # 目标混合器，参数通常是定期从主混合器复制过来的，以保持一定的稳定性
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        # rewards是当前时间步的即时奖励
        # gamma是折扣因子
        # terminated是一个指示序列，指示是否终止的二进制张量，1表示终止，0表示未终止
        # target_max_qvals目标网络计算出的每个时间步的最大Q值
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        # 主网络计算出的Agent选择的动作对应的Q值 - 目标Q值。使用detach()方法是为了避免梯度回传到目标网络
        td_error = (chosen_action_qvals - targets.detach())
        # 将mask扩展为与td_error相同的形状，表示哪些时间步的数据是有效的
        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        # 使得来自填充数据的TD误差被置为0，避免使用填充数据来计算损失
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        # 计算L2损失(均方误差)，衡量了预测Q值与目标Q值之间的差距
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        # self.optimiser优化器对象，例如Adam、SGD等，用于更新网络参数
        # zero_gard()清除优化器内部的梯度缓存，在每次反向传播前都需要清除梯度，以免梯度积累
        self.optimiser.zero_grad()
        # 启动反向传播，计算损失函数关于网络参数的梯度
        loss.backward()
        # 对网络参数的梯度进行裁剪，防止梯度爆炸
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        # 使用优化器更新网络参数
        self.optimiser.step()

        # 判断从上一次更新目标网络以来，是否已经过了足够的训练轮次，如果满足条件，则更新目标网络的参数，并更新上一次更新目标网络时的训练轮次编号
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # 判断上一次记录日志以来，是否已经过了足够的环境步数，如果满足，则记录以下统计信息
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
