import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
import math

from    copy import deepcopy


class MetaNet(nn.Module):
    """
    The net need to be meta learned.
    Parameters are maintained manually.
    """

    def __init__(self, config):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(MetaNet, self).__init__()
        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if name == 'conv2d':
                # args = [out_channel, in_channel, kernel_size,]
                w = nn.Parameter(torch.Tensor(*param[:4]))
                b = nn.Parameter(torch.Tensor(param[0]))
                n = param[1]*param[2]*param[3]
                stdv = 1./math.sqrt(n)
                w.data.uniform_(-stdv, stdv)
                b.data.uniform_(-stdv, stdv)
                self.vars.append(w)
                self.vars.append(b)

            elif name == 'linear':
                # args = [out_channel, in_channel]
                w = nn.Parameter(torch.Tensor(*param))
                b = nn.Parameter(torch.Tensor(param[0]))
                stdv = 1./math.sqrt(w.size(1))
                w.data.uniform_(-stdv, stdv)
                b.data.uniform_(-stdv, stdv)
                self.vars.append(w)
                self.vars.append(b)

            elif name == 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                w.data.uniform_()
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])

            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid','log_softmax',
                          'dropout',]:
                continue
            else:
                print(name)
                raise NotImplementedError

    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name == 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name == 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'

            elif name == 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'

            elif name == 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'

            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info

    def forward(self, x, vars=None):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn == updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_stat==tics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        if vars == None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for name, param in self.config:
            if name == 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # print([i.item() for i in x[:2, 0, 0, 0].view(-1)])
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                # print([i.item() for i in x[:2,0,0,0].view(-1)])
                # print('++++')
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name == 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=self.training)
                idx += 2
                bn_idx += 2
            elif name == 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name == 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name == 'relu':
                x = F.relu(x, inplace=param[0])
            elif name == 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name == 'tanh':
                x = F.tanh(x)
            elif name == 'sigmoid':
                x = torch.sigmoid(x)
            elif name == 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name == 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name == 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])
            elif name == 'log_softmax':
                x = F.log_softmax(x, param[0])
            elif name == 'dropout':
                x = F.dropout(x, param[0])

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)

        return x

    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars

    def loadfromPM(self, filename):

        PMvars = torch.load(filename)
        var_id = 0
        var_bn_id = 0
        # var_keys = self.vars.keys()
        # var_bn_keys = self.vars_bn.keys()
        for key in PMvars:
            if 'running' not in key:
                self.vars[var_id].data = PMvars[key]
                var_id += 1
            else:
                self.vars_bn[var_bn_id].data = PMvars[key]
                var_bn_id += 1

        assert var_id == len(self.vars) and var_bn_id ==len(self.vars_bn), 'Failed to load from PM'


class MetaLearner(nn.Module):
    """
    Meta Learner.
    Optimize the net "Learner.learner" in a meta learning way
    """
    def __init__(self, args, config):
        """

        :param args:
        """
        super(MetaLearner, self).__init__()
        self.device = torch.device(args.device)
        self.update_lr = args.alpha
        self.meta_lr = args.beta
        self.n_way = args.n_way
        self.k_spt = args.shot_k
        self.k_qry = args.query_k
        self.task_num = args.task_num

        self.metanet = MetaNet(config)
        # self.meta_optim = optim.Adam(self.metanet.parameters(), lr=self.meta_lr)

    def forward(self, task_loader, update_step):
        """

        :param update_step:
        :param task_loader:
        :return:
        """

        # losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        loss_q = 0
        corrects = [0 for _ in range(update_step + 1)]
        # print(self.metanet.vars_bn[-1])

        for i, (data, _) in enumerate(task_loader):

            data = data.view(self.n_way,self.k_spt+self.k_qry,3,84,84).to(self.device)
            x_spt = data[:,:self.k_spt,:,:,:].contiguous().view(-1,3,84,84)
            x_qry = data[:,self.k_spt:,:,:,:].contiguous().view(-1,3,84,84)
            y_spt = torch.arange(self.n_way).view(-1, 1).repeat(1, self.k_spt).view(-1).long().to(self.device)
            y_qry = torch.arange(self.n_way).view(-1, 1).repeat(1,self.k_qry).view(-1).long().to(self.device)


            fast_weights = self.metanet.parameters()

            for step in range(0, update_step):

                with torch.no_grad():
                    # [setsz, nway]
                    logits_q = self.metanet(x_qry, fast_weights)
                    loss_q = F.cross_entropy(logits_q, y_qry)
                    # print(logits_q[0])
                    # print(loss_q.item())

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry).sum().item()
                    corrects[step] = corrects[step] + correct

                # 1. run the i-th task and compute loss for k=1~K-1

                logits = self.metanet(x_spt, fast_weights)
                loss = F.cross_entropy(logits, y_spt)
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            # print('QUERY')
            logits_q = self.metanet(x_qry, fast_weights)
            loss_q += F.cross_entropy(logits_q, y_qry)

            # print(logits_q[0])
            # print(loss_q.item())

            with torch.no_grad():
                pred_q = logits_q.argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[update_step] = corrects[update_step] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q /= len(task_loader)
        accs = np.array(corrects) / (self.n_way*self.k_qry * len(task_loader))

        return accs, loss_q


class ProtoMetaLearner(nn.Module):
    """

    """

    def __init__(self, opts, config):
        """

        :param opts:
        :param metanet:
        :param shot_flip:
        """
        super().__init__()
        self.opts = opts
        self.metanet = MetaNet(config)
        self.device = torch.device(opts.device)
        self.n_way = opts.n_way
        self.shot_k = opts.shot_k
        self.query_k = opts.query_k
        self.update_lr = opts.alpha

        self.mask = torch.eye(self.n_way).view(self.n_way, 1, self.n_way, 1).repeat(1, self.shot_k, 1, self.shot_k)\
            .view(self.n_way*self.shot_k, self.n_way*self.shot_k).to(self.device)

    def forward(self, taskloader, update_step, n_way, shot_k):
        if n_way != self.n_way:
            # print(self.n_way)
            self.n_way = n_way
            # print(self.n_way)
            self.mask = torch.eye(self.n_way).view(self.n_way, 1, self.n_way, 1).repeat(1, self.shot_k, 1, self.shot_k)\
                .view(self.n_way * self.shot_k, self.n_way * self.shot_k).to(self.device)
        if shot_k != self.shot_k:
            self.shot_k = shot_k
            self.mask = torch.eye(self.n_way).view(self.n_way, 1, self.n_way, 1).repeat(1, self.shot_k, 1, self.shot_k)\
                .view(self.n_way * self.shot_k, self.n_way * self.shot_k).to(self.device)

        corrects = [0] * (update_step + 1)
        loss_q = 0

        for idx, (data, _) in enumerate(taskloader):
            _, C, H, W = data.size()
            data = data.to(self.device)
            label_shot = torch.arange(self.n_way).view(-1, 1).repeat(1, self.shot_k).view(-1).to(self.device).long()
            label_query = torch.arange(self.n_way).view(-1, 1).repeat(1, self.query_k).view(-1).to(self.device).long()

            fast_weights = self.metanet.parameters()

            # meta inner update with shot and tested with query
            for step in range(update_step):
                emb = self.metanet(data, fast_weights).view(n_way, self.shot_k + self.query_k, -1)
                emb_shot = emb[:, :self.shot_k, :].contiguous()
                # test on query
                with torch.no_grad():
                    emb_query = emb[:, self.shot_k:, :].contiguous().view(self.n_way * self.query_k, -1)
                    shot_ext = emb_shot.mean(dim=1).unsqueeze(0).repeat(self.n_way * self.query_k, 1, 1)
                    query_ext = emb_query.unsqueeze(1).repeat(1, self.n_way, 1)
                    dis = torch.pow(shot_ext - query_ext, 2).sum(dim=2)
                    pred = dis.argmin(dim=1)
                    correct = torch.eq(pred, label_query).sum()
                    corrects[step] = corrects[step] + correct

                # inner update on shot
                emb_shot = emb_shot.view(self.n_way * self.shot_k, -1)
                embshot_ext0 = emb_shot.unsqueeze(1).repeat(1, self.n_way*self.shot_k, 1)
                embshot_ext1 = emb_shot.unsqueeze(0).repeat(self.n_way*self.shot_k, 1, 1)
                dis_s = torch.pow(embshot_ext0 - embshot_ext1, 2).sum(dim=-1)
                with torch.no_grad():
                    farin = (dis_s*self.mask).argmax(dim=1)
                    nearout = (dis_s+self.mask*dis_s.max()).argmin(dim=1)
                loss_s = dis_s.gather(1, farin.view(-1,1)).mean() - dis_s.gather(1, nearout.view(-1,1)).mean()

                grad = torch.autograd.grad(loss_s, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            # the last test on query is used for meta update
            emb = self.metanet(data, fast_weights).view(n_way, self.shot_k + self.query_k, -1)
            # print(emb)
            emb_shot, emb_query = emb[:, :self.shot_k, :].contiguous(), emb[:, self.shot_k:, :].contiguous().view(self.n_way * self.query_k, -1)
            # emb_shot = self.metanet(data_shot, fast_weights).view(self.n_way, self.shot_k, -1)
            # emb_query = self.metanet(data_query, fast_weights).view(self.n_way * self.query_k, -1)
            shot_ext = emb_shot.mean(dim=1).unsqueeze(0).repeat(self.n_way * self.query_k, 1, 1)
            query_ext = emb_query.unsqueeze(1).repeat(1, self.n_way, 1)
            dis = torch.pow(shot_ext - query_ext, 2).sum(dim=2)
            pred = dis.argmin(dim=1)
            correct = torch.eq(pred, label_query).sum()
            # correct = torch.eq(pred, label_query).sum().item()
            corrects[update_step] = corrects[update_step] + correct
            relations = F.log_softmax(-dis, dim=1)
            # print(dis.mean())
            loss_q += -relations.gather(1, label_query.view(-1, 1)).mean()
            # print(loss_q)

        loss_q /= len(taskloader)
        accs = np.array(corrects) / (self.n_way*self.query_k * len(taskloader))
        # print(corrects, self.n_way, self.query_k, len(taskloader))

        return accs, loss_q


class ProtoMetaLearnerAUG(nn.Module):
    """

    """

    def __init__(self, opts, config):
        """

        :param opts:
        :param metanet:
        :param shot_flip:
        """
        super().__init__()
        self.opts = opts
        self.metanet = MetaNet(config)
        self.device = torch.device(opts.device)
        self.n_way = opts.n_way
        self.shot_k = opts.shot_k
        self.query_k = opts.query_k
        self.update_step = opts.update_step
        self.update_lr = opts.alpha

    def forward(self, taskloader, n_way, shot_k):
        if n_way:
            self.n_way = n_way
        if shot_k:
            self.shot_k = shot_k

        init_loss_q = 0
        losses_s = torch.zeros(self.update_step)
        loss_q = 0
        # loss_q = torch.Tensor([0])
        # losses_q = torch.zeros(self.update_step+1)
        accs_q = torch.zeros(self.update_step + 1)

        for idx, (data, _) in enumerate(taskloader):
            _, C, H, W = data.size()
            data = data.to(self.device)
            data = data.view(self.n_way, self.shot_k + self.query_k, C, H, W)
            data_shot = data[:, :self.shot_k, :, :, :].contiguous()
            data_shot = data_shot.view(-1, C, H, W)
            data_query = data[:, self.shot_k:, :, :, :].contiguous()
            data_query = data_query.view(-1, C, H, W)
            label_shot = torch.arange(self.n_way).view(-1, 1).repeat(1, self.shot_k).view(-1).to(self.device).long()
            label_query = torch.arange(self.n_way).view(-1, 1).repeat(1, self.query_k).view(-1).to(self.device).long()

            # flipping is used for shot augmentation especially for one-shot configuration
            data_shot_ori = data_shot.view(self.n_way, self.shot_k, C, H, W)
            data_shot_flip = data_shot_ori[:, :, :, :, range(W - 1, -1, -1)]  # fliped shot data
            data_shot_aug = torch.cat([data_shot_ori, data_shot_flip], dim=1).view(-1, C, H, W).contiguous()
            label_shot_aug = torch.arange(self.n_way).view(-1, 1).repeat(1, 2 * self.shot_k).to(self.device).long()


            fast_weights = self.metanet.parameters()

            # loss on query of the initialization of metanet
            init_emb_shot = self.metanet(data_shot, fast_weights).view(self.n_way, self.shot_k, -1)
            init_emb_query = self.metanet(data_query, fast_weights).view(self.n_way * self.query_k, -1)
            init_shot_ext = init_emb_shot.mean(dim=1).unsqueeze(0).repeat(self.n_way * self.query_k, 1, 1)
            init_query_ext = init_emb_query.unsqueeze(1).repeat(1, self.n_way, 1)
            init_dis = torch.pow(init_shot_ext - init_query_ext, 2).sum(dim=2)
            init_relations = F.log_softmax(-init_dis, dim=1)
            init_loss_q += -init_relations.gather(1, label_query.view(-1, 1)).mean()

            # meta inner update with shot and tested with query
            for step in range(self.update_step):
                # test on query
                with torch.no_grad():
                    emb_shot = self.metanet(data_shot, fast_weights).view(self.n_way, self.shot_k, -1)
                    emb_query = self.metanet(data_query, fast_weights).view(self.n_way * self.query_k, -1)
                    shot_ext = emb_shot.mean(dim=1).unsqueeze(0).repeat(self.n_way * self.query_k, 1, 1)
                    query_ext = emb_query.unsqueeze(1).repeat(1, self.n_way, 1)
                    dis = torch.pow(shot_ext - query_ext, 2).sum(dim=2)
                    pred = dis.argmin(dim=1)
                    acc = (pred == label_query).float().mean()
                    accs_q[step] += acc.item()

                # inner update on shot
                emb_shot_aug = self.metanet(data_shot_aug, fast_weights)
                emb_shot_aug = emb_shot_aug.view(self.n_way, 2*self.shot_k, -1)

                # inner update based on distances between shots and class center
                loss_s = 0
                # for i in range(1):
                #     shuffle_idx = torch.randperm(2*self.shot_k)
                #     emb_shot_aug_shuffled = emb_shot_aug[:,shuffle_idx,:]
                #     label_shot_aug_shuffled = label_shot_aug[:,shuffle_idx].view(-1)
                #     emb_shot_aug_s = emb_shot_aug_shuffled[:,:self.shot_k,:].contiguous()
                #     emb_shot_aug_q = emb_shot_aug_shuffled[:,self.shot_k:,:].contiguous()
                #     label_shot_aug_q = label_shot_aug_shuffled[self.n_way*self.shot_k:].contiguous()
                #     emb_shot_aug_mean = emb_shot_aug_s.mean(dim=1)
                #     emb_shot_aug_mean_ext = emb_shot_aug_mean.unsqueeze(0).repeat(self.n_way*self.shot_k, 1, 1)
                #
                #     emb_shot_aug_q_ext = emb_shot_aug_q.view(self.n_way*self.shot_k,-1).unsqueeze(1).repeat(1, self.n_way, 1)
                #     dis_s = torch.pow(emb_shot_aug_mean_ext - emb_shot_aug_q_ext, 2).sum(dim=2)
                #     relations_s = F.log_softmax(-dis_s, dim=1)
                #     loss_s += -relations_s.gather(1, label_shot_aug_q.view(-1, 1)).mean()

                losses_s[step] = loss_s.item()
                grad = torch.autograd.grad(loss_s, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            # the last test on query is used for meta update
            emb_shot = self.metanet(data_shot, fast_weights).view(self.n_way, self.shot_k, -1)
            emb_query = self.metanet(data_query, fast_weights).view(self.n_way * self.query_k, -1)
            shot_ext = emb_shot.mean(dim=1).unsqueeze(0).repeat(self.n_way * self.query_k, 1, 1)
            query_ext = emb_query.unsqueeze(1).repeat(1, self.n_way, 1)
            dis = torch.pow(shot_ext - query_ext, 2).sum(dim=2)
            pred = dis.argmin(dim=1)
            acc = (pred == label_query).float().mean()
            accs_q[self.update_step] += acc.item()
            relations = F.log_softmax(-dis, dim=1)
            # print(relations.device)
            loss_q += -relations.gather(1, label_query.view(-1, 1)).mean()

        init_loss_q /= len(taskloader)
        losses_s /= len(taskloader)
        loss_q /= len(taskloader)
        accs_q /= len(taskloader)

        return init_loss_q, loss_q, losses_s, accs_q


class HashMetaLearner(nn.Module):
    """

    """
    def __init__(self, args, config):
        """

        :param opts:
        :param config:
        """
        super(HashMetaLearner, self).__init__()
        self.device = torch.device(args.device)
        self.update_lr = args.alpha
        self.n_way = args.n_way
        self.k_spt = args.shot_k
        self.k_qry = args.query_k
        # self.task_num = args.task_num
        self.bits_num = args.num_bits

        self.metanet = MetaNet(config)

    def forward(self, task_loader, update_step, n_way):
        """

        :param n_way:
        :param task_loader:
        :param update_step:
        :return:
        """
        if n_way:
            self.n_way = n_way
        loss = 0
        corrects = [0 for _ in range(update_step + 1)]

        for i, (data, _) in enumerate(task_loader):
            data = data.view(self.n_way, self.k_spt + self.k_qry, 3, 84, 84).to(self.device)
            x_spt = data[:, :self.k_spt, :, :, :].contiguous().view(-1, 3, 84, 84)
            x_qry = data[:, self.k_spt:, :, :, :].contiguous().view(-1, 3, 84, 84)
            data = data.view(-1,3,84,84)
            y_spt = torch.arange(self.n_way).view(-1, 1).repeat(1, self.k_spt).view(-1).long().to(self.device)
            y_qry = torch.arange(self.n_way).view(-1, 1).repeat(1, self.k_qry).view(-1).long().to(self.device)

            fast_weights = self.metanet.parameters()

            for step in range(0, update_step):
                with torch.no_grad():
                    proto = self.metanet(x_spt, fast_weights).view(self.n_way, self.k_spt, -1).mean(dim=1)
                    codeq = self.metanet(x_qry, fast_weights).view(self.n_way*self.k_qry, -1)
                    proto = F.sigmoid(proto)
                    codeq = F.sigmoid(codeq)
                    proto_ext = proto.unsqueeze(0).repeat(self.n_way*self.k_qry,1,1)
                    codeq_ext = codeq.unsqueeze(1).repeat(1, self.n_way,1)
                    dis_q = torch.pow(proto_ext-codeq_ext,2).sum(dim=-1)
                    pred_q = dis_q.argmin(dim=1)
                    correct = torch.eq(pred_q, y_qry).sum().item()
                    corrects[step] = corrects[step] + correct

                codeall = self.metanet(data, fast_weights)
                bits = codeall.mean(dim=0)
                endsloss = -F.binary_cross_entropy_with_logits(codeall, codeall)
                uniloss = F.binary_cross_entropy_with_logits(bits, bits)
                codeall = F.sigmoid(codeall)
                codeall_mean = codeall.mean(dim=0, keepdim=True).repeat(self.n_way*(self.k_spt+self.k_qry),1)
                codeall_uni = codeall - codeall_mean
                corelation = 4*torch.mm(codeall_uni.t(), codeall_uni)
                crloss = torch.pow(corelation-torch.eye(self.bits_num).to(self.device), 2).mean()
                hashloss = endsloss+uniloss+crloss
                # print(endsloss, uniloss, crloss)

                grad = torch.autograd.grad(hashloss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            proto = self.metanet(x_spt, fast_weights).view(self.n_way, self.k_spt, -1).mean(dim=1)
            codeq = self.metanet(x_qry, fast_weights).view(self.n_way * self.k_qry, -1)
            proto = F.sigmoid(proto)
            codeq = F.sigmoid(codeq)
            proto_ext = proto.unsqueeze(0).repeat(self.n_way * self.k_qry, 1, 1)
            codeq_ext = codeq.unsqueeze(1).repeat(1, self.n_way, 1)
            dis_q = torch.pow(proto_ext - codeq_ext, 2).sum(dim=-1)
            relation = F.log_softmax(dis_q, dim=1)
            loss += -relation.gather(1, y_qry.view(-1,1)).mean()
            with torch.no_grad():
                pred_q = dis_q.argmin(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects[update_step] = corrects[update_step] + correct

        loss /= len(task_loader)
        accs = np.array(corrects) / (self.n_way * self.k_qry * len(task_loader))

        return accs, loss


if __name__ == '__main__':
    pass
