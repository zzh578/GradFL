import torch
from torch import nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import copy

if __name__ == '__main__':
    from utils import init_param, Scaler
else:
    from .utils import init_param, Scaler


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, scaler_rate=1, track=False):
        super(Block, self).__init__()
        self.idx = OrderedDict()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.n1 = nn.BatchNorm2d(out_channels, momentum=None, track_running_stats=track)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=int(self.expansion * out_channels), kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.n2 = nn.BatchNorm2d(int(self.expansion * out_channels), momentum=None, track_running_stats=track)
        self.shortcut = nn.Conv2d(in_channels, int(self.expansion * out_channels), kernel_size=1, stride=stride,
                                  bias=False)
        self.scaler = Scaler(scaler_rate)

    def forward(self, x):
        shortout = self.shortcut(x[1])
        out = self.conv1(x[0])
        out = F.relu(self.n1(self.scaler(out)))
        out = self.conv2(out)
        # temp = self.conv2(out)
        # out = self.scaler(out)
        # out = self.n2(out)
        # out = (F.relu(out)) * 1.0
        temp = out + shortout
        '''
        out += shortout
        out = (F.relu(self.n2(self.scaler(out)))) * 1.0
        '''
        out1 = out + shortout
        out1 = (F.relu(self.n2(self.scaler(out1)))) * 1.0
        # out += shortout
        return out1, temp

    def get_idx_aware(self, input, first_channels, rate, param_name, topmode):
        shortout = self.shortcut(input[1])
        out = F.relu(self.n1(self.scaler(self.conv1(input[1]))))
        second_channels = get_topk_index(out, int(rate * self.conv1.weight.shape[0]), topmode)
        self.idx[param_name + '.conv1.weight'], self.idx[param_name + '.conv1.bias'] = (second_channels,
                                                                                        first_channels), second_channels
        self.idx[param_name + '.n1.weight'], self.idx[param_name + '.n1.bias'] = second_channels, second_channels
        out = F.relu(self.n2(self.scaler(self.conv2(out))))
        third_channels = get_topk_index(out, int(rate * self.conv2.weight.shape[0]), topmode)
        self.idx[param_name + '.conv2.weight'], self.idx[param_name + '.conv2.bias'] = (third_channels,
                                                                                        second_channels), third_channels
        self.idx[param_name + '.n2.weight'], self.idx[param_name + '.n2.bias'] = third_channels, third_channels
        temp = out + shortout
        out += shortout
        self.idx[param_name + '.shortcut.weight'] = (third_channels, first_channels)
        return (out, temp), third_channels

    def get_idx_roll(self, first_channels, rate, roll, param_name):
        second_channels = torch.roll(torch.arange(self.conv1.weight.shape[0]), shifts=roll % self.conv1.weight.shape[0],
                                     dims=-1)[: int(rate * self.conv1.weight.shape[0])]
        self.idx[param_name + '.conv1.weight'] = (second_channels, first_channels)
        self.idx[param_name + '.n1.weight'], self.idx[param_name + '.n1.bias'] = second_channels, second_channels
        third_channels = torch.roll(torch.arange(self.conv2.weight.shape[0]), shifts=roll % self.conv2.weight.shape[0],
                                    dims=-1)[: int(rate * self.conv2.weight.shape[0])]
        self.idx[param_name + '.conv2.weight'] = (third_channels, second_channels)
        self.idx[param_name + '.n2.weight'], self.idx[param_name + '.n2.bias'] = third_channels, third_channels
        self.idx[param_name + '.shortcut.weight'] = (third_channels, first_channels)

        return third_channels

    def get_idx_rand(self, first_channels, rate, param_name):
        second_channels = torch.randperm(self.conv1.weight.shape[0])[: int(rate * self.conv1.weight.shape[0])]
        self.idx[param_name + '.conv1.weight'] = (second_channels, first_channels)
        self.idx[param_name + '.n1.weight'], self.idx[param_name + '.n1.bias'] = second_channels, second_channels

        third_channels = torch.randperm(self.conv2.weight.shape[0])[: int(rate * self.conv2.weight.shape[0])]
        self.idx[param_name + '.conv2.weight'] = (third_channels, second_channels)
        self.idx[param_name + '.n2.weight'], self.idx[param_name + '.n2.bias'] = third_channels, third_channels
        self.idx[param_name + '.shortcut.weight'] = (third_channels, first_channels)

        return third_channels

    def get_idx_hetero(self, first_channels, rate, param_name):
        second_channels = torch.arange(int(rate * self.conv1.weight.shape[0]))
        self.idx[param_name + '.conv1.weight'] = (second_channels, first_channels)
        self.idx[param_name + '.n1.weight'], self.idx[param_name + '.n1.bias'] = second_channels, second_channels

        third_channels = torch.arange(int(rate * self.conv2.weight.shape[0]))
        self.idx[param_name + '.conv2.weight'] = (third_channels, second_channels)
        self.idx[param_name + '.n2.weight'], self.idx[param_name + '.n2.bias'] = third_channels, third_channels
        self.idx[param_name + '.shortcut.weight'] = (third_channels, first_channels)

        return third_channels

    def clear_idx(self):
        self.idx.clear()


class ResNet(nn.Module):
    def __init__(self, datashape, hidden_size, block, num_blocks, num_classes, scaler_rate=1, track=False):
        super(ResNet, self).__init__()
        self.roll = 0
        self.datashape = datashape
        self.idx = OrderedDict()
        self.in_channels = hidden_size[0]
        self.conv = nn.Conv2d(datashape[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.n1 = nn.BatchNorm2d(hidden_size[0], momentum=None, track_running_stats=track)
        self.scaler = Scaler(scaler_rate)
        self.layer1 = self._make_layer(block, hidden_size[0], num_blocks[0], stride=1, scaler_rate=scaler_rate,
                                       track=track)
        self.layer2 = self._make_layer(block, hidden_size[1], num_blocks[1], stride=2, scaler_rate=scaler_rate,
                                       track=track)
        self.layer3 = self._make_layer(block, hidden_size[2], num_blocks[2], stride=2, scaler_rate=scaler_rate,
                                       track=track)
        self.layer4 = self._make_layer(block, hidden_size[3], num_blocks[3], stride=2, scaler_rate=scaler_rate,
                                       track=track)
        self.linear = nn.Linear(int(hidden_size[3] * block.expansion), num_classes)

    def forward(self, x):
        '''
        out = F.relu(self.n1(self.scaler(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
        '''
        temp = self.conv(x)
        out = (F.relu(self.n1(self.scaler(self.conv(x)))), temp)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out, _ = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _make_layer(self, block, out_channels, num_blocks, stride, scaler_rate, track=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_channels, out_channels, scaler_rate=scaler_rate, stride=stride, track=track))
            self.in_channels = int(out_channels * block.expansion)
        return nn.Sequential(*layers)

    def get_idx_aware(self, input, rate, topmode):
        start_channels = (torch.arange(self.datashape[0]))
        temp = self.conv(input)
        out = (F.relu(self.n1(self.scaler(self.conv(input)))), temp)
        first_channels = get_topk_index(out[0], int(rate * self.conv.weight.shape[0]), topmode)
        self.idx['conv.weight'], self.idx['conv.bias'] = (first_channels, start_channels), first_channels
        self.idx['n1.weight'], self.idx['n1.bias'] = first_channels, first_channels
        fun_name = 'self.layer'
        seq_index = ['1', '2', '3', '4']
        for s in seq_index:
            param_name = 'layer' + s
            for i, blc in enumerate(eval(fun_name + s)):
                param_sub_name = param_name + '.' + str(i)
                out, first_channels = blc.get_idx_aware(out, first_channels, rate, param_sub_name, topmode)
                self.idx.update(blc.idx)
        self.idx['linear.weight'] = (torch.arange(self.linear.weight.shape[0]), first_channels)
        self.idx['linear.bias'] = torch.arange(self.linear.weight.shape[0])

    def get_idx_aware_grad(self, rate, select_mode, gradient):
        start_channels = (torch.arange(self.datashape[0]))
        first_channels = get_topk_index(gradient['conv'], int(rate * self.conv.weight.shape[0]), select_mode)
        self.idx['conv.weight'], self.idx['conv.bias'] = (first_channels, start_channels), first_channels
        self.idx['n1.weight'], self.idx['n1.bias'] = first_channels, first_channels
        fun_name = 'layer'
        seq_index = ['1', '2', '3', '4']
        for s in seq_index:
            for lay in ['0', '1']:
                param_name = fun_name + s + '.' + lay
                blc = self.get_attar(param_name)
                second_channels = get_topk_index(gradient[param_name + '.conv1'], int(rate * blc.conv1.weight.shape[0]),
                                                 select_mode)
                self.idx[param_name + '.conv1.weight'] = (second_channels, first_channels)
                self.idx[param_name + '.n1.weight'], self.idx[
                    param_name + '.n1.bias'] = second_channels, second_channels

                third_channels = get_topk_index(gradient[param_name + '.conv2'], int(rate * blc.conv2.weight.shape[0]),
                                                select_mode)
                self.idx[param_name + '.conv2.weight'] = (third_channels, second_channels)
                self.idx[param_name + '.n2.weight'], self.idx[param_name + '.n2.bias'] = third_channels, third_channels
                self.idx[param_name + '.shortcut.weight'] = (third_channels, first_channels)
                first_channels = third_channels
        self.idx['linear.weight'] = (torch.arange(self.linear.weight.shape[0]), first_channels)
        self.idx['linear.bias'] = torch.arange(self.linear.weight.shape[0])

    def get_idx_roll(self, rate):
        start_channels = (torch.arange(self.datashape[0]))
        first_channels = torch.roll(torch.arange(self.conv.weight.shape[0]),
                                    shifts=self.roll % self.conv.weight.shape[0], dims=-1)[
                         : int(rate * self.conv.weight.shape[0])]
        self.idx['conv.weight'] = (first_channels, start_channels)
        self.idx['n1.weight'], self.idx['n1.bias'] = first_channels, first_channels
        fun_name = 'self.layer'
        seq_index = ['1', '2', '3', '4']
        for s in seq_index:
            param_name = 'layer' + s
            for i, blc in enumerate(eval(fun_name + s)):
                param_sub_name = param_name + '.' + str(i)
                first_channels = blc.get_idx_roll(first_channels, rate, self.roll, param_sub_name)
                self.idx.update(blc.idx)
        self.idx['linear.weight'] = (torch.arange(self.linear.weight.shape[0]), first_channels)
        self.idx['linear.bias'] = torch.arange(self.linear.weight.shape[0])

    def get_idx_rand(self, rate):
        start_channels = (torch.arange(self.datashape[0]))
        first_channels = torch.randperm(self.conv.weight.shape[0])[: int(rate * self.conv.weight.shape[0])]
        self.idx['conv.weight'] = (first_channels, start_channels)
        self.idx['n1.weight'], self.idx['n1.bias'] = first_channels, first_channels
        fun_name = 'self.layer'
        seq_index = ['1', '2', '3', '4']
        for s in seq_index:
            param_name = 'layer' + s
            for i, blc in enumerate(eval(fun_name + s)):
                param_sub_name = param_name + '.' + str(i)
                first_channels = blc.get_idx_rand(first_channels, rate, param_sub_name)
                self.idx.update(blc.idx)
        self.idx['linear.weight'] = (torch.arange(self.linear.weight.shape[0]), first_channels)
        self.idx['linear.bias'] = torch.arange(self.linear.weight.shape[0])

    def get_idx_hetero(self, rate):
        start_channels = (torch.arange(self.datashape[0]))
        first_channels = torch.arange(int(rate * self.conv.weight.shape[0]))
        self.idx['conv.weight'] = (first_channels, start_channels)
        self.idx['n1.weight'], self.idx['n1.bias'] = first_channels, first_channels
        fun_name = 'self.layer'
        seq_index = ['1', '2', '3', '4']
        for s in seq_index:
            param_name = 'layer' + s
            for i, blc in enumerate(eval(fun_name + s)):
                param_sub_name = param_name + '.' + str(i)
                first_channels = blc.get_idx_hetero(first_channels, rate, param_sub_name)
                self.idx.update(blc.idx)
        self.idx['linear.weight'] = (torch.arange(self.linear.weight.shape[0]), first_channels)
        self.idx['linear.bias'] = torch.arange(self.linear.weight.shape[0])

    def get_idx_aware_layer_wise(self, input, rate, topmode, device):
        start_channels = (torch.arange(self.datashape[0]))
        temp = self.conv(input)
        out = F.relu(self.n1(self.scaler(self.conv(input))))
        first_channels = self.topk(out, 'conv', int(rate * self.conv.weight.shape[0]), topmode, device)
        self.idx['conv.weight'], self.idx['conv.bias'] = (first_channels, start_channels), first_channels
        self.idx['n1.weight'], self.idx['n1.bias'] = first_channels, first_channels
        fun_name = 'layer'
        seq_index = ['1', '2', '3', '4']
        # second_channels, third_channels = torch.tensor([]), torch.tensor([])
        for s in seq_index:
            for lay in ['0', '1']:
                param_name = fun_name + s + '.' + lay
                blc = self.get_attar(param_name)
                shortout = blc.shortcut(temp)
                out = F.relu(blc.n1(blc.scaler(blc.conv1(out))))
                second_channels = self.topk(out, param_name + '.conv1', int(rate * blc.conv1.weight.shape[0]), topmode,
                                            device)
                self.idx[param_name + '.conv1.weight'] = (second_channels, first_channels)
                self.idx[param_name + '.n1.weight'], self.idx[
                    param_name + '.n1.bias'] = second_channels, second_channels
                out = F.relu(blc.n2(blc.scaler(blc.conv2(out))))
                third_channels = self.topk(out, param_name + '.conv2', int(rate * blc.conv2.weight.shape[0]), topmode,
                                           device)
                self.idx[param_name + '.conv2.weight'] = (third_channels, second_channels)
                self.idx[param_name + '.n2.weight'], self.idx[param_name + '.n2.bias'] = third_channels, third_channels
                temp = out + shortout
                out += shortout
                self.idx[param_name + '.shortcut.weight'] = (third_channels, first_channels)
                first_channels = third_channels
        self.idx['linear.weight'] = (torch.arange(self.linear.weight.shape[0]), first_channels)
        self.idx['linear.bias'] = torch.arange(self.linear.weight.shape[0])

    def topk(self, input, name, k, topmode, device):
        names = name.split('.')
        # pre_sum = torch.sum(input)
        if names[0] == 'conv1':
            blc = self.get_attar('layer1.0')
            return self.get_diff_channels_sum(input, blc, 1, k, topmode, device)
        elif names[0][:-1] == 'layer':
            attr_name = names[0]
            for s in names[1:-1]:
                attr_name += ('.' + s)
            if names[-1] == 'conv1':
                blc = self.get_attar(attr_name)
                return self.get_diff_channels_sum(input, blc, 2, k, topmode, device)
            elif names[-1] == 'conv2':
                if names[1] == '0':
                    blc = self.get_attar(attr_name.replace('0', '1'))
                    return self.get_diff_channels_sum(input, blc, 1, k, topmode, device)
                elif names[1] == '1':
                    if names[0][-1] == '4':
                        blc = self.get_attar('linear')
                        return self.get_diff_channels_sum(input, blc, 3, k, topmode, device)
                    else:
                        next_layer = str(int(names[0][-1]) + 1)
                        names[0] = names[0][:-1] + next_layer
                        names[1] = '0'
                        attr_name_temp = names[0]
                        for s in names[1:-1]:
                            attr_name_temp += ('.' + s)
                        blc = self.get_attar(attr_name_temp)
                        return self.get_diff_channels_sum(input, blc, 1, k, topmode, device)
                else:
                    raise ValueError('layer.0 or 1 error')
        else:
            raise ValueError('names[0]'.format(names[0]))

    def get_diff_channels_sum(self, input, blc, conv_index, k, topmode, device):
        # x = copy.deepcopy(input)
        res = []
        if conv_index == 1 or conv_index == 2:
            if conv_index == 1:
                conv = blc.conv1
                n = blc.n1
            elif conv_index == 2:
                conv = blc.conv2
                n = blc.n2
            else:
                raise ValueError('not match conv')
            for i in range(input.shape[1] + 1):
                # temp = copy.deepcopy(input)
                temp = input.clone()
                if i != input.shape[1]:
                    temp[:, i, :, :] = 0
                i_sum = F.relu(n(blc.scaler(conv(temp))))
                i_sum = torch.sum(i_sum)
                res.append(i_sum)
            pre_sum = res.pop()
            pre_sum = pre_sum.to(device)
            diff_channels_sum = torch.tensor(res).to(device)
            diff_channels_sum = torch.abs(diff_channels_sum - pre_sum)
            return get_topk_index(diff_channels_sum, k, topmode)
        else:  # layer4.1.conv2  using Linear
            for i in range(input.shape[1] + 1):
                # temp = copy.deepcopy(input)
                temp = input.clone()
                if i != input.shape[1]:
                    temp[:, i, :, :] = 0
                temp = F.adaptive_avg_pool2d(temp, 1)
                temp = temp.view(temp.size(0), -1)
                temp = blc(temp)
                i_sum = torch.sum(temp)
                res.append(i_sum)
            pre_sum = res.pop()
            pre_sum = pre_sum.to(device)
            diff_channels_sum = torch.tensor(res).to(device)
            diff_channels_sum = torch.abs(diff_channels_sum - pre_sum)
            return get_topk_index(diff_channels_sum, k, topmode)

    def get_attar(self, attar_name):
        names = attar_name.split('.')
        obj = self
        for x in names:
            obj = getattr(obj, x)
        return obj

    def clear_idx(self):
        self.idx.clear()
        fun_name = 'self.layer'
        seq_index = ['1', '2', '3', '4']
        for s in seq_index:
            for blc in eval(fun_name + s):
                blc.clear_idx()


def get_topk_index(x, k, topmode):
    if topmode == 'absmax':
        if x.dim() > 2:  # conv
            temp = torch.sum(x, dim=(0, 2, 3))
            return torch.topk(temp, k)[1]
        else:  # linear
            return torch.topk(x, k)[1]
    elif topmode == 'probs':
        if x.dim() > 2:
            temp = torch.sum(x, dim=(0, 2, 3))
            probs = torch.abs(temp) / torch.sum(torch.abs(temp))
        else:
            probs = torch.abs(x) / torch.sum(torch.abs(x))
        samples = torch.multinomial(probs, num_samples=k, replacement=False)
        return samples
    elif topmode == 'absmin':
        if x.dim() > 2:
            temp = -torch.sum(x, dim=(0, 2, 3))
            return torch.topk(temp, k)[1]
        else:
            return torch.topk(-x, k)[1]
    else:
        raise ValueError('no method!')


def get_model_params(global_model, client_model_idx):
    client_model_params = OrderedDict()
    for k, v in global_model.state_dict().items():
        if k in client_model_idx:
            if v.dim() > 1:
                client_model_params[k] = copy.deepcopy(v[torch.meshgrid(client_model_idx[k], indexing='ij')])
            else:
                client_model_params[k] = copy.deepcopy(v[client_model_idx[k]])
        else:
            raise NameError('Can\'t match {}'.format(k))
    return client_model_params


def resnet18(datashape, hidden_size, num_blocks, num_classes, track=False, model_rate=1):
    hidden_size = [int(item * model_rate) for item in hidden_size]
    model = ResNet(datashape, hidden_size, Block, num_blocks, num_classes, model_rate, track)
    model.apply(init_param)
    return model




if __name__ == '__main__':
    rate = 0.5
    x = torch.rand((4, 3, 25, 25))
    model = resnet18([3], [8, 8, 16, 16], [3, 4, 6, 3], 10, model_rate=1)
    print(model(x).shape)
    model_half = resnet18([3], [8, 8, 16, 16], [2, 2, 2, 2], 10, model_rate=rate)
    model.get_idx_aware(x, 0.5, 'absmax')
    model_half_shape = copy.deepcopy(model.idx)
    model.clear_idx()
    model_half_params = OrderedDict()
    # for k, v in model.state_dict().items():
    #     print(k)
    print('yes')
    for k, v in model.named_modules():
        print(k)
