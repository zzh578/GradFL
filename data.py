import os
import random
import sys
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
import torchvision
import localdatasets


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_transform(dataset, model_name):
    transform = None
    if dataset == 'emnist':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    elif dataset == 'cifar10':
        transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.Resize(224) if model_name == 'vgg16' else transforms.RandomHorizontalFlip(p=0),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    elif dataset == 'cifar100':
        transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.Resize(224) if model_name == 'vgg16' else transforms.RandomHorizontalFlip(p=0),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    elif dataset == 'tinyimagenet':
        # transform = None    # Have finished in dataset
        transform = transforms.Compose([
        transforms.Resize(224) if model_name == 'vgg16' else transforms.RandomHorizontalFlip(p=0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    else:
        raise ValueError('Can\'t find {}'.format(dataset))
    return transform


def get_dataset(dataset_name, transform, dataset_dir):
    path = os.path.join(dataset_dir, 'data', dataset_name)
    processed = os.path.join(path, 'processed')
    if os.path.exists(processed):
        shutil.rmtree(processed)
        print('Deleted last processed!')
    else:
        print('Dataset is clear!')
    dataset = {}
    if dataset_name == 'emnist':
        dataset['train'] = localdatasets.MNIST(path, 'train', 'label', transform=transform)
        dataset['test'] = localdatasets.MNIST(path, 'test', 'label', transform=transform)
    elif dataset_name == 'cifar10':
        dataset['train'] = localdatasets.CIFAR10(path, 'train', 'label', transform=transform)
        dataset['test'] = localdatasets.CIFAR10(path, 'test', 'label', transform=transform)
    elif dataset_name == 'cifar100':
        dataset['train'] = localdatasets.CIFAR100(path, 'train', 'label', transform=transform)
        dataset['test'] = localdatasets.CIFAR100(path, 'test', 'label', transform=transform)
    elif dataset_name == 'tinyimagenet':
        dataset['train'] = localdatasets.TinyImagenet(path, 'train', 'label', transform=transform)
        dataset['test'] = localdatasets.TinyImagenet(path, 'valid', 'label', transform=transform)
    else:
        raise ValueError('can\'t find {}'.format(dataset_name))
    return dataset


def get_inferen_data(mode, inferen_batch, class_list, inferen_label, client_dataset, global_dataset):
    inferen_data = None
    if mode == 'awareGrad':
        '''
        # using local data
        if inferen_batch == -1:
            inferen_batch = len(dataset)
        '''
        inferen_list = []
        for i in inferen_label:
            inferen_list.extend(class_list[i][:inferen_batch // len(inferen_label)])
        inferen_dataset = SplitDataset(global_dataset, inferen_list)
        dataloader = make_dataloader(inferen_dataset, len(inferen_list))
    elif mode == 'aware':
        if inferen_batch == -1:
            inferen_batch = len(client_dataset)
        dataloader = make_dataloader(client_dataset, inferen_batch if inferen_batch <= len(client_dataset) else len(client_dataset))
    else:
        return inferen_data
    iterator = iter(dataloader)
    inferen_data = next(iterator)
    return inferen_data


def split_dataset(dataset, num_users, data_split_mode):
    data_split = {}
    if data_split_mode == 'iid':
        data_split['train'], label_splid = iid(dataset['train'], num_users)
        data_split['test'], _ = iid(dataset['test'], num_users)
    elif data_split_mode == 'non-iid':
        data_split['train'], label_splid = non_iid(dataset['train'], num_users)
        data_split['test'], _ = non_iid(dataset['test'], num_users)
    return data_split, label_splid


# data_split: dict{0-num_users: [list]}  label_split: [[list]]
def non_iid(dataset, num_users, shard_per_user):
    setup_seed(2023)
    data_split = {i: [] for i in range(num_users)}
    label_split = []
    shard_per_class = shard_per_user * num_users // len(dataset.classes)
    label_idx_split = {}
    label = dataset.target
    for i in range(len(label)):
        label_i = label[i]
        if label_i not in label_idx_split:
            label_idx_split[label_i] = []
        label_idx_split[label_i].append(i)
    for label_i in label_idx_split:
        label_idx = label_idx_split[label_i]
        num_leftover = len(label_idx) % shard_per_class
        # leftover = label_idx[-num_leftover:] if num_leftover!=0 else []
        new_label_idx = label_idx[:-num_leftover] if num_leftover != 0 else label_idx
        label_idx_split[label_i] = np.array(new_label_idx).reshape(shard_per_class, -1).tolist()
    if not label_split:
        label_split = list(range(len(dataset.classes))) * shard_per_class
        label_split = torch.tensor(label_split)[torch.randperm(len(label_split))]
        label_split = label_split.reshape(num_users, -1).tolist()
        for i in range(len(label_split)):
            label_split[i] = torch.tensor(label_split[i]).unique().tolist()
    for i in range(num_users):
        for label_i in label_split[i]:
            idx = torch.arange(len(label_idx_split[label_i]))[torch.randperm(len(label_idx_split[label_i]))[0]].item()
            # print(idx)
            data_split[i].extend(label_idx_split[label_i].pop(idx))
    print(data_split[0])
    print(label_split)
    exit(0)
    return data_split, label_split


def train_data_split(data_train_split, frc):
    setup_seed(2023)
    train_split = {}
    test_split = {}

    for user, data_train in data_train_split.items():
        # print(data_train)
        selected_data = np.random.choice(np.array(data_train), size=int(frc * len(data_train)), replace=False)
        remained_data = np.setdiff1d(np.array(data_train), selected_data)
        train_split[user] = selected_data.tolist()
        test_split[user] = remained_data.tolist()

    with open("./output/test_data_idx/test_data_idx.txt", "w") as f:
        for user in test_split:
            f.write(str(test_split[user]))
            f.write('\n')
    # exit(0)
    return train_split, test_split


def non_iid2(dataset, num_users, shard_per_user, dist_num):  # dist_num为分布数量
    setup_seed(2023)
    data_split = {i: [] for i in range(num_users)}
    label_split = []
    shard_per_class = (dist_num * shard_per_user // len(dataset.classes)) * (num_users // dist_num)
    shard_per_dist = (dist_num * shard_per_user // len(dataset.classes))
    class_dist_remain_times = {i: shard_per_dist for i in range(len(dataset.classes))}
    label_idx_split = {}
    label = dataset.target
    for i in range(len(label)):
        label_i = label[i]
        if label_i not in label_idx_split:
            label_idx_split[label_i] = []
        label_idx_split[label_i].append(i)
    for label_i in label_idx_split:
        label_idx = label_idx_split[label_i]
        num_leftover = len(label_idx) % shard_per_class
        # leftover = label_idx[-num_leftover:] if num_leftover!=0 else []
        new_label_idx = label_idx[:-num_leftover] if num_leftover != 0 else label_idx
        label_idx_split[label_i] = np.array(new_label_idx).reshape(shard_per_class, -1).tolist()

    dist_split = []  # 分布的种类

    while len(dist_split) < dist_num:
        label_idx = np.arange(len(dataset.classes)).tolist()
        for i in range(len(dataset.classes)):  # 去除已经没有的元素
            if class_dist_remain_times[i] == 0:
                label_idx.remove(i)

        dist = []
        while len(dist) < shard_per_user:
            n = random.randint(0, len(label_idx) - 1)
            dist.append(label_idx[n])
            if len(label_idx) > 1:
                label_idx.pop(n)

        dist.sort()
        if dist in dist_split:
            continue

        dist_split.append(dist)
        for val in dist:
            class_dist_remain_times[val] -= 1

    # print(dist_split)
    # exit(0)

    if not label_split:
        label_split = np.tile(dist_split, (num_users // dist_num, 1))
        np.random.shuffle(label_split)
        # label_split = torch.tensor(label_split)[torch.randperm(len(label_split))]
        # label_split = label_split.reshape(num_users, -1).tolist()
        # for i in range(len(label_split)):
        #     label_split[i] = torch.tensor(label_split[i]).unique().tolist()
    for i in range(num_users):
        for label_i in label_split[i]:

            idx = torch.arange(len(label_idx_split[label_i]))[torch.randperm(len(label_idx_split[label_i]))[0]].item()
            # print(idx)
            data_split[i].extend(label_idx_split[label_i].pop(idx))
    # print(data_split[0])
    # print(label_split)
    # exit(0)
    return data_split, label_split


def dataset_class_list(dataset):
    class_list = {i: [] for i in range(len(dataset.classes))}
    label = dataset.target
    for i in range(len(label)):
        label_i = label[i]
        class_list[label_i].append(i)
    return class_list


def iid(dataset, num_users):
    pass


class SplitDataset(Dataset):
    def __init__(self, dataset, idx):
        super().__init__()
        self.dataset = dataset
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        return self.dataset[self.idx[index]]


# dataloader API
def make_dataloader(dataset, batch_size=16, shuffle=True):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


