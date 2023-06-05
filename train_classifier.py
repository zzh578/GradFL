import copy

import torch
import random
import os
import yaml
import wandb
import torch.nn.functional as F
from collections import OrderedDict

from config import cfg
from utils import set_control, set_seed, get_model, make_scheduler, make_optimizer, generate_submodel_rate, generate_fix_submodel_rate_list
from utils import load_params_to_client_model
from data import get_transform, get_dataset, SplitDataset, non_iid, make_dataloader, get_inferen_data, dataset_class_list


def main():
    set_control(cfg)
    set_seed(cfg['seed'])
    print('Config is OK!\n {}'.format(cfg))
    device = torch.device(cfg['device'])

    # use wandb to record
    if cfg['use_wandb']:
        wandb.init(
            project=cfg['project_name'],
            group=cfg['group_name'],
            config=cfg
        )
        print("Wandb is inited!")

    # global_dataset and non-iid progress
    transform = get_transform(cfg['dataset'])
    dataset = get_dataset(cfg['dataset'], transform)
    print('Dataset {} is OK!'.format(cfg['dataset']))
    data_train_split, label_train_split = non_iid(dataset['train'], cfg['numbers'], cfg['shardperuser'])

    # global_model
    global_model = get_model(cfg['model_name'], dataset, mode_rate=1)
    # global_model = global_model.to(device)

    # optimizer
    optimizer = make_optimizer(global_model, cfg['lr'])

    # scheduler
    scheduler = make_scheduler(optimizer)

    # fix mode sub_rate list
    sub_model_rate_list = generate_fix_submodel_rate_list()

    # for grad-base mode we need konw class-list
    class_list = dataset_class_list(dataset['train'])

    print('Experiment is beginning!')
    for i in range(cfg['rounds']):
        print(' ROUNDS {}'.format(i+1), end=' ')
        num_activate_users = int(cfg['numbers'] * cfg['frc'])
        user_idxs = torch.randperm(cfg['numbers'])[:num_activate_users]
        clients_models_params = {}
        clients_models_shape = {}
        for user_idx in user_idxs:
            # sub_dataset
            client_dataset = SplitDataset(dataset['train'], data_train_split[int(user_idx)])
            client_dataloader = make_dataloader(client_dataset, batch_size=cfg['batchsize'])
            label_mask = torch.zeros(len(dataset['train'].classes), device=device)
            label_mask[label_train_split[user_idx]] = 1

            # sub_model
            rate = generate_submodel_rate(cfg['client_state'], sub_model_rate_list, user_idx)
            client_model = get_model(cfg['model_name'], dataset, mode_rate=rate)
            inferen_data = get_inferen_data(cfg['mode'], cfg['inferen_batch'], class_list, label_train_split[user_idx],
                                            client_dataset, dataset['train'])
            load_params_to_client_model(global_model, client_model, inferen_data, cfg['mode'], device, rate,
                                        cfg['select_mode'], label_mask, clients_models_shape, user_idx)

            # train model
            run_train(client_model, client_dataloader, label_mask, device, optimizer, clients_models_params, user_idx)
        combine_global_model_from_clients(global_model, user_idxs, clients_models_shape, clients_models_params)
        run_global_test(global_model, dataset, device)
        scheduler.step()
    if cfg['user_wandb']:
        wandb.finish()


def run_train(client_model, client_dataloader, label_mask, device, optimizer, clients_models_params, user_idx):
    client_model.to(device)
    lr = optimizer.param_groups[0]['lr']
    local_optimizer = make_optimizer(client_model, lr)
    client_model.train()
    local_loss = 0
    for local_epoch in range(cfg['epoch']):
        for batch in client_dataloader:
            img, target = batch['img'], batch['label']
            img, target = img.to(device), target.to(device)
            client_model.zero_grad()
            output = client_model(img)
            output = output.masked_fill(label_mask == 0, 0)
            loss = F.cross_entropy(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(client_model.parameters(), 1)
            local_optimizer.step()
            # 将loss相加
            local_loss += loss
    # print('     user_idx {}: Local Loss {}'.format(user_idx, local_loss))
    client_model.cpu()
    clients_models_params[int(user_idx)] = copy.deepcopy(client_model.state_dict())


def combine_global_model_from_clients(global_model, user_idxs, clients_models_shape, clients_models_params):
    # 将clients集合赋值给global model
    client_num_model_param = OrderedDict()
    global_temp_params = OrderedDict()
    for k, v in global_model.state_dict().items():
        client_num_model_param[k] = torch.ones_like(v)
        global_temp_params[k] = copy.deepcopy(v)
    for user_idx in user_idxs:
        for k, v in clients_models_params[int(user_idx)].items():
            temp_shape = clients_models_shape[int(user_idx)][k]
            if k in global_temp_params:
                if v.dim() > 1:
                    global_temp_params[k][torch.meshgrid(temp_shape, indexing='ij')] += v
                    client_num_model_param[k][torch.meshgrid(temp_shape, indexing='ij')] += 1
                else:
                    global_temp_params[k][temp_shape] += v
                    client_num_model_param[k][temp_shape] += 1
            else:
                raise NameError('{} not in global_temp_params'.format(k))
    # 获得取平均值之后的global_params
    for k in global_temp_params:
        global_temp_params[k] /= client_num_model_param[k]
    global_model.load_state_dict(global_temp_params)


def run_global_test(global_model, dataset, device):
    # 测试global model
    test_dataloader = make_dataloader(dataset['test'], batch_size=32)
    correct, total = 0, 0
    global_model.to(device)
    global_model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            img, target = batch['img'], batch['label']
            img, target = img.to(device), target.to(device)
            output = global_model(img)
            loss = F.cross_entropy(output, target)
            _, predicted = torch.max(output, dim=1)
            test_loss += loss.data
            total += img.shape[0]
            correct += (predicted == target).sum().item()
    print(' Global Model Acc: {}, Test Loss: {}'.format(correct / total, test_loss))
    if cfg['use_wandb']:
        wandb.log({'acc': correct / total, 'loss': test_loss})
    global_model.cpu()


if __name__ == "__main__":
    main()
