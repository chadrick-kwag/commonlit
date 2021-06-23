import os, sys, torch, datetime, numpy as np, collections, argparse, yaml, copy, shutil
from munch import munchify

from torch._six import string_classes

from torch.utils.data.dataloader import DataLoader
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from tqdm import tqdm


from torch.utils.tensorboard import SummaryWriter


sys.path.append(os.path.abspath(".."))

from model.v1 import Model_v1
from dataprovider.dataset import TrainDataset

from callback.validcallback import ValidationCallback
from callback.ManualSaveCallback import ManualSaveCallback
from callback.SaveReceiver import MetricTrackSaveReceiver



parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='config file path')

args = parser.parse_args()

with open(args.config, 'r') as fd:
    config = yaml.load(fd)

config = munchify(config)





tokenizer = RobertaTokenizer.from_pretrained(config.tokenizer_dirpath)


if config.ckpt == 'pretrain':

    model = Model_v1.from_pretrained(config.pretrain_dir)
elif config.ckpt is not None:
    model = Model_v1.from_pretrained(None, config= config.model_config_file, state_dict=torch.load(config.ckpt))
else:
    # init from scratch
    model = Model_v1.from_pretrained(None, config= config.model_config_file, state_dict={})

train_dataset = TrainDataset(config.train_file_list, tokenizer, config.maxlength)
valid_dataset = TrainDataset(config.valid_file_list, tokenizer, config.maxlength)

train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size)
valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size)


timestamp=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
outputdir = f'ckpt/train/{timestamp}_{config.suffix}'
os.makedirs(outputdir)

# copy used config file

savepath = os.path.join(outputdir, 'usedconfig.yaml')
shutil.copy2(args.config, savepath)


log_dir = os.path.join(outputdir, 'logs')
os.makedirs(log_dir)

writer = SummaryWriter(log_dir)


mean_mse_save_receiver = MetricTrackSaveReceiver(model, os.path.join(outputdir, 'valid_mean_mse_save'), 'mse', 'min')

valid_callback = ValidationCallback(model, valid_dataloader, 'valid', mse_subscribers=[mean_mse_save_receiver], writer=writer)


periodic_save_callback = ManualSaveCallback(model, os.path.join(outputdir, 'periodic_save'))

if config.optimizer.type == 'adam':
    kwargs = vars(config.optimizer)
    del kwargs['type']

    optimizer = torch.optim.Adam(model.parameters(), **kwargs)
else:
    raise Exception(f'not supported optimizer type: {config.optimizer.type}')



if config.scheduler.type == 'CosineAnnealingLR':
    kwargs = vars(config.scheduler)
    del kwargs['type']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
else:
    raise Exception(f'not supported scheduler type: {config.scheduler.type}')



gpu=0

if gpu is None or gpu=='cpu':
    device = torch.device('cpu')
else:
    device = torch.device(f'cuda:{gpu}')


epochs = config.epochs
run_valid_interval = config.run_valid_interval
periodic_save_interval = config.periodic_save_interval
loss_log_interval = config.loss_log_interval

scheduler_interval = config.scheduler_interval



global_step = 0

model.to(device)
model.train()


for epoch_index in range(epochs):

    epoch_step = 0


    for data in train_dataloader:



        global_step+=1
        epoch_step+=1

        optimizer.zero_grad()

        
        
        # move input data to device

        input_data_keys = ['input_ids', 'attention_mask']

        input_data = {}

        for k in input_data_keys:
            v = data[k]
            input_data[k] = v.to(device)
        

        outputs = model(**input_data)

        score = data['score'].to(device).unsqueeze(-1)


        # loss = torch.log(torch.square(outputs - score)).mean()
        loss = torch.square(outputs - score).mean()

        if loss_log_interval is not None and global_step % loss_log_interval ==0:
            print(f"loss={loss.item()}")
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.flush()

        loss.backward()

        optimizer.step()


        if global_step % run_valid_interval == 0:

            valid_callback.run(global_step)


        if scheduler_interval is not None and global_step % scheduler_interval == 0 :
            scheduler.step()

        if periodic_save_interval is not None and global_step % periodic_save_interval==0:
            periodic_save_callback.run(global_step)
