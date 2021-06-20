import os, sys, torch, datetime, numpy as np, collections

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




train_file_list = [
    '/home/chadrick/prj/kaggle/commonlit/data/train.csv'
]

valid_file_list = [ 
    '/home/chadrick/prj/kaggle/commonlit/datamanage/testoutput/sample_dataset/210613_2204/sample.csv'
]

maxlength = 256




tokenizer = RobertaTokenizer.from_pretrained("/home/chadrick/prj/kaggle/commonlit/data/hfmodels/roberta-base")



model = Model_v1.from_pretrained('/home/chadrick/prj/kaggle/commonlit/data/hfmodels/roberta-base')

train_dataset = TrainDataset(train_file_list, tokenizer, maxlength)
valid_dataset = TrainDataset(valid_file_list, tokenizer, maxlength)

train_dataloader = DataLoader(train_dataset, batch_size=4)
valid_dataloader = DataLoader(valid_dataset, batch_size=4 )


timestamp=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
outputdir = f'ckpt/train/{timestamp}'
os.makedirs(outputdir)


log_dir = os.path.join(outputdir, 'logs')
os.makedirs(log_dir)

writer = SummaryWriter(log_dir)


mean_mse_save_receiver = MetricTrackSaveReceiver(model, os.path.join(outputdir, 'valid_mean_mse_save'), 'mse', 'min')

valid_callback = ValidationCallback(model, valid_dataloader, 'valid', mse_subscribers=[mean_mse_save_receiver], writer=writer)


periodic_save_callback = ManualSaveCallback(model, os.path.join(outputdir, 'periodic_save'))


optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, 1e-7)



gpu=0

if gpu is None or gpu=='cpu':
    device = torch.device('cpu')
else:
    device = torch.device(f'cuda:{gpu}')


epochs = 10000
run_valid_interval = 100
periodic_save_interval = 100
loss_log_interval = 2

scheduler_interval = 10



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


        loss = torch.log(torch.square(outputs - score)).mean()

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
