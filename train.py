import os, sys, torch, datetime, numpy as np, collections

from torch._six import string_classes

from torch.utils.data.dataloader import DataLoader
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from tqdm import tqdm
from dataprovider.dataset import TrainDataset


# sys.path.append(os.path.abspath(".."))




train_file_list = [
    '/home/chadrick/prj/kaggle/commonlit/data/train.csv'
]

valid_file_list = [ 
    '/home/chadrick/prj/kaggle/commonlit/datamanage/testoutput/sample_dataset/210613_2204/sample.csv'
]

maxlength = 256




tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


class Model(torch.nn.Module):

    def __init__(self, hiddensize):
        super().__init__()

        self.base_model = RobertaModel.from_pretrained('roberta-base')
        self.head_dense = torch.nn.Linear(hiddensize, 1)

    def forward(self,**kwargs):

        y = self.base_model(**kwargs)

        y = y.pooler_output

        y = self.head_dense(y)

        return y


model = Model(768)

train_dataset = TrainDataset(train_file_list, tokenizer, maxlength)
valid_dataset = TrainDataset(valid_file_list, tokenizer, maxlength)

train_dataloader = DataLoader(train_dataset, batch_size=8 )
valid_dataloader = DataLoader(valid_dataset, batch_size=8)

optimzier = torch.optim.Adam(model.parameters(), lr=1e-4)




# timestamp=datetime.datetime.now().strftime("%y%m%d_%H%M")
# outputdir = f'testoutput/sample_dataset/{timestamp}'
# os.makedirs(outputdir)




epochs = 100
run_valid_interval = 2
periodic_save_interval = 2



global_step = 0

for epoch_index in range(epochs):

    epoch_step = 0


    for data in train_dataloader:

        global_step+=1
        epoch_step+=1

        optimzier.zero_grad()

        
        encoding = data['encoding']
        score = data['score']
        id_str = data['id_str']

        


        outputs = model(**encoding)

        loss = torch.log(torch.square(outputs - score)).mean()

        

        loss.backward()

        optimzier.step()

        if global_step % run_valid_interval == 0:

            model.eval()

            mse_list = []
            
            for data in valid_dataloader:

                encoding, score, std = data

                with torch.no_grad():
                    outputs = model(**encoding)

                logits = outputs.logits.cpu().numpy()

                mse_arr = (np.square(logits - score)).mean(axis=-1)

                mse_list.extend(mse_arr.tolist())
            

            mean_mse = sum(mse_list) / len(mse_list)

            print(f"mean mse: {mean_mse}")

            model.train()

