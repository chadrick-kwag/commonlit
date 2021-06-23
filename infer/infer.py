import torch, yaml, os, datetime, argparse, sys, json
from munch import munchify
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from tqdm import tqdm

sys.path.append(os.path.abspath('..'))

from model.v1 import Model_v1
from dataprovider.dataset import PredictDataset



parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='config file path')

args = parser.parse_args()


with open(args.config, 'r') as fd:
    config = yaml.load(fd)

config = munchify(config)


model = Model_v1.from_pretrained(None, config=config.model_config_file, state_dict = torch.load(config.ckpt))


if config.gpu == 'cpu' or config.gpu is None:
    device = torch.device('cpu')
else:
    device = torch.device(f'cuda:{config.gpu}')

model.to(device)
model.eval()



tokenizer = RobertaTokenizer.from_pretrained(config.tokenizer_dirpath)

dataset = PredictDataset([config.predict_file],tokenizer, config.maxlength)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)


result = {}

for data in dataloader:


    id_str_list = data['id_str']

    input_data_keys = ['input_ids', 'attention_mask']

    input_data = {}

    for k in input_data_keys:
        v = data[k]
        input_data[k] = v.to(device)
    
    with torch.no_grad():
        outputs = model(**input_data)

    outputs = outputs.cpu().detach().numpy()

    for id_str, score in zip(id_str_list, outputs):
        result[id_str] = float(score[0])



timestamp=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
outputdir = f'testoutput/infer/{timestamp}'
os.makedirs(outputdir)

savepath = os.path.join(outputdir, 'result.json')


with open(savepath, 'w') as fd:
    json.dump(result, fd, indent=4, ensure_ascii=False)


