"""
split given dataset into train and valid csv file
"""

import csv, datetime, os, json, argparse, random

random.seed()


parser = argparse.ArgumentParser()

parser.add_argument('-datafile', type=str, help='path to data csv file')
parser.add_argument('-validratio', type=float, help='ratio of dataset to set as validation data')

args = parser.parse_args()

f = args.datafile

assert os.path.exists(f), f'{f} not exist'
assert args.validratio > 0 and args.validratio < 1, f'invalid valid ratio={args.validratio}'




with open(f, 'r') as fd:
    reader = csv.reader(fd)
    lines = list(reader)

firstline = lines[0]
other_lines = lines[1:]


valid_size = int(len(other_lines) * args.validratio)

assert valid_size > 0, 'valid size is not >0'

index_list = list(range(len(other_lines)))
is_train_bool_list = [True] * len(other_lines)

valid_index_list = random.sample(index_list, valid_size)

for i in valid_index_list:
    is_train_bool_list[i] = False

train_lines = []
valid_lines = []


for i, v in enumerate(is_train_bool_list):
    if v:
        train_lines.append(other_lines[i])
    else:
        valid_lines.append(other_lines[i])


timestamp=datetime.datetime.now().strftime("%y%m%d_%H%M")
outputdir = f'testoutput/split_train_and_valid/{timestamp}'
os.makedirs(outputdir)


train_file = os.path.join(outputdir, 'train.csv')
valid_file = os.path.join(outputdir, 'valid.csv')


save_lines = [firstline] + train_lines

with open(train_file, 'w') as fd:
    writer = csv.writer(fd, delimiter=',')

    for l in save_lines:
        writer.writerow(l)

save_lines = [firstline] + valid_lines


with open(valid_file, 'w') as fd:
    writer = csv.writer(fd, delimiter=',')

    for l in save_lines:
        writer.writerow(l)
    

savejson = vars(args)
savejson['train_size'] = len(train_lines)
savejson['valid_size'] = len(valid_lines)

savepath = os.path.join(outputdir, 'info.json')

with open(savepath, 'w') as fd:
    json.dump(savejson, fd, indent=4, ensure_ascii=False)

print('done')
