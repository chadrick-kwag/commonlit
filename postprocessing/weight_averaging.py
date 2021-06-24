"""
average weights in multiple weights
input arg will be dirpath containing weight files (named as *.pt)
and the weight files should be more than one.
"""

import argparse, os, datetime, torch, glob, json
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('weightdir', type=str, help='dirpath containing weights')

args = parser.parse_args()

files = glob.glob(os.path.join(args.weightdir, '*.pt'))

assert len(files) > 1, 'weight file count <=1'

avg_sd = None
stack_count = 0

for f in tqdm(files):

    sd = torch.load(f, map_location='cpu')

    if avg_sd is None:
        avg_sd = sd
    else:
        for k,v in avg_sd.items():
            v = v* stack_count + sd[k]
            v = v / (stack_count+1)

            avg_sd[k] = v

    stack_count+=1




timestamp=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
outputdir = f'testoutput/weight_averaging/{timestamp}'
os.makedirs(outputdir)


savepath = os.path.join(outputdir, f'{timestamp}_avg_weight.pt')

torch.save(avg_sd, savepath)

# save info

savejson = {
    'weight_dir': args.weightdir,
    'weight_files': files
}


savepath = os.path.join(outputdir, 'info.json')

with open(savepath, 'w') as fd:
    json.dump(savejson, fd, indent=4, ensure_ascii=False)


print('done')