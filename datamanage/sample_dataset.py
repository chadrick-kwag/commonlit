"""
sample data from csv file
"""


import argparse, csv, datetime, os, random, sys

random.seed()


sys.path.append(os.path.abspath('..'))


parser = argparse.ArgumentParser()

parser.add_argument('datafile', type=str)

args = parser.parse_args()

f = args.datafile

assert os.path.exists(f), f'{f} not exist'



timestamp=datetime.datetime.now().strftime("%y%m%d_%H%M")
outputdir = f'testoutput/sample_dataset/{timestamp}'
os.makedirs(outputdir)



with open(f, 'r') as fd:
    reader = csv.reader(fd)

    lines = list(reader)

firstline = lines[0]
other_lines = lines[1:]

sample_size = 10

sampled_lines = random.sample(other_lines, sample_size)


save_lines = [firstline] + sampled_lines
# print(save_lines)


savepath  = os.path.join(outputdir, 'sample.csv')

with open(savepath, 'w') as fd:
    writer = csv.writer(fd, delimiter=',')

    for l in save_lines:
        writer.writerow(l)
    



    

    