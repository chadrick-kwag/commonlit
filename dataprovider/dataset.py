import torch, csv



def get_data_from_csv_file(f):

    file = '/home/chadrick/prj/kaggle/commonlit/data/train.csv'


    with open(f, 'r') as fd:
        reader = csv.reader(fd)

        lines = list(reader)

    data = [(a[0], a[3], float(a[4])) for a in lines[1:]] #id, text, score

    return data




def tokenize_text(text, tokenizer, maxlength):

    encoding = tokenizer(text, padding='max_length', max_length=maxlength, truncation=True, return_tensors='pt')

    return encoding


class TrainDataset(torch.utils.data.IterableDataset):

    def __init__(self, file_list, tokenizer, maxlength):

        super().__init__()

        self.file_list = file_list
        self.tokenizer = tokenizer
        self.maxlength = maxlength

        self.data = []
        
        for f in self.file_list:
            
            _data = get_data_from_csv_file(f)
            self.data.extend(_data)
        

    def __len__(self):
        return len(self.data)

    def __iter__(self):

        self.num = 0

        return self

    def __next__(self):

        if self.num >= len(self):
            raise StopIteration
        
        return_data = self.data[self.num]

        id_str, text, score = return_data

        encoding = tokenize_text(text, self.tokenizer, self.maxlength)

        # encoding['labels'] = score

        _encoding = {
            'input_ids': encoding['input_ids'][0],
            'attention_mask': encoding['attention_mask'][0]
        }

        self.num+=1

        return {
            'encoding': _encoding,
            'score': score,
            'id_str': id_str
        }

        # return [encoding, score, std]


        