import torch, numpy as np
from torch.utils.tensorboard import SummaryWriter

class ValidationCallback:

    def __init__(self, model, dataloader, writer: SummaryWriter = None, mse_subscribers = [], name=None):

        self.model = model
        self.dataloader = dataloader
        self.writer = writer

        self.mse_subscribers = mse_subscribers

    def run(self, global_step):

        self.model.eval()

        # calculate mean mse

        device = self.model.device

        mse_list = []

        for data in self.dataloader:



            input_data_keys = ['input_ids', 'attention_mask']

            input_data = {}

            for k in input_data_keys:
                v = data[k]
                input_data[k] = v.to(device)

            with torch.no_grad():

                outputs = self.model(**input_data)

            # calculate mse

            pred_score_arr = outputs.cpu().numpy()

            gt_score_arr = data['score'].unsqueeze(-1).numpy()

            mse_arr = np.sqrt(np.square(gt_score_arr - pred_score_arr))

            mse_list.extend(mse_arr.tolist())


        self.model.train()

        mean_mse = np.array(mse_list).mean()



        for a in self.mse_subscribers:
            a.notify(mean_mse, global_step)

        if self.writer is not None:

            self.writer.add_scalar(f'{self.name}_valid/mean_mse', mean_mse)
            self.writer.flush()
