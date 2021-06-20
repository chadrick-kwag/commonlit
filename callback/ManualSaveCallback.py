from torch.nn import Module
import os, torch


class ManualSaveCallback:
    def __init__(self, model: Module, savedir, keep_count=3):

        assert model is not None

        self.model = model

        if not os.path.exists(savedir):
            os.makedirs(savedir)
        
        self.savedir = savedir

        assert keep_count is None or isinstance(keep_count, int)

        if isinstance(keep_count, int):
            assert keep_count > 0, 'keep_count should be > 0'

        self.keep_count = keep_count

        self.keep_history=[]

    def run(self, global_step):


        savepath = os.path.join(self.savedir, f'step={global_step}.pt')

        torch.save(self.model.state_dict(), savepath)

        if self.keep_count is not None:

            self.keep_history.append(savepath)

            if len(self.keep_history) > self.keep_count:

                # remove first

                first_path = self.keep_history[0]

                os.remove(first_path)

                self.keep_history = self.keep_history[1:]
