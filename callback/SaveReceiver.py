import os, torch
from torch.nn import Module


class ManualSaveReceiver:
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

    def notify(self, val, global_step):


        savepath = os.path.join(self.savedir, f'step={global_step}.pt')

        torch.save(self.model.state_dict(), savepath)

        if self.keep_count is not None:

            self.keep_history.append(savepath)

            if len(self.keep_history) > self.keep_count:

                # remove first

                first_path = self.keep_history[0]

                os.remove(first_path)

                self.keep_history = self.keep_history[1:]


                


class MetricTrackSaveReceiver:
    def __init__(self, model, savedir, metric_name, mode, keep_count=3):


        assert model is not None

        self.model = model


        if not os.path.exists(savedir):
            os.makedirs(savedir)
        
        self.savedir = savedir

        assert metric_name is not None and metric_name != "", 'invalid metric_name'

        self.metric_name = metric_name


        assert mode in ['min', 'max'], f'invalide mode={mode}'

        self.mode = mode


        assert keep_count is None or isinstance(keep_count, int)

        if isinstance(keep_count, int):
            assert keep_count > 0, 'keep_count should be > 0'

        self.keep_count = keep_count

        self.keep_history = []
        self.prev_val = None


    def notify(self, val, global_step):



        if self.keep_count is None:

            append_flag = False

            if self.prev_val is None:
                self.prev_val = val
                append_flag = True
            else:

                if val > self.prev_val and self.mode == 'max':
                    append_flag = True
                
                if val < self.prev_val and self.mode == 'min':
                    append_flag = True
            

            if append_flag:

                savepath = os.path.join(self.savedir, f'{self.metric_name}={val}_global_step={global_step}.pt')
                torch.save(self.model.state_dict(), savepath)

        else:
            # need to track history

            # check if okay to add to history

            # check if current val triggers save

            append_flag = False

            if len(self.keep_history)==0:
                append_flag= True
            else:


                for a in self.keep_history:


                    if val < a[0] and self.mode == 'min':
                        append_flag = True
                        break

                    if val > a[0] and self.mode == 'max':
                        append_flag = True
                        break
            
                
            if append_flag:

                savepath = os.path.join(self.savedir, f'{self.metric_name}={val}_global_step={global_step}.pt')

                torch.save(self.model.state_dict(), savepath)

                self.keep_history.append((val, savepath))

                # remove overflow if exists
                
                if len(self.keep_history) > self.keep_count:

                    if self.mode == 'max': 
                        reverse = True
                    else:
                        reverse = False

                    self.keep_history.sort(key=lambda x: x[0], reverse=reverse)

                    last_path = self.keep_history[-1][1]

                    os.remove(last_path)

                    self.keep_history = self.keep_history[:-1]
