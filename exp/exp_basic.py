import tensorflow as tf
import os
import torch

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args


    def _build_model(self):
        raise NotImplementedError
        return None

    def _get_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            # fixme change api to tf
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            # fixme
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def train(self):
        pass

    def valid(self):
        pass

    def test(self):
        pass