# @Time    : 2022/7/18 4:23 PM
# @Author  : Heshuai
# @Email   : heshuai.sec@gmail.com
from abc import abstractmethod, ABCMeta


class MetricsInterface(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        self.reset()

    @abstractmethod
    def reset(self):
        return

    @abstractmethod
    def get_scores(self, verbose=False):
        scores = None
        if verbose: print(scores)
        return scores

    @abstractmethod
    def update(self, pred, mask):
        return

    @abstractmethod
    def test_folder(self, pred_f: str, gt_f: str, verbose: bool = False, *args, **kwargs):
        scores = None
        if verbose: print(scores)
        return scores
