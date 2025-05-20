from .base import BaseModel
from .sasrec_modules.sasrec import SASRec

import torch.nn as nn


class SASRecModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.sasrec = SASRec(args)
        self.out = nn.Linear(self.sasrec.hidden, args.num_items + 1)

    @classmethod
    def code(cls):
        return 'sasrec'

    def forward(self, x):
        x = self.sasrec(x)
        return self.out(x) 