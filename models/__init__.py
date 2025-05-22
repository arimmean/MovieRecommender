from .bert import BERTModel
from .dae import DAEModel
from .vae import VAEModel
from .sasrec import SASRecModel

MODELS = {
    BERTModel.code(): BERTModel,
    DAEModel.code(): DAEModel,
    VAEModel.code(): VAEModel,
    SASRecModel.code(): SASRecModel
}


def model_factory(args):
    model = MODELS[args.model_code]
    return model(args)
