from .base_engine import BaseTrainer
from .e2e import E2E

def build_engine(args,device):
    # Init engine
    engine = E2E(args, device)
    # Init trainer
    trainer = BaseTrainer(engine=engine, args=args)
    return trainer.train, trainer.validate
