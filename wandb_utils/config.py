import wandb

class Config:
    def __init__(self, epoch, batch_size):
        self.epoch = epoch
        self.batch_size = batch_size

        self.run = wandb.init(
            project="Detection of plant diseases",
            config={
                "epoch": epoch,
                "batch_size": batch_size,
            }
        )

    def config(self):
        return self.run.config

    def finish(self):
        self.run.config.finish()


