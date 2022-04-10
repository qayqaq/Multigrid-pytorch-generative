
class Config:
    def __init__(self, batchsize=100, learning_rate=0.003, beta1=0.5, weight_decay=0.0001, Langevin_T=90, delta=0.003, path='./data/cifar10', sample_dir='sample', epochs=500):
        self.batchsize = batchsize
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.weight_decay = weight_decay
        self.Langevin_T = Langevin_T
        self.delta = delta
        self.path = path
        self.sample_dir = sample_dir
        self.epochs = epochs


# config = Config()
