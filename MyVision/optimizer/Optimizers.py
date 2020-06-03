import torch.optim as optim


class Optimizer:
    def __init__(self):
        self.optimizers = {"SGD": self._sgd, "Adam": self._adam, "AdamW": self._adamw}

    def __call__(self, optimizer, learning_rate, momentum, parameters, weight_decay):
        if optimizer not in self.optimizers:
            raise Exception("Optimizer not implemented")
        else:
            if momentum is None:
                return self.optimizers[optimizer](
                    parameters, learning_rate, weight_decay
                )
            else:
                return self.optimizers[optimizer](
                    parameters, learning_rate, weight_decay, momentum
                )

    @staticmethod
    def _sgd(parameters, learning_rate, weight_decay, momentum):
        return optim.SGD(
            params=parameters,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    @staticmethod
    def _adam(parameters, learning_rate, weight_decay):
        return optim.Adam(
            params=parameters, lr=learning_rate, weight_decay=weight_decay
        )

    @staticmethod
    def _adamw(parameters, learning_rate, weight_decay):
        return optim.AdamW(
            params=parameters, lr=learning_rate, weight_decay=weight_decay
        )
