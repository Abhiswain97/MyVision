class ModelLayers:
    def __init__(self, model, freeze_from):
        self.model = model
        self.freeze_from = freeze_from

    def freeze_layers(self):

        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.freeze_from.parameters():
            p.requires_grad = True

        return self.model
