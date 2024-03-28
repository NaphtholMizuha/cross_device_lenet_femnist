import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.functional as F

class AutoEncoder(nn.Cell):
    def __init__(self, in_feature, hidden_feature, out_feature, dropout=0.2):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.SequentialCell(
            nn.Dense(in_feature, hidden_feature),
            nn.ReLU(),
            nn.Dropout(keep_prob=dropout),
            nn.Dense(hidden_feature, out_feature)
        )
        
        self.decoder = nn.SequentialCell(
            nn.Dense(out_feature, hidden_feature),
            nn.ReLU(),
            nn.Dropout(keep_prob=dropout),
            nn.Dense(hidden_feature, in_feature)
        )
        
    def construct(self, x):
        z = self.encoder(x)
        x_recover = self.decoder(z)
        return x_recover

'''
if __name__ == "__main__":
    np.random.seed(0)
    network = AutoEncoder(105,32,8)
    criterion = nn.MSELoss(reduction="mean")
    net_opt = nn.Adam(network.trainable_params(), learning_rate=0.01, weight_decay=1e-5)
    net_with_criterion = nn.WithLossCell(network, criterion)
    train_network = nn.TrainOneStepCell(net_with_criterion, net_opt)
    train_network.set_train()

    data = ms.Tensor(np.random.rand(16,105).astype(np.float32))
    label = data
    ms.export(train_network, data, label, file_name='autoencoder_train.mindir', file_format='MINDIR')
'''


