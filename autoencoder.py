import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.functional as F

from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.parallel._utils import (_get_device_num,
                                       _get_enable_parallel_optimizer,
                                       _get_gradients_mean, _get_parallel_mode,
                                       _is_pynative_parallel)

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
        return x_recover, z
    

class LossNetwork(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(LossNetwork, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
        if backbone.jit_config_dict:
            self._jit_config_dict = backbone.jit_config_dict

    def construct(self, data, label):
        out, emb = self._backbone(data)
        loss = self._loss_fn(out, label)
        return loss
       
    @property
    def backbone_network(self):
        return self._backbone


class Wrapper(nn.Cell):
    def __init__(self, net_with_loss, criterion, optimizer, sens=1.0):
        super(Wrapper, self).__init__(auto_prefix=False)
        self.net_with_loss = net_with_loss
        self.net_with_loss.set_grad()
        self.criterion = criterion
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        self.reducer_flag = self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL) or \
                            _is_pynative_parallel()
        if self.reducer_flag:
            self.mean = _get_gradients_mean()
            self.degree = _get_device_num()
            from mindspore.communication.management import GlobalComm
            group = GlobalComm.WORLD_COMM_GROUP
            if isinstance(self.optimizer, (nn.AdaSumByGradWrapCell, nn.AdaSumByDeltaWeightWrapCell)):
                from mindspore.communication.management import get_group_size, create_group, get_rank
                group_number = get_group_size() // 8
                self.degree = int(self.degree / group_number)
                group_list = [list(range(x * self.degree, (x + 1) * self.degree)) for x in range(group_number)]
                current_index = get_rank() // 8
                server_group_name = "allreduce_" + str(current_index)
                create_group(server_group_name, group_list[current_index])
                group = server_group_name
            self.grad_reducer = DistributedGradReducer(self.weights, self.mean, self.degree, group=group)
        
    def construct(self, *inputs):
        data = inputs[0]
        loss = self.net_with_loss(*inputs)
        emb = self.net_with_loss._backbone(data)[1]
        sens = F.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.net_with_loss, self.weights)(*inputs, sens)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        return loss, emb