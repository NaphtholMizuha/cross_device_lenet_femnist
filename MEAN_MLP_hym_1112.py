import numpy as np
from mindspore import jit, nn, ops
from utils import *
from mindspore.common.initializer import initializer, XavierNormal


def get_param(shape):
    param = mindspore.Parameter(initializer(XavierNormal(), shape, mindspore.float32))
    return param
    
class MEAN_MLP_hym_1112(nn.Cell):
    def __init__(self, user_num, country_num, device_num, app_num, class_num, init_dim, max_count, max_time, hist, features_len):
        super().__init__()
        ## scalers init
        self.user_num = user_num
        self.country_num = country_num
        self.device_num = device_num
        self.app_num = app_num
        self.class_num = class_num
        self.init_dim = init_dim
        self.hist = hist
        self.features_len = features_len
        # parameters init
        # act func init
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # embedding init
        self.app_emb = get_param((app_num, init_dim))
        self.user_emb = get_param((user_num, init_dim))
        self.country_emb = get_param((country_num, init_dim))
        self.device_emb = get_param((device_num, init_dim))
        self.class_emb = get_param((class_num, init_dim))
        
        self.MAX_COUNT = max_count
        self.MIN_COUNT = 0
        self.max_count_emb = get_param((1, 1, init_dim))
        self.min_count_emb = get_param((1, 1, init_dim))
        
        self.MAX_TIME = max_time
        self.MIN_TIME = 0
        self.max_time_emb = get_param((1, 1, init_dim))
        self.min_time_emb = get_param((1, 1, init_dim))
        
        input_dim = hist
        input_dim += features_len
        input_dim += hist - 1
        input_dim += hist - 1
        input_dim += hist - 1
        input_dim *= init_dim
        shared_layers = [20, 10]
        app_layers = [10]
        # time_layers = [4, 4]
        
        self.Shared_Layer = nn.SequentialCell(
            nn.Dense(input_dim, shared_layers[0]),
            self.act,
            nn.Dense(shared_layers[0], shared_layers[1]),
            self.act
            # ,
            # nn.Dense(shared_layers[1], shared_layers[2]),
            # self.act
        )
        
        self.App_Layers = nn.SequentialCell(
            nn.Dense(shared_layers[-1], app_layers[0]),
            self.act
            # ,
            # nn.Dense(app_layers[0], app_layers[1]),
            # self.act
        )
        
        # self.Time_Layers = nn.SequentialCell(
        #     nn.Dense(shared_layers[-1], time_layers[0]),
        #     self.act,
        #     nn.Dense(time_layers[0], time_layers[1]),
        #     self.act
        # )
        
        self.Final_App_Layer = nn.Dense(app_layers[-1], app_num)
        # self.Final_Time_Layer = nn.Dense(time_layers[-1], app_num)
        
        # init ops
        self.zeros_like = ops.ZerosLike()
        self.concat = ops.Concat(1)
        self.reshape = ops.Reshape()
        self.stack = ops.Stack()
        self.reduce_sum = ops.ReduceSum()
        self.batch_mat_mul = ops.BatchMatMul()
        self.app_emb[0] = self.zeros_like(self.app_emb[0])
        self.class_emb[0] = self.zeros_like(self.class_emb[0])
        
    def construct(self, batch_user, batch_country, batch_device, batch_nums, batch_hist_ids, batch_hist_classes, 
                  batch_hist_counts, batch_hist_times):
        # self.app_emb[0] = self.zeros_like(self.app_emb[0])
        # self.class_emb[0] = self.zeros_like(self.class_emb[0])
        
        variable_embeddings = [self.user_emb[batch_user], self.country_emb[batch_country], self.device_emb[batch_device]]
        sequence_embeddings = []
        # batch_number = batch_nums.unsqueeze(-1)
        sum_embeddings = self.reduce_sum(self.app_emb[batch_hist_ids], 2)
        sum_embeddings /= batch_nums
        sequence_embeddings.append(sum_embeddings)
      
        cur_batch_size = sequence_embeddings[0].shape[0]

        sum_embeddings = self.reduce_sum(self.class_emb[batch_hist_classes], 2)
        sum_embeddings /= batch_nums
        sequence_embeddings.append(sum_embeddings)
        
        # batch_hist_counts = batch_hist_counts.unsqueeze(-1)
        max_rate =  (self.MAX_COUNT-batch_hist_counts).float()/(self.MAX_COUNT-self.MIN_COUNT)
        min_rate =  (batch_hist_counts-self.MIN_COUNT).float()/(self.MAX_COUNT-self.MIN_COUNT)
        
        resutl1 = max_rate*self.max_count_emb.float().squeeze(0)
        result2 = min_rate*self.min_count_emb.float().squeeze(0)
        # resutl1 = self.batch_mat_mul(max_rate, self.max_count_emb.float())
        # result2 = self.batch_mat_mul(min_rate, self.min_count_emb.float())
        
        count_emb = resutl1 + result2
        
        # count_emb = self.int2emb(batch_hist_counts, self.MAX_COUNT, self.MIN_COUNT, self.max_count_emb, self.min_count_emb)
        sum_embeddings = self.reduce_sum(count_emb, 2)
        sum_embeddings /= batch_nums
        sequence_embeddings.append(sum_embeddings)
        
        # batch_hist_times = batch_hist_times.unsqueeze(-1)
        max_rate =  (self.MAX_TIME-batch_hist_times).float()/(self.MAX_TIME-self.MIN_TIME)
        min_rate =  (batch_hist_times-self.MIN_TIME).float()/(self.MAX_TIME-self.MIN_TIME)
        
        resutl1 = max_rate*self.max_time_emb.float().squeeze(0)
        result2 = min_rate*self.min_time_emb.float().squeeze(0)
        # resutl1 = self.batch_mat_mul(max_rate, self.max_time_emb.float())
        # result2 = self.batch_mat_mul(min_rate, self.min_time_emb.float())
        
        time_emb = resutl1 + result2
        # time_emb = self.int2emb(batch_hist_times, self.MAX_TIME, self.MIN_TIME, self.max_time_emb, self.min_time_emb)
        
        sum_embeddings = self.reduce_sum(time_emb, 2)
        sum_embeddings /= batch_nums
        sequence_embeddings.append(sum_embeddings)
       
        input_embeddings = self.concat((self.reshape(self.concat(variable_embeddings), (cur_batch_size, -1)), 
                            self.reshape(self.concat(sequence_embeddings), (cur_batch_size, -1))))
       
        shared_embeddings = self.Shared_Layer(input_embeddings)
        app_output = self.Final_App_Layer(self.App_Layers(shared_embeddings))
        # time_output = self.Final_Time_Layer(self.Time_Layers(shared_embeddings))

        return self.sigmoid(app_output)

    # def construct(self, batch):
    #     # self.app_emb[0] = self.zeros_like(self.app_emb[0])s
    #     # self.class_emb[0] = self.zeros_like(self.class_emb[0])
    #     batch_user, batch_country, batch_device, batch_nums, batch_hist_ids, batch_hist_classes, batch_hist_counts, batch_hist_times, batch_time, batch_mask, batch_label = batch['batch_user'], batch['batch_country'], batch['batch_device'], batch['batch_nums'].unsqueeze(-1), batch['batch_hist_ids'], batch['batch_hist_classes'], batch['batch_hist_counts'].unsqueeze(-1), batch['batch_hist_times'].unsqueeze(-1), batch['batch_time'], batch['batch_mask'], batch['batch_label']
    #     variable_embeddings = [self.user_emb[batch_user], self.country_emb[batch_country], self.device_emb[batch_device]]
    #     sequence_embeddings = []
    #     # batch_number = batch_nums.unsqueeze(-1)
    #     sum_embeddings = self.reduce_sum(self.app_emb[batch_hist_ids], 2)
    #     sum_embeddings /= batch_nums
    #     sequence_embeddings.append(sum_embeddings)
      
    #     cur_batch_size = sequence_embeddings[0].shape[0]

    #     sum_embeddings = self.reduce_sum(self.class_emb[batch_hist_classes], 2)
    #     sum_embeddings /= batch_nums
    #     sequence_embeddings.append(sum_embeddings)
        
    #     # batch_hist_counts = batch_hist_counts.unsqueeze(-1)
    #     max_rate =  (self.MAX_COUNT-batch_hist_counts).float()/(self.MAX_COUNT-self.MIN_COUNT)
    #     min_rate =  (batch_hist_counts-self.MIN_COUNT).float()/(self.MAX_COUNT-self.MIN_COUNT)
        
    #     resutl1 = max_rate*self.max_count_emb.float().squeeze(0)
    #     result2 = min_rate*self.min_count_emb.float().squeeze(0)
    #     # resutl1 = self.batch_mat_mul(max_rate, self.max_count_emb.float())
    #     # result2 = self.batch_mat_mul(min_rate, self.min_count_emb.float())
        
    #     count_emb = resutl1 + result2
        
    #     # count_emb = self.int2emb(batch_hist_counts, self.MAX_COUNT, self.MIN_COUNT, self.max_count_emb, self.min_count_emb)
    #     sum_embeddings = self.reduce_sum(count_emb, 2)
    #     sum_embeddings /= batch_nums
    #     sequence_embeddings.append(sum_embeddings)
        
    #     # batch_hist_times = batch_hist_times.unsqueeze(-1)
    #     max_rate =  (self.MAX_TIME-batch_hist_times).float()/(self.MAX_TIME-self.MIN_TIME)
    #     min_rate =  (batch_hist_times-self.MIN_TIME).float()/(self.MAX_TIME-self.MIN_TIME)
        
    #     resutl1 = max_rate*self.max_time_emb.float().squeeze(0)
    #     result2 = min_rate*self.min_time_emb.float().squeeze(0)
    #     # resutl1 = self.batch_mat_mul(max_rate, self.max_time_emb.float())
    #     # result2 = self.batch_mat_mul(min_rate, self.min_time_emb.float())
        
    #     time_emb = resutl1 + result2
    #     # time_emb = self.int2emb(batch_hist_times, self.MAX_TIME, self.MIN_TIME, self.max_time_emb, self.min_time_emb)
        
    #     sum_embeddings = self.reduce_sum(time_emb, 2)
    #     sum_embeddings /= batch_nums
    #     sequence_embeddings.append(sum_embeddings)
       
    #     input_embeddings = self.concat((self.reshape(self.concat(variable_embeddings), (cur_batch_size, -1)), 
    #                         self.reshape(self.concat(sequence_embeddings), (cur_batch_size, -1))))
       
    #     shared_embeddings = self.Shared_Layer(input_embeddings)
    #     app_output = self.Final_App_Layer(self.App_Layers(shared_embeddings))
    #     # time_output = self.Final_Time_Layer(self.Time_Layers(shared_embeddings))
        
    #     return app_output
    
    
    # def sum_and_mean(self, embeddings, numbers, dim):
    #     sum_embeddings = self.reduce_sum(embeddings, dim)
    #     return sum_embeddings/numbers
    
   
    # def int2emb(self, number, max_num, min_num, max_emb, min_emb):
    #     max_rate =  (max_num-number)/(max_num-min_num)
    #     min_rate =  (number-min_num)/(max_num-min_num)
        
    #     resutl1 = self.batch_mat_mul(max_rate.float(), max_emb.float())
    #     result2 = self.batch_mat_mul(min_rate.float(), min_emb.float())
        
    #     return resutl1 + result2
    
    
    