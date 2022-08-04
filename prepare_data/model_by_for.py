# class POINTCNN_SEG_2(torch.nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         layer_down      = [256,512,768,1024]
#         layer_up        = [1024,768,512,256]
#         hidden_layer_down      = [int((0 + 256) / 2), int((256 + 512) / 2), int((512 + 768) / 2), int((768 + 1024) / 2)]
#         hidden_layer_up   = [int((1024 + 768) / 2), int((768 + 512) / 2), int((512 + 256) / 2), int((256 + 0) / 2)]

#         kernel_size_down       = [8,12,16,16]
#         kernel_size_up         = [16,16,12,8]

#         dilation_down          = [1,2,2,4]
#         dilation_up            = [4,2,2,2]

#         dim_size = 3
#         self.down_sample = [0.375,0.375,0.375,0]
    
#         self.num_classes = num_classes
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.batch_size = 16
#         self.number_of_point = 2048

#         self.Down_layers = []
#         self.Up_layers = []
#         self.MLPs    = []
#         self.down_sampler = down_sample_layer

#         prev = 0 
#         for indx,layer in enumerate(layer_down):
#             print(prev,"|",layer,"|",hidden_layer_down[indx],"|",kernel_size_down[indx],"|",dilation_down[indx])
#             self.Down_layers.append(XConv(prev,layer,dim = dim_size,kernel_size= kernel_size_down[indx],hidden_channels = hidden_layer_down[indx]))
#             if indx > 0 :
#                  prev = layer_down[indx-1]


#         prev = layer_up[-1] 
#         for indx,layer in enumerate(layer_up):
#             self.Up_layers.append(XConv(prev,layer,dim = dim_size ,kernel_size= kernel_size_up[indx],hidden_channels = hidden_layer_up[indx]))
#             if indx > 0 :
#                  prev = layer_up[indx-1]

#         for index,layer in enumerate(layer_up):
#             self.MLPs.append(nn.Conv1d(layer, layer, kernel_size=1))
        
#         self.fully_conecteds_1 = nn.Conv1d(layer_up[-1], 128, kernel_size=1)
#         self.Relu              = nn.ReLU()
#         self.fully_conecteds_2 = nn.Conv1d(128, num_classes, kernel_size=1)

#     def after_pred(self,preds,batch):

#         out_batch = torch.zeros(self.batch_size,self.num_classes,self.number_of_point )
#         out = preds.T
   
#         for b in range(self.batch_size):
#             out_batch[b,:,:] = out[batch == b].T
#         preds = out_batch
#         preds = preds.to(self.device)
#         return preds

#     def pre_pointcnn(self,points):

#         self.batch_size = points.shape[0]
#         self.number_of_point = points.shape[1]

#         batch_zero = torch.zeros(points[0].shape[0],dtype=torch.int64)
#         batch = torch.zeros(points[0].shape[0],dtype=torch.int64)
#         point_for_pointcnn = points[0]
#         for b in range(1,points.shape[0]):
#                         batch = torch.cat((batch,batch_zero + b),dim=0)
#                         point_for_pointcnn = torch.cat((point_for_pointcnn,points[b]),dim=0)
#         points = point_for_pointcnn
#         points = points.to(self.device)
#         batch = batch.to(self.device)

        
#         return points,batch

    

#     def forward(self,points):
#         pos0,batch0 = self.pre_pointcnn(points)
#         print(pos0,batch0)

#         forward_down_features = []

#         forward_down_pos = []
#         forward_down_batch = []

#         forward_down_pos.append(pos0)
#         forward_down_batch.append(batch0)

#         pos_loop = pos0
#         batch_loop = batch0
#         # X_loop = None
#         # X_loop =  X_loop.to(self.device)

#         for index,layer in enumerate(self.Down_layers):
#             if index == 0:
                
#                 X_loop = layer(None,pos_loop,batch_loop)
#                 print(1)
                
#             X_loop = layer(X_loop,pos_loop,batch_loop)
#             forward_down_features.append(X_loop)
#             if self.down_sample[index] != 0 :
#                   X_loop, pos_loop, batch_loop = self.down_sampler(X_loop, pos_loop, batch_loop,ratio = self.down_sample[index]) 

#             forward_down_pos.append(pos_loop)
#             forward_down_batch.append(batch_loop)
        
#         X_loop = forward_down_features[-1]
#         for index,layer in enumerate(self.Up_layers):
#             X_loop = layer(X_loop,forward_down_pos[-1],forward_down_batch[-1])
#             ADDed = (X_loop + forward_down_pos[-1]).T
#             X_loop  = self.MLPs[index](ADDed).T

            
#         X_OUT = self.fully_conecteds_1(X_loop)
#         X_OUT = self.Relu(X_OUT)
#         X_OUT = self.fully_conecteds_2(X_OUT)
#         X_OUT = self.after_pred(X_OUT,batch0)

#         return X_OUT
    