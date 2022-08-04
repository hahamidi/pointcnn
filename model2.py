
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import XConv, fps,knn_interpolate




def down_sample_layer(x ,pose,batch,ratio = 0.375 ):
    idx = fps(pose, batch, ratio=ratio)
    x, pose, batch = x[idx], pose[idx], batch[idx]
    return x,pose,batch 



class POINTCNN_SEG(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()


        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        self.conv1 = XConv(0, 256, dim=3, kernel_size=8,dilation = 1, hidden_channels= 128)
        self.conv2 = XConv(256, 512, dim=3, kernel_size=12, hidden_channels=384,dilation=2)
        self.conv3 = XConv(512, 768, dim=3, kernel_size=16, hidden_channels=640,dilation=2)
        self.conv4 = XConv(768, 1024, dim=3, kernel_size=16,hidden_channels=896, dilation=4)

        self.conv_up4 = XConv(1024, 1024, dim=3, kernel_size=16,hidden_channels=896, dilation=4)
        self.conv_up3 = XConv(1024 , 768 , dim=3, kernel_size=16,hidden_channels=896, dilation=2)
        self.conv_up2 = XConv(768 , 512 , dim=3, kernel_size=12,hidden_channels=640, dilation=2)
        self.conv_up1 = XConv(512, 256 , dim=3, kernel_size=8,hidden_channels=120, dilation=2)

        self.mlp_out4 = nn.Conv1d(1024, 1024, kernel_size=1)
        self.mlp_out3 = nn.Conv1d(768, 768, kernel_size=1)
        self.mlp_out2 = nn.Conv1d(512, 512, kernel_size=1)
        self.mlp_out1 = nn.Conv1d(256, 256, kernel_size=1)
        


        self.down_sampler = down_sample_layer

        self.fc_lyaer = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, self.num_classes, kernel_size=1),
            )


        ##head because of in pytorch geometric we can't use batch dimention like [1,num_classes,number_of_point] we cant use nn.sequential 
        self.fc_lyaer1 = nn.Conv1d(256, 128, kernel_size=1,bias = False)
        self.Relu = nn.ReLU(True)
        self.DROP = nn.Dropout(0.5)
        self.fc_lyaer2 =nn.Conv1d(128, self.num_classes, kernel_size=1)


        self.batch_size = 16
        self.number_of_point = 2048



    #pytorch geometric layer need batch and pos as input batch is like batch = [1,1,2,2,3,3] means firest data is batch 1 
    # and second data is batch 2 and so on
    def pre_pointcnn(self,points):

        self.batch_size = points.shape[0]
        self.number_of_point = points.shape[1]

        batch_zero = torch.zeros(points[0].shape[0],dtype=torch.int64)
        batch = torch.zeros(points[0].shape[0],dtype=torch.int64)
        point_for_pointcnn = points[0]
        for b in range(1,points.shape[0]):
                        batch = torch.cat((batch,batch_zero + b),dim=0)
                        point_for_pointcnn = torch.cat((point_for_pointcnn,points[b]),dim=0)
        points = point_for_pointcnn
        points = points.to(self.device)
        batch = batch.to(self.device)

        
        return points,batch


    #after predict we need to reshape the output to be like [batch,num_classes,number_of_point] beacuse of pytorch geometric layer condition 
    #see pre_pointcnn comment for more information

    def after_pred(self,preds,batch):

        out_batch = torch.zeros(self.batch_size,self.num_classes,self.number_of_point )
        out = preds.T
   
        for b in range(self.batch_size):
            out_batch[b,:,:] = out[batch == b].T
        preds = out_batch
        preds = preds.to(self.device)
        return preds
        


    def forward(self,points):
        pos0,batch0 = self.pre_pointcnn(points)
        
        pos1,batch1 = pos0, batch0
        # print(pos1.shape)
        x1 = F.relu(self.conv1(None, pos1, batch1))

        x2, pos2, batch2 = self.down_sampler(x1, pos1, batch1)
        # print(x2.shape)
        x2 = F.relu(self.conv2(x2, pos2, batch2))

        x3, pos3, batch3 = self.down_sampler(x2, pos2, batch2 , ratio = 0.375)
        # print(x3.shape)
        x3 = F.relu(self.conv3(x3, pos3, batch3))

        x4, pos4, batch4 = self.down_sampler(x3, pos3, batch3,ratio=0.375)
        # print(x4.shape)
        x4 = F.relu(self.conv4(x4, pos4, batch4))


        xo4 = F.relu(self.conv_up4(x4, pos4, batch4))

        xo4_concat = (xo4 + x4).T
        xo4_after_mlp = self.mlp_out4(xo4_concat).T
        Xo3_in = knn_interpolate(x = xo4_after_mlp, pos_x=pos4 , batch_x=batch4 , k=3 ,pos_y=pos3,batch_y=batch3)
        xo3 = F.relu(self.conv_up3(Xo3_in, pos3, batch3))


        
        xo3_concat = (xo3 + x3).T
        xo3_after_mlp = self.mlp_out3(xo3_concat).T
        Xo2_in = knn_interpolate(x = xo3_after_mlp, pos_x=pos3 , batch_x=batch3 , k=3 ,pos_y=pos2,batch_y=batch2)


        xo2 = F.relu(self.conv_up2(Xo2_in, pos2, batch2))


        xo2_concat = (xo2 + x2).T

        xo2_after_mlp = self.mlp_out2(xo2_concat).T

        Xo1_in = knn_interpolate(x = xo2_after_mlp, pos_x=pos2 , batch_x=batch2 , k=3 ,pos_y=pos1,batch_y=batch1)

        xo1 = F.relu(self.conv_up1(Xo1_in, pos1, batch1))

        xo1_concat = (xo1 + x1).T

        xo1_after_mlp = self.mlp_out1(xo1_concat)

        X_OUT = torch.unsqueeze(xo1_after_mlp.T, 0)
        # X_OUT = self.BN(X_OUT)

        X_OUT = self.fc_lyaer1(xo1_after_mlp)
        X_OUT = self.Relu(X_OUT)
        X_OUT = self.DROP(X_OUT)
        X_OUT = self.fc_lyaer2(X_OUT)

        X_OUT = self.after_pred(X_OUT,batch=batch0)


        return X_OUT



class POINTCNN_SEG_attention(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()


        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        self.conv1 = XConv(0, 256, dim=3, kernel_size=8,dilation = 1, hidden_channels= 128)
        self.conv2 = XConv(256, 512, dim=3, kernel_size=12, hidden_channels=384,dilation=2)
        self.conv3 = XConv(512, 768, dim=3, kernel_size=16, hidden_channels=640,dilation=2)
        self.conv4 = XConv(768, 1024, dim=3, kernel_size=16,hidden_channels=896, dilation=4)

        self.conv_up4 = XConv(1024, 1024, dim=3, kernel_size=16,hidden_channels=896, dilation=4)
        self.conv_up3 = XConv(1024 , 768 , dim=3, kernel_size=16,hidden_channels=896, dilation=2)
        self.conv_up2 = XConv(768 , 512 , dim=3, kernel_size=12,hidden_channels=640, dilation=2)
        self.conv_up1 = XConv(512, 256 , dim=3, kernel_size=8,hidden_channels=120, dilation=2)

        self.mlp_out4 = nn.Conv1d(1024, 1024, kernel_size=1)
        self.mlp_out3 = nn.Conv1d(768, 768, kernel_size=1)
        self.mlp_out2 = nn.Conv1d(512, 512, kernel_size=1)
        self.mlp_out1 = nn.Conv1d(256, 256, kernel_size=1)


        #attention 
        self.multihead_attn = nn.MultiheadAttention( 1024 , 2,batch_first =True,need_weights =False)

        


        self.down_sampler = down_sample_layer

        self.fc_lyaer = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, self.num_classes, kernel_size=1),
            )


        ##head because of in pytorch geometric we can't use batch dimention like [1,num_classes,number_of_point] we cant use nn.sequential 
        self.fc_lyaer1 = nn.Conv1d(256, 128, kernel_size=1,bias = False)
        self.Relu = nn.ReLU(True)
        self.DROP = nn.Dropout(0.5)
        self.fc_lyaer2 =nn.Conv1d(128, self.num_classes, kernel_size=1)


        self.batch_size = 16
        self.number_of_point = 2048



    #pytorch geometric layer need batch and pos as input batch is like batch = [1,1,2,2,3,3] means firest data is batch 1 
    # and second data is batch 2 and so on
    def pre_pointcnn(self,points):

        self.batch_size = points.shape[0]
        self.number_of_point = points.shape[1]

        batch_zero = torch.zeros(points[0].shape[0],dtype=torch.int64)
        batch = torch.zeros(points[0].shape[0],dtype=torch.int64)
        point_for_pointcnn = points[0]
        for b in range(1,points.shape[0]):
                        batch = torch.cat((batch,batch_zero + b),dim=0)
                        point_for_pointcnn = torch.cat((point_for_pointcnn,points[b]),dim=0)
        points = point_for_pointcnn
        points = points.to(self.device)
        batch = batch.to(self.device)

        
        return points,batch


    #after predict we need to reshape the output to be like [batch,num_classes,number_of_point] beacuse of pytorch geometric layer condition 
    #see pre_pointcnn comment for more information

    def after_pred(self,preds,batch):

        out_batch = torch.zeros(self.batch_size,self.num_classes,self.number_of_point )
        out = preds.T
   
        for b in range(self.batch_size):
            out_batch[b,:,:] = out[batch == b].T
        preds = out_batch
        preds = preds.to(self.device)
        return preds
        


    def forward(self,points):
        pos0,batch0 = self.pre_pointcnn(points)
        
        pos1,batch1 = pos0, batch0
        # print(pos1.shape)
        x1 = F.relu(self.conv1(None, pos1, batch1))

        x2, pos2, batch2 = self.down_sampler(x1, pos1, batch1)
        # print(x2.shape)
        x2 = F.relu(self.conv2(x2, pos2, batch2))

        x3, pos3, batch3 = self.down_sampler(x2, pos2, batch2 , ratio = 0.375)
        # print(x3.shape)
        x3 = F.relu(self.conv3(x3, pos3, batch3))

        x4, pos4, batch4 = self.down_sampler(x3, pos3, batch3,ratio=0.375)
        # print(x4.shape)
        x4 = F.relu(self.conv4(x4, pos4, batch4))
        ################################################
        # from batch [N] and data [P,C] to batch [N,P,C]


        out_batch = torch.zeros(self.batch_size, int(pos4.shape[0] / self.batch_size) ,1024,).to(self.device) 
        
        

        for b in range(self.batch_size):
            out_batch[b,:,:] = x4[batch4 == b]
        print()        
        X_attention  = self.multihead_attn(out_batch, out_batch, out_batch)
        print(X_attention.shape)
        X_attention = self.pre_pointcnn(X_attention)
        print(X_attention.shape)
       


        # X_Key = self.K(x4.T)
        # X_Query = self.Q(x4.T)
        # X_Value = self.V(x4.T)









        #################################################

        xo4 = F.relu(self.conv_up4(x4, pos4, batch4))

        xo4_concat = (xo4 + x4).T
        xo4_after_mlp = self.mlp_out4(xo4_concat).T
        Xo3_in = knn_interpolate(x = xo4_after_mlp, pos_x=pos4 , batch_x=batch4 , k=3 ,pos_y=pos3,batch_y=batch3)
        xo3 = F.relu(self.conv_up3(Xo3_in, pos3, batch3))


        
        xo3_concat = (xo3 + x3).T
        xo3_after_mlp = self.mlp_out3(xo3_concat).T
        Xo2_in = knn_interpolate(x = xo3_after_mlp, pos_x=pos3 , batch_x=batch3 , k=3 ,pos_y=pos2,batch_y=batch2)


        xo2 = F.relu(self.conv_up2(Xo2_in, pos2, batch2))


        xo2_concat = (xo2 + x2).T

        xo2_after_mlp = self.mlp_out2(xo2_concat).T

        Xo1_in = knn_interpolate(x = xo2_after_mlp, pos_x=pos2 , batch_x=batch2 , k=3 ,pos_y=pos1,batch_y=batch1)

        xo1 = F.relu(self.conv_up1(Xo1_in, pos1, batch1))

        xo1_concat = (xo1 + x1).T

        xo1_after_mlp = self.mlp_out1(xo1_concat)

        X_OUT = torch.unsqueeze(xo1_after_mlp.T, 0)
        # X_OUT = self.BN(X_OUT)

        X_OUT = self.fc_lyaer1(xo1_after_mlp)
        X_OUT = self.Relu(X_OUT)
        X_OUT = self.DROP(X_OUT)
        X_OUT = self.fc_lyaer2(X_OUT)

        X_OUT = self.after_pred(X_OUT,batch=batch0)
        # print(X_OUT.shape)


        return X_OUT










        
        
        
if __file__ == "__main__":
    Net = POINTCNN_SEG(10)