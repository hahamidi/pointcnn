import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear as Lin
from torch_geometric.nn import XConv, fps, global_mean_pool,knn_interpolate
from torch_geometric.profile import rename_profile_file

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--inference', action='store_true')
parser.add_argument('--profile', action='store_true')
args = parser.parse_args()



def down_sample_layer(x ,pose,batch,ratio = 0.375 ):
    idx = fps(pose, batch, ratio=ratio)
    x, pose, batch = x[idx], pose[idx], batch[idx]
    return x,pose,batch 



class segmention_Net(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()


        self.num_classes = num_classes

        forward_laye_down = [64,96,192,384]
        forward_laye_up   = [384,192,96,96]
        hidden_down       = [32,64,128,256]
        hidden_up         = [256,128,64,64]
        kernel_size_down  = [16,16,16,16]
        kernel_size_up    = [16,16,16,16]
        self.Down_layers = nn.ModuleList()
        self.Up_layers = nn.ModuleList()

        self.down_sample = [1,0.375,0.375,1,1]

        prev = 0 
        for indx,layer in enumerate(forward_laye_down):
            self.Down_layers.append(XConv(prev,layer,kernel_size= kernel_size_down[0],hidden_channels = hidden_down[0]))
            if indx > 0 :
                 prev = layer[indx-1]
        prev = 0 
        for indx,layer in enumerate(forward_laye_up):
            self.Up_layers.append(XConv(prev,layer,kernel_size= kernel_size_up[0],hidden_channels = hidden_up[0]))
            if indx > 0 :
                 prev = layer[indx-1]

        
        # self.conv1 = XConv(0, 64, dim=3, kernel_size=16, hidden_channels=32)
        # self.conv2 = XConv(64, 96, dim=3, kernel_size=16, hidden_channels=64,dilation=2)
        # self.conv3 = XConv(96, 192, dim=3, kernel_size=16, hidden_channels=128,dilation=2)
        # self.conv4 = XConv(192, 384, dim=3, kernel_size=16,hidden_channels=256, dilation=2)
        # self.conv4_up = XConv(384 + 128 , 192 , dim=3, kernel_size=16,hidden_channels=320, dilation=2)
        # self.conv3_up = XConv(192 + 192 , 96 , dim=3, kernel_size=16,hidden_channels=256, dilation=2)
        # self.conv2_up = XConv(96 + 96 , 96 , dim=3, kernel_size=16,hidden_channels=125, dilation=2)
        # self.conv1_up = XConv(96 + 64 , 128 , dim=3, kernel_size=16,hidden_channels=120, dilation=2)
        

        self.lin1 = Lin(384, 256)
        self.lin2 = Lin(256, 128)

        self.down_sampler = down_sample_layer

        self.fc_lyaer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, self.num_classes, kernel_size=1),
            )
        
    def forward(self, pos, batch):
        layer_features = []
        layer_pos = []
        layer_batch = []

        x1 = F.relu(self.conv1(None, pos, batch))
        layer_features.append(x1)
        layer_pos.append(pos)
        layer_batch.append(batch)

        x_loop = layer_features[0]
        pos_loop = layer_pos[0]
        batch_loop = layer_pos[0]

        for layer in range(1,len(self.Down_layers)):
            if self.down_sample[layer] != 1:
                 x_loop,pos_loop,batch_loop = self.down_sampler(x_loop,pos_loop,batch_loop,ratio=self.down_sample[layer])
            x_loop =  F.relu(self.Down_layers[layer])(x_loop, pos_loop, batch_loop)
            layer_features.append(x_loop)
            pos_loop.append(pos_loop)
            batch_loop.append(batch_loop)
        

        x_glob = global_mean_pool(layer_features[-1], batch_loop[-1])
        x_glob = F.relu(self.lin1(x_glob))
        x_glob = F.relu(self.lin2(x_glob))
        x_con_glob = x_glob[batch_loop[-1]]

        up_feature = torch.cat((x_con_glob,layer_features[-1]),1)


        for layer in range(len(self.Up_layers)):
            up_feature = torch.cat((up_feature,layer_features[-1-layer]),1)
            up_feature = F.relu(self.Up_layers[layer](up_feature, pos_loop[-1-layer], batch_loop[-1-layer]))
            
            if self.down_sample[-1-layer] != 1:
                up_feature = knn_interpolate(x = up_feature,pos_x=pos_loop[-1-layer],batch_x=batch_loop[-1-layer],k=3,pos_y=pos_loop[-2-layer],batch_y=batch_loop[-2-layer])

        out = torch.unsqueeze(up_feature.T, 0)
        out = self.fc_lyaer(out)
        # x1 = F.relu(self.conv1(None, pos, batch))
        # x2, pos1, batch1 = self.down_sample(x1, pos, batch)
        # x2 = F.relu(self.conv2(x2, pos1, batch1))
        # x3, pos2, batch2 = self.down_sample(x2, pos1, batch1)
        # x3 = F.relu(self.conv3(x3, pos2, batch2))
        # x4 = F.relu(self.conv4(x3, pos2, batch2))
        
        # x_glob = global_mean_pool(x4, batch2)
        # x_glob = F.relu(self.lin1(x_glob))
        # x_glob = F.relu(self.lin2(x_glob))
        
        # layer_up1 = torch.cat((x_con_glob,x4),1)
        # up4 = F.relu(self.conv4_up(layer_up1, pos2, batch2))
        # layer_up2 = torch.cat((up4,x3),1)
        # up3 = F.relu(self.conv3_up(layer_up2, pos2, batch2))
        # layer_up3 = torch.cat((knn_interpolate(x = up3,pos_x=pos2,batch_x=batch2,k=4,pos_y=pos1,batch_y=batch1),x2),1)

        # up2 = F.relu(self.conv2_up(layer_up3, pos1, batch1))
        # layer_up4 = torch.cat((knn_interpolate(x = up2,pos_x=pos1,batch_x=batch1,k=4,pos_y=pos,batch_y=batch),x1),1)
        # up1 = F.relu(self.conv1_up(layer_up4, pos, batch))
        

        # out = torch.unsqueeze(up1.T, 0)
        
        # out = self.fc_lyaer(out)
        
        # out_batch = torch.zeros(batch_size,point_number, self.num_classes)
        # out = out.squeeze(0).T
        # for b in range(batch_size):
        #     out_batch[b,:,:] = out[batch == b]
        return out.squeeze(0)

        











print("runnig")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = segmention_Net(num_classes=13).to(device)
pos = torch.load('/content/ss/tensor_pos.pt')
batch = torch.load('/content/ss/tensor_batch.pt')
print(pos.size())
print(batch.size())
model(pos,batch)
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        # print(p)
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
print(get_n_params(model))

torch.save(model.state_dict(), '/content/ss/model_state_dict.pt')
