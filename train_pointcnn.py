
import argparse
import os
import torch
import tensorflow
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random

from tensorflow.keras.metrics import MeanIoU
from torch.nn import CrossEntropyLoss
from sklearn.manifold import TSNE as sklearnTSNE
from models import POINTCNN_SEG_attention,POINTCNN_SEG
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import ShapeNetPart

dire = os.getcwd().split('/')
dire = '/'.join(dire)

print("=======>",dire)
class Trainer():
    def __init__(self,model,
                        train_data_loader, 
                        val_data_loader , 
                        optimizer ,
                        epochs,
                        number_of_classes,
                        start_index,
                        loss_function,
                        scheduler,
                        device):
        self.model = model
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.optimizer = optimizer
        self.epochs = epochs
        self.number_of_classes = number_of_classes
        self.loss_function = loss_function
        self.scheduler = scheduler
        self.device = device
        self.start_index = start_index
        self.load_model = False
        self.load_epoch = 1980

        self.blue= lambda x: '\033[94m' + x + '\033[0m'
        self.red = lambda x: '\033[91m' + x + '\033[0m'
        self.yellow = lambda x: '\033[93m' + x + '\033[0m'



   
    def train_one_epoch(self,epoch):

                epoch_train_loss = []
                epoch_train_acc = []
                batch_number = 0
                self.model = self.model.train()
                #label is class in classification
                #target is class in segmentation
               
                for points,labels,targets in self.train_data_loader:
                    batch_number += 1

                    # make target from zero to number_of_classes-1
                    targets = targets - self.start_index
                    points, targets = points.to(self.device), targets.to(self.device)

                    if points.shape[0] <= 1:
                        continue


                    self.optimizer.zero_grad()
                    preds = self.model(points)
                    loss =  self.loss_function(preds, targets)  # * regularization_loss
                    loss.backward()
                    print(self.blue(str(epoch)+"/"+str(batch_number)+": Train Loss = " + str(loss.item())))

                    epoch_train_loss.append(loss.cpu().item())
                    self.optimizer.step()
                    if args.scheduler == 'cos':
                        self.scheduler.step()
                    elif args.scheduler == 'step':
                        if self.optimizer.param_groups[0]['lr'] > 1e-5:
                             self.scheduler.step()
                        if self.optimizer.param_groups[0]['lr'] < 1e-5:
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = 1e-5

                    preds = preds.data.max(1)[1]
                    corrects = preds.eq(targets.data).cpu().sum()
                    accuracy = corrects.item() / float(self.train_data_loader.batch_size*args.num_points)
                    epoch_train_acc.append(accuracy)

                print(self.yellow("Loss "+str(np.mean(epoch_train_loss))))
                print(self.yellow("Accuracy "+str(np.mean(epoch_train_acc))))
                

                                                                        
    def val_one_epoch(self,epoch):
        epoch_val_loss = []
        epoch_val_acc = []
        batch_number = 0
        shape_ious = []
        self.model = self.model.eval()
        # use tensorflow to calculate IoU
        with tensorflow.device('/cpu:0'):
            MIOU_obj = MeanIoU(self.number_of_classes, name=None, dtype=None)
            for points,labels,targets in self.val_data_loader:

                        targets = targets - self.start_index
                        batch_number += 1
                        points, targets = points.to(self.device), targets.to(self.device)


                
                        if points.shape[0] <= 1:
                            continue

                        with torch.no_grad():                        
                                preds = self.model(points)
                                loss =  self.loss_function(preds, targets)
                                # print(self.red(str(epoch)+"/"+str(batch_number)+": Val Loss = "+ str(loss.item())))
                        epoch_val_loss.append(loss.cpu().item())

                        preds = preds.data.max(1)[1]
                        pred_np = preds.cpu().data.numpy()
                        target_np = targets.cpu().data.numpy()
                        MIOU_obj.update_state(pred_np, target_np)
                        part_ious = MIOU_obj.result().numpy()
                        shape_ious.append(np.mean(part_ious))

                        corrects = preds.eq(targets.data).cpu().sum()
                        accuracy = corrects.item() / float(self.val_data_loader.batch_size*args.num_points)
                        epoch_val_acc.append(accuracy)

        print("Loss",np.mean(epoch_val_loss))
        print("Accuracy",np.mean(epoch_val_acc))
        print("Mean IOU: ", np.mean(shape_ious))


    def save_model_optimizer(self,epoch_num):
        torch.save(self.model.state_dict(), dire+'/checkpoints/model_epoch_' + str(epoch_num) + '.pth')
        torch.save(self.optimizer.state_dict(),  dire+'/checkpoints/optimizer_epoch_' + str(epoch_num) + '.pth')
        self.model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        print('Model and optimizer saved!')

    def load_model_optimizer(self,epoch_num):
        self.model.load_state_dict(torch.load( dire+'/checkpoints/model_epoch_' + str(epoch_num) + '.pth'))
        self.optimizer.load_state_dict(torch.load( dire+'/checkpoints/optimizer_epoch_' + str(epoch_num) + '.pth'))
        print('Model and optimizer loaded!')
    def new_head(self,number_of_class):
        head =  torch.nn.Conv1d(128, number_of_class, kernel_size=1, bias=False)
        head = head.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.conv11 = head


        print(self.model)

    def show_embedding_sklearn(self,tsne_embs_i, lbls,title = "", cmap=plt.cm.tab20,highlight_lbls = None):
            
            labels = lbls.flatten()
            print(labels.shape)
            print(tsne_embs_i.shape)
            feat = np.zeros((tsne_embs_i.shape[1],tsne_embs_i.shape[2])).T

            for b in tsne_embs_i:
                feat= np.concatenate((feat, b.T), axis=0)

            feat= feat[tsne_embs_i.shape[2]: , :]
            number_of_labels = np.amax(labels) + 1
            selected = np.zeros((tsne_embs_i.shape[1],1)).T
            labels_s = []
            print(feat.shape)

            for i in range(number_of_labels):
                selected= np.concatenate((selected,feat[labels == i][0:100]), axis=0)
                labels_s= np.concatenate((labels_s,labels[labels == i][0:100]), axis=0)
            selected = selected[1:]

            tsne = sklearnTSNE(n_components=2, random_state=0)  # n_components means you mean to plot your dimensional data to 2D
            x_test_2d = tsne.fit_transform(selected)

            fig,ax = plt.subplots(figsize=(10,10))
            ax.scatter(x_test_2d[:,0], x_test_2d[:,1], c=labels_s, cmap=cmap, alpha=1 if highlight_lbls is None else 0.1)
            random_str = str(random.randint(0,1000000))
            plt.savefig("/./content/embed"+random_str+"-"+str(title)+'.png')



    def train(self):
            if self.load_model == True:
                self.load_model_optimizer(self.load_epoch)
                self.new_head(self.number_of_classes )

            for epoch in range(self.epochs):
                
                self.train_one_epoch(epoch)
                self.val_one_epoch(epoch)
                # if epoch % 20 == 0:
                #    self.save_model_optimizer(epoch)

                
                # self.scheduler.step()
                # torch.save(self.model.state_dict(), 'model_%d.pkl' % epoch)

    

    

    
if __name__ == '__main__':
    
    # number_ofpoints = number of points in the point cloud
    # class_choice = which class to train on
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--epochs', type=int, default=2000, help='number of epochs')
    parser.add_argument('--num_points', type=int, default=2048,help='num of points to use')

    
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')

    parser.add_argument('--class_choice', type=str, default="chair", metavar='N',
                        choices=['airplane', 'bag', 'cap', 'car', 'chair',
                                 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 
                                 'motor', 'mug', 'pistol', 'rocket', 'skateboard', 'table'])

    args = parser.parse_args()
    print(args)

    # select training data and test data
    train_dataset = ShapeNetPart(partition='trainval', num_points=args.num_points, class_choice=args.class_choice)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        num_workers=2,drop_last=True)
    test_dataset = ShapeNetPart(partition='test', num_points=args.num_points, class_choice=args.class_choice)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        num_workers=2,drop_last=True)


    # create model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = POINTCNN_SEG(train_dataset.seg_num_all)
    model.to(device)       
    opt = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.5)

    # trainer loop 
    # number_of_classes is number of class in selected category for example, if we select guitar, then number of class is 3 see data.py for more details
    # start_index is the index of the first class in selected category for example, if we select guitar, then start_index is 19 see data.py for more details

    trainer = Trainer(model = model,
                        train_data_loader = train_dataloader, 
                        val_data_loader = test_dataloader, 
                        optimizer = opt,
                        epochs=args.epochs,
                        number_of_classes = train_dataset.seg_num_all,
                        start_index = train_dataset.seg_start_index,
                        loss_function = CrossEntropyLoss(),
                        scheduler = scheduler,
                        device =device)
    trainer.train()

    #comments added






