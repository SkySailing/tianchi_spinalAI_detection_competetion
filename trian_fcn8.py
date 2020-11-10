# author: 
# contact: onlydgi@foxmail.com
# datetime:2020/7/15 下午11:00
# software: PyCharm
"""
文件说明：
"""
import os
import pprint
import cv2
import numpy as np
import torch
from torch.backends import cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from time import time
from time import strftime
import torchvision.transforms as transforms


from sipnalDataset import mydata
from torchvision.models.vgg import VGG
from unet import unet_model, unet_parts
from fcn import FCN8s,VGGNet

# from apex import amp, optimizers
#
# from apex.parallel import DistributedDataParallel

#导入包
from visdom import Visdom

config = dict()
# be care to set config before submit
config['showFlag'] = 0
config["valid"] = 1
config['epoch_num'] = 1000
config['tranform'] = 1
config['saveName'] ='FCN8PosiLoss_orignal_final_model.ckpt'
config['checkout'] = "FCN8PosiLoss_pretrian_model.ckpt"
# 'FCN8_orignal_pretrian_model.ckpt'
# 'UNet_double_orignal_pretrain_model.ckpt'
# pretrian_Unet_final_model.ckpt
# pretrainNet_model.ckpt
# pretrian_Unet_final_model.ckpt

config['lr'] = 0.00001
config['momentum'] = 0.9
config['weight_decay'] = 1e-4
config['start_epoch'] = 0
config['batch_size'] = 26
config['valid_batch_size'] = 4
config['save_freq'] = 100
config['sigma'] = 5.
config['root_dir'] = r"./tcdata/round2train"
config['label_dir'] = r"./tcdata/round2train_checked.json"
config['valid_root_dir'] = r"./tcdata/round2_valid53/valid"
config['valid_label_dir'] = r"./tcdata/round2_valid53/round2_valid53_annotation51.json"



if config['showFlag'] == 1:
    #生成一个viz的环境
    viz = Visdom()
    # python -m visdom.server
    #初始化两个小的窗格，来分别绘制train,test的情况
    # 绘制初始点，原点
    viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))  #single-line
    viz.line([0.], [0.], win='coor_loss', opts=dict(title='coor loss'))  #single-line
    viz.line([0.], [0.], win='valid_train_loss', opts=dict(title='valid_train loss'))  #single-line
    viz.line([0.], [0.], win='acc_coor', opts=dict(title='average acc_coor'))  #single-line

# 数据预处理设置
# normMean = [0.16783774]
# normStd = [0.18892017]
# normMean = [0.168036]
# normStd = [0.177935]
normMean = [0.168036]
normStd = [0.177935]
normTransform = transforms.Normalize(normMean, normStd)
if config['tranform'] == 1:
    trainTransform = transforms.Compose([
        transforms.ToTensor()
        # ,normTransform
    ])

    validTransform = transforms.Compose([
        transforms.ToTensor()
        # ,normTransform
    ])
else:
    trainTransform = None
    validTransform = None

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class posiLoss(nn.Module):
    def __init__(self,alph=100,beta=20):
        super(posiLoss, self).__init__()
        self.alph = alph
        self.beta = beta
        self.criterion_part1 = nn.MSELoss()
        self.sz = 11
        self.topKposi = 5

    def posiLoss(self,pred_heatmap, gt_heatmap):
        n,c,h,w = pred_heatmap.shape
        n_loss = list()
        for n_idx in range(n):
            # print("n:",n)
            pred_posiRation = self.relationPosi(pred_heatmap[n_idx])
            gt_posiRation = self.relationPosi(gt_heatmap[n_idx])
            psoiLossL = torch.Tensor(torch.abs(torch.sub(pred_posiRation,gt_posiRation))).cuda()
            # print("psoiLoss", psoiLossL)
            ret_psoiLoss = torch.sum(torch.topk(psoiLossL, self.topKposi).values)
            # print("ret_psoiLoss", ret_psoiLoss)
            n_loss.append(ret_psoiLoss)
        ret_n_loss = torch.mean(torch.Tensor(n_loss))
        return ret_n_loss

    def relationPosi(self,heatmap):
        # print(heatmap.shape)
        ret = list()
        ret.append(self.criterion_part1(heatmap[0], heatmap[1]))
        for i in range(1, self.sz-1):
            # print("i:",i)
            up = self.criterion_part1(heatmap[i-1], heatmap[i])
            down = self.criterion_part1(heatmap[i+1], heatmap[i])
            # print("down",down)
            ret.append(torch.mean(torch.Tensor((up, down))))
        ret.append(self.criterion_part1(heatmap[self.sz-1], heatmap[self.sz -2]))
        return torch.Tensor(ret)

    def forward(self, pred_heatmap, gt_heatmap):
        mse_loss = self.criterion_part1(pred_heatmap,gt_heatmap)
        # print("mse_loss", mse_loss)
        posiLosss = self.posiLoss(pred_heatmap,gt_heatmap)
        # print("posiLosss",posiLosss)
        return self.alph*mse_loss+self.beta*posiLosss



def get_peak_points(heatmaps):
    """
    :param heatmaps: numpy array (N,11,96,96)
    :return:numpy array (N,11,2),(width ,hight)
    """
    N,C,H,W = heatmaps.shape
    all_peak_points = []
    for i in range(N):
        peak_points = []
        for j in range(C):
            yy, xx = np.where(heatmaps[i][j] == heatmaps[i][j].max())
            y = yy[0]
            x = xx[0]
            peak_points.append([x, y])
        all_peak_points.append(peak_points)
    all_peak_points = np.array(all_peak_points)
    return all_peak_points

def get_mse(pred,gt):
    summ = 0
    pred = np.array(pred)
    gt = np.array(gt)
    for i in range(len(gt)):
        summ += ((pred[i]-gt[i])**2).sum()
    return np.sqrt(summ/len(gt))

def validDistance(all_peak_points, keyPsoi):
    """
    static infomation of the pred point and GT keyPoint

    Args:
        all_peak_points:pred point,like [[1,2],[2,3]]
        keyPsoi: ground truth
    """
    N, num_point, dimes = keyPsoi.shape
    staticPosi = [0]*num_point
    for i in range(N):
        a = all_peak_points[i]
        b = keyPsoi[i]
        for k in range(num_point):
            distance = np.sqrt(np.sum(np.square(a[k] - b[k])))
            if distance <= 6:
                staticPosi[k]+=1
    return np.array(staticPosi)
# validDistance(np.array([[[1,2],[1,1]],[[1,2],[1,1]]]),np.array([[[0,1],[7,8]],[[0,1],[7,8]]]))

def plotPoint(img,point):
    j = 1
    for gtPoint in point:
        cv2.circle(img, (gtPoint[0], gtPoint[1]), 8, thickness = 2, color=(0, 0, 255))
        cv2.putText(img, str(j), (gtPoint[0] - 10, gtPoint[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        j += 1
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img


if __name__ == '__main__':
    if config['showFlag'] == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = "1"
        config['batch_size'] = 48
        config['valid_batch_size'] = 10
        pprint.pprint(config)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        print("######### training is start")
    torch.manual_seed(0)
    cudnn.benchmark = True

    # net = unet_model.UNet(1, 11)
    # net = unet_model.UNet_double()
    # net = unet_model.UNet_twoPart(1, (6,5))
    vgg_model = VGGNet(requires_grad = True, pretrained=False)
    net = FCN8s(pretrained_net=vgg_model, n_class=11)

    # initial the net weight
    net = net.float().cuda()
    # net = nn.DataParallel(net).float().cuda()
    # net = nn.parallel.DistributedDataParallel(net.float().cuda()) # device_ids will include all GPU devices by default
    # net.float().cuda()
    # criterion = nn.MSELoss()
    criterion = posiLoss()
    optimizer = optim.Adam(net.parameters(), lr=config['lr'])  # good
    # optimizer = optim.Adagrad(net.parameters(), lr=config['lr']) # no good
    # optimizer = optim.RMSprop(net.parameters(), lr = config['lr'], alpha=0.9)
    # optimizer =optimizers.FusedAdam(net.parameters(), lr=config['lr'])

    # net, optimizer = amp.initialize(net, optimizer, opt_level='O0')
    # net, optimizer = amp.initialize(net, optimizer, opt_level="O0")
    # net = DistributedDataParallel(net)
    # net = nn.parallel.DistributedDataParallel(net.float().cuda())

    #initi the train dataset
    trainDataset = mydata(config['root_dir'], config['label_dir'], transforms=trainTransform)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(trainDataset)
    trainDataLoader = DataLoader(trainDataset, config['batch_size'], True)
    sample_num = len(trainDataset)

    # initi the valid dataset
    if config["valid"] == 1:
        validDataset = mydata(config['valid_root_dir'],config['valid_label_dir'], transforms=validTransform)
        validDataLoader = DataLoader(validDataset, config['valid_batch_size'], True)
        valid_num = len(validDataset)

    if config['checkout'] != '':
        cpu_checkpoint = torch.load(config['checkout'], map_location=torch.device('cpu'))
        net.load_state_dict(cpu_checkpoint)
        # net.load_state_dict(torch.load(config['checkout']))
    else:
        # inital the network with func
        net.apply(weights_init)

    for epoch in range(config['start_epoch'], config['epoch_num'] + config['start_epoch']):
        start = time()
        net.train()
        running_loss = 0.0
        running_loss_coor =0.0
        for i, (origi_shape, img_aug, keyPsoi, distance_maps_normalized, type_label, origi_keyposi) in enumerate(trainDataLoader):
            if config['tranform'] == 0:
                img_aug = img_aug[:, np.newaxis, :, :]
            img_aug = Variable(img_aug).float().cuda()
            # print(img_aug.shape)
            heatmaps_targets = Variable(distance_maps_normalized, requires_grad=True).float().cuda()
            # keyPsoi= Variable(keyPsoi).float().cuda()
            optimizer.zero_grad()
            outputs = net(img_aug)

            loss = 100 * criterion(outputs, heatmaps_targets)
            loss.backward()
            optimizer.step()
            # 评估
            all_peak_points = get_peak_points(outputs.cpu().data.numpy())
            loss_coor = get_mse(all_peak_points, keyPsoi)

            if config['showFlag'] == 1:
                # print("img_aug[0][0].cpu().numpy()",img_aug[0][0].cpu().numpy())
                plot_img = plotPoint(img_aug[0][0].cpu().numpy(), keyPsoi[0])
                # print("plot_img",plot_img)
                viz.images(plot_img, win='inputIMG', opts={'title': 'input_imgs'})
                print('[ Epoch {:005d} -> {:005d} / {} ] loss : {:15} loss_coor : {:15} '.format(
                    epoch, i * config['batch_size'],
                    sample_num, loss.item(), loss_coor.item()))

            running_loss += loss.item()
            running_loss_coor += loss_coor.item()
            torch.cuda.empty_cache()
        end = time()
        if config['showFlag'] == 1:
            print("this epoch cost time: ", end - start)

        running_loss /= sample_num
        running_loss_coor /= sample_num
        if config['showFlag'] == 0:
            print('[Epoch {}/{}] loss:{:.10} loss_coor:{:.10} time:{:.2f}'.format(
                epoch, config['epoch_num'], loss.item(), loss_coor.item(), end - start))
        if config['showFlag'] == 1:
            viz.line([running_loss], [epoch], win='train_loss', update='append')
            viz.line([running_loss_coor], [epoch], win='coor_loss', update='append')

        if (epoch-1) % config['save_freq'] == 0 or epoch == config['epoch_num'] - 1:
            torch.save(net.state_dict(), config['saveName'])
            # torch.save(net.state_dict(), 'Unet_lr1e-４_Adam_sigma10_epoch_{}_dataset203_model.ckpt'.format(epoch))

######################################################################　valid
        # valid the net just learned
        if config['showFlag'] == 1:
            net.eval()
            running_loss = 0.0
            staticPosi_6pixel_coor = np.array([0]*11, 'float32')
            with torch.no_grad():
                for i, (origi_shape, img_aug, keyPsoi, distance_maps_normalized, type_label, origi_keyposi) in enumerate(validDataLoader):
                    if config['tranform'] == 0:
                        img_aug = img_aug[:, np.newaxis, :, :]
                    # print(img_aug.shape)
                    img_aug = Variable(img_aug).float().cuda()
                    heatmaps_targets = Variable(distance_maps_normalized, requires_grad=True).float().cuda()
                    # keyPsoi= Variable(keyPsoi).float().cuda()

                    optimizer.zero_grad()
                    outputs = net(img_aug)

                    valid_loss = 100 * criterion(outputs, heatmaps_targets)

                    all_peak_points = get_peak_points(outputs.cpu().data.numpy())

                    plot_img = plotPoint(img_aug[0][0].cpu().numpy(), all_peak_points[0])
                    viz.images(plot_img, win='outputIMG', opts={'title': 'output_imgs'})

                    # print("all_peak_points", all_peak_points)
                    # print("keyPsoi", keyPsoi.cpu().data.numpy())
                    staticPosi_6pixel_coor += validDistance(all_peak_points, keyPsoi.cpu().data.numpy())
                    staticPosi_6pixel_coor = np.array(staticPosi_6pixel_coor)
                    running_loss += valid_loss.item()
                    torch.cuda.empty_cache()
                end = time()
                running_loss /= valid_num
                print("#################### valid")
                print("staticPosi_6pixel_coor", staticPosi_6pixel_coor)
                staticPosi_6pixel_coor /= float(valid_num)
                print("each point acc is :", staticPosi_6pixel_coor)
                accPoint = staticPosi_6pixel_coor.sum()/float(11)
                print('[valid_Epoch] valid_loss:{:.10} Agerage_accPoint_6coor:{} valid_time:{:.2f}'.format(
                    running_loss, accPoint, end - start))
                viz.line([running_loss], [epoch], win='valid_train_loss', update='append')
                viz.line([accPoint], [epoch], win='acc_coor', update='append')
                print("this valid epoch cost time: ", end - start)
                print("   ")
                print("############################################training ")


