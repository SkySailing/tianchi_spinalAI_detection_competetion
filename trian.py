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
import matplotlib.pyplot as plt
import torch
from torch.backends import cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from time import time
from time import strftime
import torchvision.transforms as transforms


from sipnalDataset import mydata, mydata_classifi
from torchvision.models.vgg import VGG
from unet import unet_model, unet_parts
from fcn import FCN8s,VGGNet
from classi_models.resnet import resnet18,ResNet,BasicBlock
# from apex import amp, optimizers
# from apex.parallel import DistributedDataParallel

#导入包
from visdom import Visdom

config = dict()

# be care to set config before submit
config['showFlag'] = 0
config['epoch_num'] = 900
config['tranform'] = 1
config['saveName'] = 'UNet_double_PosiLoss_classi_100v20_final_model.ckpt'
config['checkout'] = "UNet_double_PosiLoss_classi_ptrianed_model.ckpt"
config['vert_net_saveName'] = "vert_resNet_final0919_model.ckpt"
config['vert_checkout'] = "vert_resNet_final0919_model.ckpt"
config['disc_net_saveName'] = "disc_resNet_final0919_model.ckpt"
config['disc_checkout'] = "disc_resNet_final0919_model.ckpt"
# UNet_double_PosiLoss_pretrian_model.ckpt
# 'FCN8_orignal_pretrian_model.ckpt'
# 'UNet_double_orignal_pretrain_model.ckpt'
# pretrian_Unet_final_model.ckpt
# pretrainNet_model.ckpt
# pretrian_Unet_final_model.ckpt

if config['showFlag'] == 1:
    config["valid"] = 1
else:
    config["valid"] = 0
config['lr'] = 0.00001
config['momentum'] = 0.9
config['weight_decay'] = 1e-4
config['start_epoch'] = 0
config['batch_size'] = 10
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
    viz.line([0.], [0.], win='vert_loss', opts=dict(title='vert_loss'))  # single-line
    viz.line([0.], [0.], win='disc_loss', opts=dict(title='disc_loss'))  # single-line
    viz.line([0.], [0.], win='vert_valid_acc', opts=dict(title='vert_valid_acc'))  # single-line
    viz.line([0.], [0.], win='disc_valid_acc', opts=dict(title='disc_valid_acc'))  # single-line

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
        # diffuse2D
        transforms.ToTensor()
        # ,normTransform
    ])

    validTransform = transforms.Compose([
        # diffuse2D
        transforms.ToTensor()
        # ,normTransform
    ])
else:
    trainTransform = None
    validTransform = None

def weights_init_old(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.item)
        nn.init.constant_(m.bias.item, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias:
            nn.init.constant_(m.bias.data, 0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.01)
        m.bias.data.zero_()


class posiLoss(nn.Module):
    def __init__(self,alph=50, beta=100):
        super(posiLoss, self).__init__()
        self.alph = alph
        self.beta = beta
        self.criterion_part1 = nn.MSELoss()
        self.sz = 11
        self.topKposi = 5

    def posiLoss(self, pred_heatmap, gt_heatmap):
        n,c,h,w = pred_heatmap.shape
        n_loss = list()
        for n_idx in range(n):
            # print("n:",n)
            pred_posiRation = self.relationPosi(pred_heatmap[n_idx])
            gt_posiRation = self.relationPosi(gt_heatmap[n_idx])
            psoiLossL = torch.Tensor(torch.abs(torch.sub(pred_posiRation, gt_posiRation))).cuda()
            # print("psoiLoss", psoiLossL)
            ret_psoiLoss = torch.sum(torch.topk(psoiLossL, self.topKposi).values)
            # print("ret_psoiLoss", ret_psoiLoss)
            n_loss.append(ret_psoiLoss)
        ret_n_loss = torch.mean(torch.Tensor(n_loss))
        return ret_n_loss

    def relationPosi(self, heatmap):
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

def corpRectangele(img, point, CropSz):
    """

    Args:
        img:
        point:
        CropSz: wdith * hight

    Returns:

    """
    cropIMG_vert = []
    cropIMG_disc = []
    img = img.cpu().numpy()
    for idx in range(len(point)):
        p = point[idx]
        w_corp, h_corp = CropSz
        w_center, h_center = p
        img_sz_h, img_sz_w = img.shape
        # the vert smaller 4 pixel in hight.
        # if idx & 1:
        #     h_corp = h_corp-4
        h_start = h_center - h_corp // 2
        h_end = h_center + h_corp // 2
        if h_start <= 0 or h_end >= img_sz_h:
            h_start = 0
            h_end = h_corp

        w_start = w_center - w_corp // 2
        w_end = w_center + w_corp // 2
        if w_start < 0 or w_end >= img_sz_w:
            w_start = 0
            w_end = w_corp
        new_img = img[h_start:h_end, w_start:w_end].copy()
        # new_img = new_img[:]
        # print(new_img)
        if idx & 1:
            cropIMG_vert.append(new_img)
        else:
            cropIMG_disc.append(new_img)
        # cv2.rectangle(img,(w_center-w_corp//2,h_center-h_corp//2),(w_center+w_corp//2,h_center+h_corp//2),color=(0,0,255),thickness=2)
    # plt.imshow(255*img)
    # plt.show()
    # for i in range(len(cropIMG)):
    #     plt.imshow(cropIMG[i])
    #     plt.show()
    cropIMG_vert = torch.Tensor(np.array(cropIMG_vert,dtype=float)).float().cuda()
    cropIMG_disc = torch.Tensor(np.array(cropIMG_disc,dtype=float)).float().cuda()
    return cropIMG_vert, cropIMG_disc


if __name__ == '__main__':
    if config['showFlag'] == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = "1"
        config['batch_size'] = 26
        config['valid_batch_size'] = 10
        pprint.pprint(config)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        print("######### training is start")
    torch.manual_seed(0)
    cudnn.benchmark = True

    # net = unet_model.UNet(1, 11)
    net = unet_model.UNet_double()
    # net = unet_model.UNet_twoPart(1, (6,5))
    vert_net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=2)
    disc_net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=5)
    # vert_net = resnet18(num_cls=2)
    # disc_net = resnet18(num_cls=5)

    # vgg_model = VGGNet(requires_grad = True, pretrained=False)
    # net = FCN8s(pretrained_net=vgg_model, n_class=11)

    # initial the net weight
    net = net.float().cuda()
    vert_net = vert_net.float().cuda()
    disc_net = disc_net.float().cuda()
    # net = nn.DataParallel(net).float().cuda()
    # net = nn.parallel.DistributedDataParallel(net.float().cuda()) # device_ids will include all GPU devices by default
    # net.float().cuda()
    # criterion = nn.MSELoss()
    criterion = posiLoss()
    optimizer = optim.Adam(net.parameters(), lr=config['lr'])  # good
    vert_criterion = torch.nn.CrossEntropyLoss()
    vert_optimizer = optim.Adam(vert_net.parameters(), lr=config['lr'])
    disc_criterion = torch.nn.CrossEntropyLoss()
    disc_optimizer = optim.Adam(disc_net.parameters(), lr=config['lr'])
    # optimizer = optim.Adagrad(net.parameters(), lr=config['lr']) # no good
    # optimizer = optim.RMSprop(net.parameters(), lr = config['lr'], alpha=0.9)
    # optimizer =optimizers.FusedAdam(net.parameters(), lr=config['lr'])


    #initi the train dataset
    trainDataset = mydata_classifi(config['root_dir'], config['label_dir'], transforms=trainTransform)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(trainDataset)
    trainDataLoader = DataLoader(trainDataset, config['batch_size'], True)
    sample_num = len(trainDataset)

    # initi the valid dataset
    if config["valid"] == 1:
        validDataset = mydata_classifi(config['valid_root_dir'],config['valid_label_dir'], transforms=validTransform)
        validDataLoader = DataLoader(validDataset, config['valid_batch_size'], True)
        valid_num = len(validDataset)

    if config['checkout'] != '':
        cpu_checkpoint = torch.load(config['checkout'], map_location=torch.device('cpu'))
        net.load_state_dict(cpu_checkpoint)
        # net.load_state_dict(torch.load(config['checkout']))
    else:
        # inital the network with func
        net.apply(weigth_init)

    if config['vert_checkout'] != '':
        cpu_checkpoint = torch.load(config['vert_checkout'], map_location=torch.device('cpu'))
        vert_net.load_state_dict(cpu_checkpoint)
        # net.load_state_dict(torch.load(config['checkout']))
    else:
        # inital the network with func
        vert_net.apply(weigth_init)
    if config['disc_checkout'] != '':
        cpu_checkpoint = torch.load(config['disc_checkout'], map_location=torch.device('cpu'))
        disc_net.load_state_dict(cpu_checkpoint)
        # net.load_state_dict(torch.load(config['checkout']))
    else:
        # inital the network with func
        disc_net.apply(weigth_init)

    for epoch in range(config['start_epoch'], config['epoch_num'] + config['start_epoch']):
        start = time()
        net.train()
        running_loss = 0.0
        running_loss_coor =0.0
        vert_runinng_loss = 0.0
        disc_runinng_loss = 0.0
        for i, (origi_shape, img_aug, keyPsoi, distance_maps_normalized, origi_keyposi, vert_type, cropIMG_vert, disc_type, cropIMG_disc) in enumerate(trainDataLoader):
            if config['tranform'] == 0:
                img_aug = img_aug[:, np.newaxis, :, :]
            vert_type = torch.unsqueeze(vert_type,1).long().cuda()
            disc_type = torch.unsqueeze(disc_type,1).long().cuda()
            # plt.imshow(img_aug[0, 0])
            # plt.show()
            # print(vert_type.shape)
            # print(cropIMG_vert.shape)

            img_aug = Variable(img_aug).float().cuda()
            cropIMG_vert = Variable(cropIMG_vert).float().cuda()
            cropIMG_disc = Variable(cropIMG_disc).float().cuda()

            # print(img_aug.shape)
            heatmaps_targets = Variable(distance_maps_normalized, requires_grad=True).float().cuda()
            # keyPsoi= Variable(keyPsoi).float().cuda()
            optimizer.zero_grad()

            outputs = net(img_aug)

            loss = 100 * criterion(outputs, heatmaps_targets)
            loss.backward()
            optimizer.step()
            for idx_vert in range(5):
                vert_optimizer.zero_grad()
                cropIMG_vert_part = cropIMG_vert[:, idx_vert, :, :]
                vert_output = vert_net(torch.unsqueeze(cropIMG_vert_part, 1))
                # print("vert_output",vert_output.shape)
                # print("vert_type[:, :, idx_vert, :]",  torch.squeeze(vert_type[:, :, idx_vert, :]).shape)
                vert_loss = vert_criterion(vert_output,  torch.argmax(torch.squeeze(vert_type[:, :, idx_vert, :]),-1))
                vert_runinng_loss += vert_loss.item()
                vert_loss.backward()
                vert_optimizer.step()

            for idx_disc in range(6):
                disc_optimizer.zero_grad()
                cropIMG_discpart = cropIMG_disc[:, idx_disc, :, :]
                disc_output = disc_net(torch.unsqueeze(cropIMG_discpart,1))
                disc_loss = disc_criterion(disc_output,  torch.argmax(torch.squeeze(disc_type[:, :, idx_disc, :]), -1))
                disc_runinng_loss += disc_loss.item()
                disc_loss.backward()
                disc_optimizer.step()

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
                print("    disc_loss:", disc_runinng_loss)
                print("    vert_loss:", vert_runinng_loss)

            running_loss += loss.item()
            running_loss_coor += loss_coor.item()
            torch.cuda.empty_cache()
        end = time()
        if config['showFlag'] == 1:
            print("this epoch cost time: ", end - start)

        running_loss /= sample_num
        running_loss_coor /= sample_num
        disc_runinng_loss /= (11*sample_num)
        vert_runinng_loss /= (11*sample_num)
        if config['showFlag'] == 0:
            print('[Epoch {}/{}] loss:{:.10} loss_coor:{:.10} time:{:.2f}'.format(
                epoch, config['epoch_num'], loss.item(), loss_coor.item(), end - start))
        if config['showFlag'] == 1:
            print("    disc_loss:", disc_runinng_loss)
            print("    vert_loss:", vert_runinng_loss)
            viz.line([running_loss], [epoch], win='train_loss', update='append')
            viz.line([running_loss_coor], [epoch], win='coor_loss', update='append')
            viz.line([vert_runinng_loss], [epoch], win='vert_loss', update='append')
            viz.line([disc_runinng_loss], [epoch], win='disc_loss', update='append')
        if (epoch-1) % config['save_freq'] == 0 or epoch == config['epoch_num'] - 1:
            torch.save(net.state_dict(), config['saveName'])
            torch.save(vert_net.state_dict(), config['vert_net_saveName'])
            torch.save(disc_net.state_dict(), config['disc_net_saveName'])
            # torch.save(net.state_dict(), 'Unet_lr1e-４_Adam_sigma10_epoch_{}_dataset203_model.ckpt'.format(epoch))

######################################################################　valid
        # valid the net just learned
        if config['showFlag'] == 1:
            net.eval()
            vert_net.eval()
            disc_net.eval()
            running_loss = 0.0
            vert_correct = 0
            disc_correct = 0
            staticPosi_6pixel_coor = np.array([0]*11, 'float32')
            with torch.no_grad():
                for i, (origi_shape, img_aug, keyPsoi, distance_maps_normalized, origi_keyposi, vert_type, cropIMG_vert, disc_type, cropIMG_disc) in enumerate(validDataLoader):
                    if config['tranform'] == 0:
                        img_aug = img_aug[:, np.newaxis, :, :]
                    # print(img_aug.shape)
                    img_aug = Variable(img_aug).float().cuda()
                    vert_type = torch.unsqueeze(vert_type, 1).long().cuda()
                    disc_type = torch.unsqueeze(disc_type, 1).long().cuda()
                    # print("vert_type shape:",vert_type.shape)
                    # print("disc_type shape:",disc_type.shape)
                    heatmaps_targets = Variable(distance_maps_normalized, requires_grad=True).float().cuda()
                    # keyPsoi= Variable(keyPsoi).float().cuda()

                    optimizer.zero_grad()
                    outputs = net(img_aug)

                    valid_loss = 100 * criterion(outputs, heatmaps_targets)

                    all_peak_points = get_peak_points(outputs.cpu().data.numpy())

                    plot_img = plotPoint(img_aug[0][0].cpu().numpy(), all_peak_points[0])
                    viz.images(plot_img, win='outputIMG', opts={'title': 'output_imgs'})
                    for batch_idx in range(len(img_aug)):

                        img_aug_one = img_aug[batch_idx][0]
                        cropIMG_vert, cropIMG_disc = corpRectangele(img_aug_one, all_peak_points[batch_idx], [48, 30])
                        vert_pred = vert_net(cropIMG_vert[:, np.newaxis, :, :])
                        disc_pred = disc_net(cropIMG_disc[:, np.newaxis, :, :])
                        # reduce to compare the row equal
                        # print("vert_pred:",vert_pred)
                        # print("vert_gt:",vert_type[batch_idx][0])
                        vert_correct += torch.eq(torch.argmax(vert_pred, dim=1), torch.argmax(vert_type[batch_idx][0], dim=1)).sum().cpu().numpy()
                        disc_correct += torch.eq(torch.argmax(disc_pred, dim=1), torch.argmax(disc_type[batch_idx][0], dim=1)).sum().cpu().numpy()
                        # vert_correct += np.logical_and.reduce(vert_pred.cpu().numpy(), vert_type[batch_idx][0].cpu().numpy()).sum().float()
                        # disc_correct += np.logical_and.reduce(disc_pred.cpu().numpy(), disc_type[batch_idx][0].cpu().numpy()).sum().float()

                        # print("disc_pred",disc_pred.shape)
                        # print("disc_type",disc_type.shape)

                    # print("all_peak_points", all_peak_points)
                    # print("keyPsoi", keyPsoi.cpu().data.numpy())
                    staticPosi_6pixel_coor += validDistance(all_peak_points, keyPsoi.cpu().data.numpy())
                    staticPosi_6pixel_coor = np.array(staticPosi_6pixel_coor)
                    running_loss += valid_loss.item()
                    torch.cuda.empty_cache()
                end = time()
                running_loss /= valid_num
                vert_correct /= (valid_num * 5)
                disc_correct /= (valid_num * 6)
                # vert_correct = vert_correct.cpu()
                # disc_correct = disc_correct.cpu()
                print("#################### valid")
                print("staticPosi_6pixel_coor", staticPosi_6pixel_coor)
                staticPosi_6pixel_coor /= float(valid_num)
                print("each point acc is :", staticPosi_6pixel_coor)
                accPoint = staticPosi_6pixel_coor.sum()/float(11)
                print('[valid_Epoch] valid_loss:{:.10} Agerage_accPoint_6coor:{} valid_time:{:.2f}'.format(
                    running_loss, accPoint, end - start))
                print("    vert_correct:", vert_correct)
                print("    disc_correct:", disc_correct)
                viz.line([running_loss], [epoch], win='valid_train_loss', update='append')
                viz.line([accPoint], [epoch], win='acc_coor', update='append')
                viz.line([vert_correct], [epoch], win='vert_valid_acc', update='append')
                viz.line([disc_correct], [epoch], win='disc_valid_acc', update='append')
                print("this valid epoch cost time: ", end - start)
                print("   ")
                print("############################################training ")


