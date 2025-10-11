import torch
import torch.nn as nn
import cv2
import torch.utils.data
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import math
from .submodule import *
from .submodule import MySlic
from .submodule import DWTModule

class TANet(nn.Module):
    def __init__(self, maxdisp):
        super(TANet, self).__init__()
        self.maxdisp = maxdisp
        self.feature_extraction = feature_extraction()
        self.fpn = FeaturePyrmaid()
        self.dwt_adjust = nn.Sequential(
            nn.Conv2d(12, 32, 1),
            nn.ReLU(inplace=True)
        )
        self.mylayer = nn.Sequential(convbn(320, 128, 3, 1, 1),  # 从320维降维到32维
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))

        self.corr_costvolume_s = CostVolume(self.maxdisp // 16, 'correlation')  # 12
        self.corr_costvolume_m = CostVolume(self.maxdisp // 8, 'correlation')  # 24
        self.corr_costvolume_l = CostVolume(self.maxdisp // 4, 'correlation')  # 48

        self.cost_s_combine = nn.Sequential(
            convbn(13, 12, 1, 1, 0),
            nn.ReLU(inplace=True),
        )

        self.sp_combine = nn.Sequential(convbn(33, 32, 1, 1, 0),
                                          nn.Sigmoid(),
                                        # nn.Dropout(p=0.1)
                                        )

        self.get_weight1 = nn.Sequential(convbn(13, 12, 3, 1, 1),
                                         nn.ReLU(inplace=True),
                                         convbn(12, 12, 3, 1, 1))
        self.get_weight2 = hourglass_2d_cbam_sigmoid(12)
        self.classif_weight = nn.Sequential(convbn(12, 12, 3, 1, 1),
                                            nn.ReLU(inplace=True))
        #self.get_superpixel = MySlic()

        self.dres_s = hourglass_2d_cbam(12)
        self.dres_m = hourglass_2d_cam(24)
        self.dres_l = hourglass_2d_cam(48)

        self.ffm_m_layer1 = nn.Sequential(convbn(12, 24, 3, 1, 1),
                                          nn.ReLU(inplace=True))
        self.ffm_m_layer2 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                          convbn(24, 24, 3, 1, 1),
                                          nn.ReLU(inplace=True),
                                          convbn(24, 24, 3, 1, 1),
                                          nn.Sigmoid())

        self.ffm_l_layer1 = nn.Sequential(convbn(24, 48, 3, 1, 1),
                                          nn.ReLU(inplace=True))
        self.ffm_l_layer2 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                          convbn(48, 48, 3, 1, 1),
                                          nn.ReLU(inplace=True),
                                          convbn(48, 48, 3, 1, 1),
                                          nn.Sigmoid())

        self.classif_l = nn.Sequential(convbn(48, 48, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(48, self.maxdisp, kernel_size=3, padding=1, stride=1, bias=False))

        self.conv_hh_adjust = nn.Conv2d(in_channels=32, out_channels=12, kernel_size=1)
        self.conv_lh_hl_adjust = nn.Conv2d(in_channels=32, out_channels=24, kernel_size=1)
        self.conv_ll_adjust = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=1)
        self.dwt_adapter = nn.ModuleDict({
            'high': nn.Sequential(
                nn.Conv2d(12, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 1)
            ),
            'mid': nn.Sequential(
                nn.Conv2d(12, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 1)
            ),
            'low': nn.Sequential(
                nn.Conv2d(12, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, 1)
            )
        })

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def corr_costvolume_s_warp(self, feat_l, feat_r, maxdisp, disp, stride=1):
        size = feat_l.size()
        batch_disp = disp[:, None, :, :, :].repeat(1, maxdisp * 2 - 1, 1, 1, 1).view(-1, 1, size[-2], size[-1])
        batch_shift = torch.arange(-maxdisp + 1, maxdisp, device='cuda').repeat(size[0])[:, None, None, None] * stride
        batch_disp = batch_disp - batch_shift.float()
        batch_feat_l = feat_l[:, None, :, :, :].repeat(1, maxdisp * 2 - 1, 1, 1, 1).view(-1, size[-3], size[-2],                                                                                   size[-1])
        batch_feat_r = feat_r[:, None, :, :, :].repeat(1, maxdisp * 2 - 1, 1, 1, 1).view(-1, size[-3], size[-2],                                                                                    size[-1])
        cost = torch.norm(batch_feat_l - self.warp(batch_feat_r, batch_disp), 1, 1)
        cost = cost.view(size[0], -1, size[2], size[3])
        return cost.contiguous()

    def warp(self, x, disp):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        xx = torch.arange(0, W, device='cuda').view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device='cuda').view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        vgrid = torch.cat((xx, yy), 1).float()
        vgrid[:, :1, :, :] = vgrid[:, :1, :, :] - disp
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        return output


    def cost_with_predisp(self, left_feature, right_feature, weight=None):
        cost = self.corr_costvolume_s(left_feature, right_feature) * weight
        return cost

    def forward(self, left, right,dwt_img):


        left_feature = self.feature_extraction(left)
        left_feature = self.mylayer(left_feature)
        right_feature = self.feature_extraction(right)
        right_feature = self.mylayer(right_feature)


        l_f_l, l_f_m, l_f_s = self.fpn(left_feature)
        r_f_l, r_f_m, r_f_s = self.fpn(right_feature)

        cost_s = self.corr_costvolume_s(l_f_s, r_f_s)
        cost_m = self.corr_costvolume_m(l_f_m, r_f_m)


        cost_l =  cost_l_corr

        out_s = self.dres_s(cost_s)
        out_m = self.dres_m(cost_m)
        out_l = self.dres_l(cost_l)

        out_s = F.interpolate(out_s, (out_m.size(2), out_m.size(3)), mode='bilinear')
        out_s = self.ffm_m_layer1(out_s)
        out_m1 = out_m + out_s * self.ffm_m_layer2(out_s + out_m)
        out_m1 = F.interpolate(out_m1, (out_l.size(2), out_l.size(3)), mode='bilinear')

        out_m1 = self.ffm_l_layer1(out_m1)
        out_l1 = out_l + out_m1 * self.ffm_l_layer2(out_m1 + out_l)

        out_l1 = self.classif_l(out_l1)
        out_l1 = F.interpolate(out_l1, (left.size(2), left.size(3)), mode='bilinear')

        pred = F.softmax(out_l1, dim=1)
        pred = disparityregression(self.maxdisp)(pred)

        return pred



