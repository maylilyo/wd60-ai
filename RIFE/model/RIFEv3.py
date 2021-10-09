import torch
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP

from model.IFNetv3 import IFNet
from model.loss import EPE, SOBEL
from model.laplacian import LapLoss

device = torch.device("cuda")


class Model:
    def __init__(self, local_rank=-1):
        self.flownet = IFNet()
        self.device()

        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-4)
        self.epe = EPE()
        self.lap = LapLoss()
        self.sobel = SOBEL()

        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }

        if rank <= 0:
            self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))))

    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(), '{}/flownet.pkl'.format(path))

    def inference(self, img0, img1, scale_list=[4, 2, 1], TTA=False):
        imgs = torch.cat((img0, img1), 1)
        # flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list)
        flow, mask, merged = self.flownet(imgs, scale_list)
        if not TTA:
            return merged[2]
        else:
            # TTA: Test-Time-Augmentation
            # flow2, mask2, merged2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3), scale_list)
            flow2, mask2, merged2 = self.flownet(imgs.flip(2).flip(3), scale_list)
            return (merged[2] + merged2[2].flip(2).flip(3)) / 2

    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate

        # img0 = imgs[:, :3]
        # img1 = imgs[:, 3:]

        # imgs: (batch_size, 3 * 2, height, width)
        # img0: (batch_size, 3, height, width)
        # img1: (batch_size, 3, height, width)

        if training:
            self.train()
        else:
            self.eval()

        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(torch.cat((imgs, gt), 1), scale=[4, 2, 1])
        loss_l1 = (self.lap(merged[2], gt)).mean()
        loss_tea = (self.lap(merged_teacher, gt)).mean()

        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_tea + loss_distill * 0.01
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow[2]
            merged_teacher = merged[2]

        return merged[2], {
            'merged_tea': merged_teacher,
            'mask': mask,
            'mask_tea': mask,
            'flow': flow[2][:, :2],
            'flow_tea': flow_teacher,
            'loss_l1': loss_l1,
            'loss_tea': loss_tea,
            'loss_distill': loss_distill,
        }