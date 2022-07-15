import torch
import torch.nn as nn
from pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG, PointnetSAModule
import pointnet2.pytorch_utils as pt_utils


def get_model(input_channels=0):
    return HandPointnet(input_channels=input_channels)


NPOINTS = [512, 64, 1]
RADIUS = [0.1, 0.2, 0.4]
NSAMPLE = [32,64,64]
MLPS = [[64, 128], [128, 256],[512, 1024]]
FP_MLPS = [[128, 128], [256, 256], [512, 512], [512, 512]]
CLS_FC = [1024,512,128]
DP_RATIO = 0.5


class HandPointnet(nn.Module):
    def __init__(self, input_channels=6):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        skip_channel_list = [input_channels]
        for k in range(NPOINTS.__len__()):
            mlps = MLPS[k].copy()
            channel_out = 0
            mlps = [channel_in]+(mlps)
            
            channel_out += mlps[-1]
            # for idx in range(mlps.__len__()):
                
            #     mlps[idx] = [channel_in] + mlps[idx]
            #     channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModule(
                    npoint=NPOINTS[k],
                    radius=RADIUS[k],
                    nsample=NSAMPLE[k],
                    mlp=mlps,
                    use_xyz=True,
                    bn=True
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(FP_MLPS.__len__()):
            pre_channel = FP_MLPS[k + 1][-1] if k + 1 < len(FP_MLPS) else channel_out
            self.FP_modules.append(
                PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + FP_MLPS[k])
            )

        cls_layers = []
        #pre_channel = FP_MLPS[0][-1]
        pre_channel = MLPS[-1][-1]        
        for k in range(0, CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, CLS_FC[k], bn=True))
            pre_channel = CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, 63, activation=None))
        cls_layers.insert(1, nn.Dropout(0.5))
        self.cls_layer = nn.Sequential(*cls_layers)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        # for i in range(-1, -(len(self.FP_modules) + 1), -1):
        #     l_features[i - 1] = self.FP_modules[i](
        #         l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
        #     )
        print('size',l_features[-1].size())
        pred_cls = self.cls_layer(l_features[-1]).contiguous()  # (B, N, 63)
        return pred_cls

if __name__ == '_main__':
    get_model(input_channels=6)