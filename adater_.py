import math
import torch
import torch.nn as nn
from collections import OrderedDict
DWCONV3D_DISABLE_CUDNN = True



class Adapter(nn.Module):

    def __init__(self, in_channels, adapter_channels, kernel_size, T):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv3d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=tuple(x // 2 for x in kernel_size),
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        self.T = T
        self.offset_adapter = OffsetAdapter(in_channels, adapter_channels, (1, 3, 3), T)
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        offset = self.offset_adapter(x)
        T = self.T
        BT, L, C = x.size()
        B = BT // T
        Ca = self.conv.in_channels
        H = W = round(math.sqrt(L - 1))

        assert L - 1 == H * W
        x_id = x
        x = x[:, 1:, :]
        x = self.fc1(x)
        x = x.view(B, T, H, W, Ca).permute(0, 4, 1, 2, 3).contiguous()

        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        x = self.conv(x)
        torch.backends.cudnn.enabled = cudnn_enabled

        x = x.permute(0, 2, 3, 4, 1).contiguous().view(BT, L - 1, Ca)
        x = self.fc2(x) + offset
        x_id[:, 1:, :] += x
        return x_id

class Adapter1(nn.Module):

    def __init__(self, in_channels, adapter_channels, kernel_size = 3, T = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv1d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        self.T = T
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        T = self.T
        BT, C = x.size()
        B = BT // T
        Ca = self.conv.in_channels
        x_id = x
        x = self.fc1(x)
        x = x.view(B, T, Ca).permute(0, 2, 1).contiguous().view(B, Ca, T)  #

        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        x = self.conv(x)
        torch.backends.cudnn.enabled = cudnn_enabled

        x = x.permute(0, 2, 1).contiguous().view(-1, Ca)
        x = self.fc2(x)
        x_id = x + x_id
        # pdb.set_trace()
        return x_id

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Adapter2(nn.Module):

    def __init__(self, in_channels, kernel_size = 3):
        super().__init__()
        adapter_channels = int(in_channels*(1/4))
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv1d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)

        self.offset_adapter = OffsetAdapter(in_channels, adapter_channels, (1, 3, 3))
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_in", nn.Linear(in_channels, adapter_channels)),
            ("gelu", QuickGELU()),
            # ("drop", nn.Dropout(0.5)),
            ("c_out", nn.Linear(adapter_channels, in_channels))
        ]))

    def forward(self, x,T):
        offset = self.offset_adapter(x,T)
        # pdb.set_trace()
        L, N, C = x.size()
        B = N // T
        Ca = self.conv.in_channels
        x_id = x
        x_id = self.mlp(x_id)
        x = self.fc1(x)
        x = x.permute(1, 0, 2).contiguous().view(B, T, L, Ca)
        x = x.permute(0, 2, 3, 1).contiguous().view(B * L, Ca, T)  #


        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        x = self.conv(x)
        torch.backends.cudnn.enabled = cudnn_enabled

        x = x.permute(0, 2, 1).contiguous().view(B, L, T, Ca)
        x = x.permute(0, 2, 1, 3).contiguous().view(N, L, Ca).permute(1, 0, 2)
        x = self.fc2(x)
        x_id[1:, :, :] += offset
        x_id = x + x_id

        return x_id


class OffsetAdapter(nn.Module):

    def __init__(self, in_channels, adapter_channels, kernel_size):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv3d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=tuple(x // 2 for x in kernel_size),
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)

        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)



    def forward(self, x,T):
        L, N, C = x.size()
        B = N // T
        Ca = self.conv.in_channels
        H = W = round(math.sqrt(L - 1))
        assert L - 1 == H * W
        x = x.permute(1,0,2)
        x = x[:, 1:, :].view(B, T, -1, C)
        former_id = [0] + [i for i in range(T)][:-1]
        x_former = x[:, former_id]
        # pdb.set_trace()
        offset = x - x_former
        offset = self.fc1(offset)
        offset = offset.view(B, T, H, W, Ca).permute(0, 4, 1, 2, 3).contiguous()
        #
        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        offset = self.conv(offset)
        torch.backends.cudnn.enabled = cudnn_enabled

        offset = offset.permute(0, 2, 3, 4, 1).contiguous().view(N, L - 1, Ca)
        offset = self.fc2(offset).permute(1,0,2)

        # x_id[:, 1:, :] += offset
        return offset

class TextAdapter(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        r = 1/4
        self.textad_fc1 = nn.Linear(in_channels, int(in_channels*r))
        self.textad_gelu = nn.GELU()
        self.textad_fc2 = nn.Linear(int(in_channels*r), in_channels)
        nn.init.constant_(self.textad_fc1.bias, 0.)
        nn.init.constant_(self.textad_fc2.bias, 0.)

    def forward(self, x):
        # pdb.set_trace()
        x1 = self.textad_fc1(x)
        x1 = self.textad_gelu(x1)
        x1 = self.textad_fc2(x1)
        x = x + x1
        return x

def test1():
    x = torch.rand(197, 296, 768)  # T = 2
    # x = x.reshape(-1, 224, 768)
    import time
    start_time = time.time()
    model = Adapter2(768)
    y = model(x,148)
    print(y.shape)
    end_time = time.time()
    print(end_time-start_time)

def test2():
    x = torch.rand(6, 2, 224, 768)  # T = 2
    x = x.reshape(-1, 224, 768)
    import time
    start_time = time.time()
    model = TextAdapter(768)
    y = model(x)
    print(y.shape)
    end_time = time.time()
    print(end_time-start_time)

# test1()
#
# test2()