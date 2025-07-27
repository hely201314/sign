import torch
import torch.nn as nn
#from layers.Embed import PositionalEmbedding
import torch.nn.functional as F


class sparse(nn.Module):
    def __init__(self,enc_in = 768,period_len = 16):
        super(sparse, self).__init__()
        self.enc_in = enc_in#D
        self.period_len = period_len
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * self.period_len // 2,
                                stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False)
    def forward(self, x):
        N,L,D = x.shape #280,197,768
        # normalization and permute     b,s,c -> b,c,s
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)#280,1,768
        x = (x - seq_mean).permute(0, 2, 1)#n,d,l = 280,768,197
        # 1D convolution aggregation
        x = self.conv1d(x.reshape(-1, 1, L)).reshape(-1, D, L) + x #280,768,197

        linear = nn.Linear(L, L, bias=False).cuda()
        # downsampling: b,c,s -> bc,n,w -> bc,w,n
        x = x.reshape(-1, L, self.period_len).permute(0, 2, 1)#(-1,24,197

        # sparse forecasting
        y = linear(x)  # bc,w,m
        #y = F.adaptive_avg_pool2d(x,(self.period_len,L))#8960,24,197
        # upsampling: bc,w,m -> bc,m,w -> b,c,s
        y = y.permute(0, 2, 1).reshape(N, D, L)#280,768,197

        # permute and denorm
        y = y.permute(0, 2, 1) + seq_mean#280,197,768

        return y


def test():
    from thop import profile

    x = torch.rand(280,197,768)

    model = sparse()
    # # for i in range(10):
    # #
    # flops, params = profile(model, (x,))
    #
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

    y = model(x)
    print(y.shape)






#test()
