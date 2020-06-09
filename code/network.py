import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ============ Unet Components ===============
class AttentionMask(nn.Module):
    """ Create attention mask for image embedding (key),
    letting attention score be conditional on voice embedding (query).
    The final output is modified image embedding (context value) ingested by the next layer.
    """
    def __init__(self):
        super(AttentionMask, self).__init__()
        self.act1 = nn.Softmax(dim=-1)
        self.act2 = nn.Sigmoid()

    def forward(self, key, query):
        assert key.size(0) == query.size(0)
        shape = key.shape  # batch_size == shape[0]

        # 1. create attention matrix where a_ij represents
        # how similar key_i is to query_j
        # key --> (bs, #channel * w * h), query --> (bs, #channel * 1 * 1)
        # note that query feature_dim=64
        att = torch.einsum('ij,ik->ijk',
                           [key.view(shape[0], -1), query.view(shape[0], -1)])

        # 2. softmax activation on each row, namely for all j corresponding to each i
        att = self.act1(att)

        # 3. generate attention mask, conditional on query
        att = torch.einsum('ijk,ikl->ijl', [att, query.view(shape[0], -1, 1)])

        # 4. sigmoid activation on each row i (one element)
        att = self.act2(att)

        # 5. resize and generate context value
        return key * att.view(shape)


class SigmoidLinearMask(nn.Module):
    def __init__(self, shape, embedding_dim=64):
        super(SigmoidLinearMask, self).__init__()
        if len(shape) == 4:
            out_features = shape[1] * shape[2] * shape[3]
            self.shape = shape[1:]
            self.linear = nn.Linear(embedding_dim, out_features)
        if len(shape) == 1:
            out_features = shape[0]
            self.shape = shape
            self.linear = nn.Linear(embedding_dim, out_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, w, embedding):
        mask = self.linear(embedding)
        mask = self.sigmoid(mask)
        mask = mask.reshape(self.shape)
        return mask * w


class Conv2dWMask(nn.Module):
    """Use a mask to let convolutional layers's weight W and b be conditional on auxiliary embedding
    """
    def __init__(self, in_c, out_c, kernel_size, padding, embedding_dim=64):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding)
        self.wshape = [out_c, in_c, kernel_size, kernel_size]
        self.bshape = [out_c]
        self.wlinear = SigmoidLinearMask(self.wshape, embedding_dim)
        self.blinear = SigmoidLinearMask(self.bshape, embedding_dim)
        self.mask = AttentionMask()

    def forward(self, x, embedding):
        # shape of voice_embedding: (batch_size, 64)
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if gender == 'm':
            gender = torch.ones((1)).to(device)
        elif gender == 'f':
            gender = -1 * torch.ones((1)).to(device)
        """

        # embedding = torch.mean(embedding, dim=0)  # TODO: can NOT use mean function!

        new_weight = self.wlinear(self.conv.weight, embedding)
        new_b = self.blinear(self.conv.bias, embedding)
        output = F.conv2d(x, new_weight, new_b, padding=self.conv.padding)

        return output


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mode='up', concat=False):
        super().__init__()
        self.mode = mode
        if mode == 'down':
            self.double_conv = self._double_conv(in_channels, out_channels)
        else:
            # self.double_conv = self._double_conv_w_mask(in_channels, out_channels)
            self.conv1 = Conv2dWMask(in_channels, out_channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = Conv2dWMask(out_channels, out_channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu2 = nn.ReLU(inplace=True)
        """
        if concat:
            self.double_conv = self.double_conv_v2(in_channels, out_channels)
        else:
            self.double_conv = self.double_conv(in_channels, out_channels)
        """

    def forward(self, x, embedding=None):
        if self.mode == 'down':
            return self.double_conv(x)
        else:
            out = self.conv1(x, embedding)
            out = self.bn1(out)
            out = self.relu1(out)
            out = self.conv2(out, embedding)
            out = self.bn2(out)
            out = self.relu2(out)
            return out

    def _double_conv_w_mask(self, in_channels, out_channels):
        return nn.Sequential(
            Conv2dWMask(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            Conv2dWMask(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    # TODO: ???
    def _double_conv_v2(self, in_channels, out_channels):
        """for upsampling where input was concatenated with label"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels, mode='down')
        )

    def forward(self, x):
        return self.avgpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # TODO: remove the device variable in the future
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels, mode='up')

    def forward(self, x1, x2, label):
        """
        :param x1:
        :param x2:
        :param label: could be gender scalar or voice embedding vector (batch_size x [scalar or vector])
        :return:
        """
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # if type(label) == int:
        #     if label == 0:
        #         labels = torch.zeros((x1.shape[0], 2*x1.shape[1], x1.shape[2], x1.shape[3])).to(self.device)
        #     else:
        #         labels = torch.ones((x1.shape[0], 2*x1.shape[1], x1.shape[2], x1.shape[3])).to(self.device)
        # else:
        #     labels = []
        #     labels = [torch.ones((2*x1.shape[1], x1.shape[2], x1.shape[3]))
        #               if i == 1 else torch.zeros((2*x1.shape[1], x1.shape[2], x1.shape[3])) for i in label]
        #     labels = torch.stack(labels).to(self.device)
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, embedding=label)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ============ Unet ===============
class UNet(nn.Module):
    def __init__(self, in_channels, channels, out_channels, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64, mode='down')  # original input image 64 x 64
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)
        print('UNet initialized!')

    def forward(self, x, embedding):
        # print(x.shape)
        x1 = self.inc(x)
        # print('x1 shape = ', x1.shape)
        x2 = self.down1(x1)
        # print('x2 shape = ', x2.shape)
        x3 = self.down2(x2)
        # print('x3 shape = ', x3.shape)
        x4 = self.down3(x3)
        # print('x4 shape = ', x4.shape)
        x5 = self.down4(x4)
        # print('UNet x5 shape = ', x5.shape)
        # print('\n')
        x = self.up1(x5, x4, embedding)
        # print('x6 shape = ', x.shape)
        x = self.up2(x, x3, embedding)
        # print('x7 shape = ', x.shape)
        x = self.up3(x, x2, embedding)
        # print('x8 shape = ', x.shape)
        x = self.up4(x, x1, embedding)
        # print('x9 shape = ', x.shape)
        out = self.outc(x)
        # print('out shape = ', out.shape)
        out = torch.tanh(out)
        return out


# ============ Embedding Networks ===============
class VoiceEmbedNet(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(VoiceEmbedNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_channel, channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[0], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[0], channels[1], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[1], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[1], channels[2], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[2], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[2], channels[3], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[3], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[3], output_channel, 3, 2, 1, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool1d(x, x.size()[2], stride=1)
        x = x.view(x.size()[0], -1, 1, 1)
        return x


# TODO: deprecated
class FaceEmbedNet(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(FaceEmbedNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channel, channels[0], 1, 1, 0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels[0], channels[1], 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels[1], channels[2], 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels[2], channels[3], 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels[3], channels[4], 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels[4], output_channel, 4, 1, 0, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


# ============ Yandong GAN ===============
# TODO: deprecated
class Generator(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(input_channel, channels[0], 4, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[0], channels[1], 4, 2, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[1], channels[2], 4, 2, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[2], channels[3], 4, 2, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[3], channels[4], 4, 2, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[4], output_channel, 1, 1, 0, bias=True),
        )
    def forward(self, x):
        x = self.model(x)
        return x


# ============ "Fc" from "Reconstruct face from voice" paper ===============
class Classifier(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(Classifier, self).__init__()
        self.model = nn.Linear(input_channel, output_channel, bias=False)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.model(x)
        return x


# ============ Network Type Switch ===============
def get_network(net_type, params, train=True):
    net_params = params[net_type]

    net = net_params['network'](
        net_params['input_channel'],
        net_params['channels'],
        net_params['output_channel'])

    if params['GPU']:
        net.cuda()

    if train:
        net.train()
        optimizer = optim.Adam(net.parameters(),
                               lr=params['lr'],
                               betas=(params['beta1'], params['beta2']))
    else:
        net.eval()
        net.load_state_dict(torch.load(net_params['model_path'], map_location=torch.device('cpu')))
        optimizer = None
    return net, optimizer