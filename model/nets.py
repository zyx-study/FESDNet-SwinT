import torch
import torch.nn as nn
import torch.nn.functional as F
from .cswin import CSWinBlock


class FrequencyDomainProcessing(nn.Module):
    def __init__(self, enhancement_factor=0.5):
        super(FrequencyDomainProcessing, self).__init__()
        self.enhancement_factor = enhancement_factor  # Controls the enhancement factor for high-frequency components

    def forward(self, x):
        # Transform the input to the frequency domain
        x_freq = torch.fft.fft2(x)
        x_freq = torch.fft.fftshift(x_freq)  # Shift the zero frequency component to the center

        # Apply high-pass filter processing
        x_freq = self.enhance_high_frequencies(x_freq)

        # Transform back to the spatial domain
        x_freq = torch.fft.ifftshift(x_freq)  # Shift the frequency components back to the original position
        x_processed = torch.fft.ifft2(
            x_freq).real  # Convert the processed frequency data back to the spatial domain (real part)

        return x_processed

    def enhance_high_frequencies(self, x_freq):
        # Get the shape of the frequency components
        freq_shape = x_freq.shape
        center = [s // 2 for s in freq_shape[2:]]  # Calculate the center of the frequency grid
        Y, X = torch.meshgrid(
            [torch.arange(s, device=x_freq.device) for s in freq_shape[2:]])  # Create a meshgrid for frequency space
        distance = torch.sqrt(
            (X - center[1]) ** 2 + (Y - center[0]) ** 2)  # Calculate the distance from the center (frequency distance)

        # Enhance high-frequency components based on their distance from the center
        high_freq_mask = distance / distance.max()  # Normalize the distance, with farther distances indicating higher frequencies
        enhancement_mask = 1 + self.enhancement_factor * high_freq_mask  # Enhance the high-frequency components

        return x_freq * enhancement_mask  # Apply the enhancement mask to the frequency components


class my_model(nn.Module):
    def __init__(self,
                 en_feature_num,
                 en_inter_num,
                 de_feature_num,
                 de_inter_num,
                 sam_number=1,
                 use_transformer=True
                 ):
        super(my_model, self).__init__()
        self.encoder = Encoder(feature_num=en_feature_num,
                               inter_num=en_inter_num,
                               sam_number=sam_number,
                               use_transformer=use_transformer)
        self.decoder = Decoder(en_num=en_feature_num,
                               feature_num=de_feature_num,
                               inter_num=de_inter_num,
                               sam_number=sam_number)

    def forward(self, x):
        y_1, y_2, y_3 = self.encoder(x)
        out_1, out_2, out_3 = self.decoder(y_1, y_2, y_3)

        return out_1, out_2, out_3

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)


class Decoder(nn.Module):
    def __init__(self, en_num, feature_num, inter_num, sam_number):
        super(Decoder, self).__init__()
        self.preconv_3 = conv_relu(4 * en_num, feature_num, 3, padding=1)
        self.decoder_3 = Decoder_Level(feature_num, inter_num, sam_number)

        self.preconv_2 = conv_relu(2 * en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_2 = Decoder_Level(feature_num, inter_num, sam_number)

        self.preconv_1 = conv_relu(en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_1 = Decoder_Level(feature_num, inter_num, sam_number)

    def forward(self, y_1, y_2, y_3):
        x_3 = y_3
        x_3 = self.preconv_3(x_3)
        out_3, feat_3 = self.decoder_3(x_3)

        x_2 = torch.cat([y_2, feat_3], dim=1)
        x_2 = self.preconv_2(x_2)
        out_2, feat_2 = self.decoder_2(x_2)

        x_1 = torch.cat([y_1, feat_2], dim=1)
        x_1 = self.preconv_1(x_1)
        out_1 = self.decoder_1(x_1, feat=False)

        return out_1, out_2, out_3


class Encoder(nn.Module):
    def __init__(self, feature_num, inter_num, sam_number, use_transformer=False):  # 添加 use_transformer 参数
        super(Encoder, self).__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(12, feature_num, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True)
        )
        self.encoder_1 = Encoder_Level(feature_num, inter_num, level=1, sam_number=sam_number,
                                       use_transformer=use_transformer, dim=48, reso=128)
        self.encoder_2 = Encoder_Level(2 * feature_num, inter_num, level=2, sam_number=sam_number,
                                       use_transformer=use_transformer, dim=96, reso=64)
        self.encoder_3 = Encoder_Level(4 * feature_num, inter_num, level=3, sam_number=sam_number,
                                       use_transformer=use_transformer, dim=192, reso=32)

    def forward(self, x):
        x = F.pixel_unshuffle(x, 2)
        x = self.conv_first(x)

        out_feature_1, down_feature_1 = self.encoder_1(x)  # 1 48 128 128
        out_feature_2, down_feature_2 = self.encoder_2(down_feature_1)
        out_feature_3 = self.encoder_3(down_feature_2)

        return out_feature_1, out_feature_2, out_feature_3


class Encoder_Level(nn.Module):
    def __init__(self, feature_num, inter_num, level, sam_number, use_transformer=False, dim=56, reso=56):
        super(Encoder_Level, self).__init__()
        self.rdb = RDB(in_channel=feature_num, d_list=(1, 2, 1), inter_num=inter_num)
        self.frequency_processing = FrequencyDomainProcessing()
        self.sam_blocks = nn.ModuleList()
        for _ in range(sam_number):
            sam_block = SAM(in_channel=feature_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_num)
            self.sam_blocks.append(sam_block)

        self.use_transformer = use_transformer
        if self.use_transformer:
            self.transformer = CSWinBlock(dim=dim, num_heads=4, reso=reso)

        if level < 3:
            self.down = nn.Sequential(
                nn.Conv2d(feature_num, 2 * feature_num, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )
        self.level = level

    def forward(self, x):
        out_feature = self.rdb(x)
        # 频域处理
        out_feature = self.frequency_processing(out_feature)

        # 应用 Transformer 块（如果启用）
        if self.use_transformer:
            B, C, H, W = out_feature.size()
            out_feature = out_feature.permute(0, 2, 3, 1)
            out_feature = out_feature.view(B, H * W, C)
            out_feature = self.transformer(out_feature)
            out_feature = out_feature.view(B, H, W, C).permute(0, 3, 1, 2)

        for sam_block in self.sam_blocks:
            out_feature = sam_block(out_feature)
        if self.level < 3:
            down_feature = self.down(out_feature)
            return out_feature, down_feature
        return out_feature


class Decoder_Level(nn.Module):
    def __init__(self, feature_num, inter_num, sam_number):
        super(Decoder_Level, self).__init__()
        self.rdb = RDB(feature_num, (1, 2, 1), inter_num)
        self.sam_blocks = nn.ModuleList()
        for _ in range(sam_number):
            sam_block = SAM(in_channel=feature_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_num)
            self.sam_blocks.append(sam_block)
        self.conv = conv(in_channel=feature_num, out_channel=12, kernel_size=3, padding=1)

    def forward(self, x, feat=True):
        x = self.rdb(x)
        for sam_block in self.sam_blocks:
            x = sam_block(x)
        out = self.conv(x)
        out = F.pixel_shuffle(out, 2)

        if feat:
            feature = F.interpolate(x, scale_factor=2, mode='bilinear')
            return out, feature
        else:
            return out


class DB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(DB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)
        t = self.conv_post(t)
        return t


class SAM(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(SAM, self).__init__()
        self.basic_block = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_2 = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_4 = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.fusion = CSAF(3 * in_channel)

    def forward(self, x):
        x_0 = x
        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')

        y_0 = self.basic_block(x_0)
        y_2 = self.basic_block_2(x_2)
        y_4 = self.basic_block_4(x_4)

        y_2 = F.interpolate(y_2, scale_factor=2, mode='bilinear')
        y_4 = F.interpolate(y_4, scale_factor=4, mode='bilinear')

        y = self.fusion(y_0, y_2, y_4)
        y = x + y

        return y


class CSAF(nn.Module):
    def __init__(self, in_chnls, ratio=4):
        super(CSAF, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress1 = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.compress2 = nn.Conv2d(in_chnls // ratio, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x0, x2, x4):
        out0 = self.squeeze(x0)
        out2 = self.squeeze(x2)
        out4 = self.squeeze(x4)
        out = torch.cat([out0, out2, out4], dim=1)
        out = self.compress1(out)
        out = F.relu(out)
        out = self.compress2(out)
        out = F.relu(out)
        out = self.excitation(out)
        out = F.sigmoid(out)
        w0, w2, w4 = torch.chunk(out, 3, dim=1)
        x = x0 * w0 + x2 * w2 + x4 * w4

        return x


class RDB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(RDB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)

        t = self.conv_post(t)
        return t + x


class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=True, dilation=dilation_rate)

    def forward(self, x_input):
        out = self.conv(x_input)
        return out


class conv_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=True, dilation=dilation_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_input):
        out = self.conv(x_input)
        return out


if __name__ == '__main__':
    from ptflops import get_model_complexity_info

    # Set the device to CUDA if a GPU is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model hyperparameters
    EN_FEATURE_NUM = 48
    EN_INTER_NUM = 32
    DE_FEATURE_NUM = 64
    DE_INTER_NUM = 32
    SAM_NUMBER = 1

    # Instantiate the model and move it to the selected device (GPU/CPU)
    model = my_model(en_feature_num=EN_FEATURE_NUM,
                     en_inter_num=EN_INTER_NUM,
                     de_feature_num=DE_FEATURE_NUM,
                     de_inter_num=DE_INTER_NUM,
                     sam_number=SAM_NUMBER).to(device)

    # Calculate model complexity and number of parameters using ptflops
    with torch.cuda.device(0):
        # Get the computational complexity (MACs) and the number of parameters
        macs, params = get_model_complexity_info(model, (3, 256, 256),
                                                 as_strings=True, print_per_layer_stat=True,
                                                 verbose=True)

    # Print the results in a readable format
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))  # Display MACs
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))  # Display number of parameters

    """
    Example output:
    Computational complexity:       19.1 GMac
    Number of parameters:           6.52 M  
    """
