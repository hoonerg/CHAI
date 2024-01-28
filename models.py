import torch
import torch.nn as nn

#########################################
#       Improve this basic model!       #
#########################################


class NaiveModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3)

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(6924, 48)

    def forward(self, pv, hrv, nonhrv, weather):
        hrv = hrv.squeeze(2)
        
        x = torch.relu(self.pool(self.conv1(hrv)))
        x = torch.relu(self.pool(self.conv2(x)))
        x = torch.relu(self.pool(self.conv3(x)))
        x = torch.relu(self.pool(self.conv4(x)))

        x = self.flatten(x)
        x = torch.concat((x, pv), dim=-1)

        x = torch.sigmoid(self.linear1(x))

        return x
    
class CNN(torch.nn.Module):
    
    def __init__(self, num_channel) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=num_channel, out_channels=num_channel*2, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(num_channel*2)
        self.conv2 = nn.Conv2d(in_channels=num_channel*2, out_channels=num_channel*2, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(num_channel*2)
        self.conv3 = nn.Conv2d(in_channels=num_channel*2, out_channels=num_channel*2, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(num_channel*2)
        self.conv4 = nn.Conv2d(in_channels=num_channel*2, out_channels=num_channel*2, kernel_size=3)
        self.bn4= nn.BatchNorm2d(num_channel*2)
        self.conv5 = nn.Conv2d(in_channels=num_channel*2, out_channels=num_channel*2, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(num_channel*2)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.output_size = num_channel * 2 * 2 * 2
        
    def forward(self, x):
        x = torch.relu(self.pool(self.bn1(self.conv1(x))))
        x = torch.relu(self.pool(self.bn2(self.conv2(x))))
        x = torch.relu(self.pool(self.bn3(self.conv3(x))))
        x = torch.relu(self.pool(self.bn4(self.conv4(x))))
        x = torch.relu(self.pool(self.bn5(self.conv5(x))))
        x = self.flatten(x)
        return x
    
class DeepCNN(torch.nn.Module):
    
    def __init__(self, num_channel, dropout=0) -> None:
        super().__init__()

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        
        self.conv1 = nn.Conv2d(in_channels=num_channel, out_channels=num_channel*2, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(num_channel*2)
        self.conv2 = nn.Conv2d(in_channels=num_channel*2, out_channels=num_channel*4, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(num_channel*4)
        self.conv3 = nn.Conv2d(in_channels=num_channel*4, out_channels=num_channel*8, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(num_channel*8)
        self.conv4 = nn.Conv2d(in_channels=num_channel*8, out_channels=num_channel*16, kernel_size=3)
        self.bn4= nn.BatchNorm2d(num_channel*16)
        self.conv5 = nn.Conv2d(in_channels=num_channel*16, out_channels=num_channel*32, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(num_channel*32)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_size = num_channel*32
        
    def forward(self, x):
        x = self.dropout(x) if self.dropout else x
        x = torch.relu(self.pool(self.bn1(self.conv1(x))))
        x = self.dropout(x) if self.dropout else x
        x = torch.relu(self.pool(self.bn2(self.conv2(x))))
        x = self.dropout(x) if self.dropout else x
        x = torch.relu(self.pool(self.bn3(self.conv3(x))))
        x = self.dropout(x) if self.dropout else x
        x = torch.relu(self.pool(self.bn4(self.conv4(x))))
        x = self.dropout(x) if self.dropout else x
        x = torch.relu(self.pool(self.bn5(self.conv5(x))))
        x = self.avgpool(x)
        x = self.flatten(x)
        return x    
    
class BasicBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0, stride=1, downsample=None):
        super(BasicBlocks, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        
    def forward(self, x):
        x = self.dropout(x) if self.dropout else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn1(self.conv2(out))
        
        if self.downsample is not None:
            x = self.downsample(x)
        out = out + x
        out = self.relu(out)
        return out
    
class ResNet(torch.nn.Module):
    
    def __init__(self, num_channel, dropout=0, layers=[32, 64, 128, 256]) -> None:
        super().__init__()    
        self.conv1 = nn.Conv2d(in_channels=num_channel, out_channels=layers[0], kernel_size=7)
        self.bn1 = nn.BatchNorm2d(num_features=layers[0])
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        
        self.res_block_1 = BasicBlocks(in_channels=layers[0], out_channels=layers[0], dropout=dropout)
        self.res_block_2 = BasicBlocks(in_channels=layers[0], out_channels=layers[1], stride=2, dropout=dropout,
                                       downsample=nn.Sequential(nn.Conv2d(in_channels=layers[0], out_channels=layers[1], stride=2, kernel_size=1, bias=False),
                                                                nn.BatchNorm2d(num_features=layers[1])))
        self.res_block_3 = BasicBlocks(in_channels=layers[1], out_channels=layers[2], stride=2, dropout=dropout,
                                       downsample=nn.Sequential(nn.Conv2d(in_channels=layers[1], out_channels=layers[2], stride=2, kernel_size=1, bias=False),
                                                                nn.BatchNorm2d(num_features=layers[2])))
        self.res_block_4 = BasicBlocks(in_channels=layers[2], out_channels=layers[3], stride=2, dropout=dropout,
                                       downsample=nn.Sequential(nn.Conv2d(in_channels=layers[2], out_channels=layers[3], stride=2, kernel_size=1, bias=False),
                                                                nn.BatchNorm2d(num_features=layers[3])))


        self.flatten = nn.Flatten()
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_size = layers[-1]

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        
    def forward(self, x):
        x = self.dropout(x) if self.dropout else x
        x = self.relu(self.pool(self.bn1(self.conv1(x))))
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_3(x)
        x = self.res_block_4(x)
        x = self.avgPool(x)
        x = self.flatten(x)
        return x   
    
class RNN(torch.nn.Module):
    
    def __init__(self, T, input_size, device, dropout=0, num_layers=1, hidden_size=None) -> None:
        super().__init__()
        
        self.T = T
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size else input_size
        self.num_layers = num_layers
        self.device = device
        
        self.rnn = nn.LSTM(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                        #   nonlinearity='relu',
                          dropout=dropout,
                          batch_first=True)
        # self.transformer = nn.Transformer
    
    def forward(self, x):
        # X:    (batch, T, rnn_input_size)
        batch_size = x.shape[0]
        
        # RNN
        h0 = (torch.randn(1, batch_size, self.hidden_size, dtype=torch.float32).to(self.device),
              torch.randn(1, batch_size, self.hidden_size, dtype=torch.float32).to(self.device))
        # x: (batch, T, hidden_size)
        x, _ = self.rnn(x, h0)
        
        # Get the last one
        # x: (batch, hidden_size)
        x = x[:, -1, :]
        
        return x
    
class BasicModel(torch.nn.Module):
    
    def __init__(self, num_weather_channel, device, T_sat=12, T_wea=6) -> None:
        super().__init__()
        
        self.num_nonhrv_channel = 11
        self.num_weather_channel = num_weather_channel
        
        self.T_sat = T_sat
        self.T_wea = T_wea
        
        self.sat_cnn = CNN(self.num_nonhrv_channel + 1)
        self.wea_cnn = CNN(self.num_weather_channel)
        
        # self.sat_pv_rnn = RNN(self.T_sat, input_size=self.sat_cnn.output_size+1, hidden_size=256, device=device)
        # self.wea_rnn = RNN(self.T_wea, input_size=self.wea_cnn.output_size, hidden_size=256, device=device)
        
        output_size = (self.sat_cnn.output_size+1) * T_sat + self.wea_cnn.output_size * T_wea

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(output_size, int(output_size / 2))
        self.linear2 = nn.Linear(int(output_size / 2), 48)

    def forward(self, pv, hrv, nonhrv, weather):
        # pv:       (batch, 12)
        batch_size = pv.shape[0]
        
        # 5-minute
        # hrv:      (batch, 12, 128, 128)
        # non-hrv:  (batch, 12, 128, 128, 11)
        # sat_img:  (batch, 12, 11 + 1, 128, 128)
        # sat_img = torch.cat((hrv, nonhrv), dim=2)
        sat_img = torch.cat((hrv.unsqueeze(2), nonhrv.permute(0, 1, 4, 2, 3)), dim=2)
        # sat_img:  (batch * 12, 11 + 1, 128, 128)
        sat_img = sat_img.reshape(batch_size * self.T_sat, self.num_nonhrv_channel + 1, *sat_img.shape[-2:])
        # sat_fea:  (batch * 12, sat_cnn.output_size)
        sat_fea = self.sat_cnn(sat_img)
        # sat_fea_pv: (batch * 12, sat_cnn.output_size + 1)
        sat_fea_pv = torch.cat((sat_fea.reshape(batch_size, self.T_sat, sat_fea.shape[-1]), pv.unsqueeze(2)), dim=-1)

        # Hourly
        # weather:  (batch, N_weather, 6, 128, 128)
        # wea_img:  (batch, 6, N_weather, 128, 128)
        wea_img = weather.transpose(1, 2)
        # wea_fea:  (batch * 6, N_weather, 128, 128)
        wea_img = wea_img.reshape(batch_size * self.T_wea, self.num_weather_channel, *weather.shape[-2:])
        # wea_fea:  (batch * 6, wea_cnn.output_size)
        wea_fea = self.wea_cnn(wea_img)
        # wea_fea:  (batch, 6, wea_cnn.output_size)
        wea_fea = wea_fea.reshape(batch_size, self.T_wea, wea_fea.shape[-1])
        
        # print(sat_fea_pv.shape, wea_fea.shape)
        
        # o_sat_pv = self.sat_pv_rnn(sat_fea_pv)
        # o_wea = self.wea_rnn(wea_fea)

        # x = torch.concat((o_sat_pv, o_wea), dim=-1)
        x = torch.concat((self.flatten(sat_fea_pv), self.flatten(wea_fea)), dim=-1)
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))

        return x

class RecurrentModel(torch.nn.Module):
    
    def __init__(self, num_weather_channel, device, dropout=0, T_sat=12, T_wea=6) -> None:
        super().__init__()
        
        self.num_nonhrv_channel = 11
        self.num_weather_channel = num_weather_channel
        
        self.T_sat = T_sat
        self.T_wea = T_wea
        
        self.sat_cnn = DeepCNN(self.num_nonhrv_channel + 1, dropout=dropout)
        self.wea_cnn = DeepCNN(self.num_weather_channel, dropout=dropout)
        
        self.sat_pv_rnn = RNN(self.T_sat, dropout=dropout, input_size=self.sat_cnn.output_size+1, hidden_size=256, device=device)
        self.wea_rnn = RNN(self.T_wea, dropout=dropout, input_size=self.wea_cnn.output_size, hidden_size=256, device=device)
        
        output_size = self.sat_pv_rnn.hidden_size + self.wea_rnn.hidden_size

        # self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(output_size, int(output_size / 2))
        self.linear2 = nn.Linear(int(output_size / 2), 48)

    def forward(self, pv, hrv, nonhrv, weather):
        # pv:       (batch, 12)
        batch_size = pv.shape[0]
        
        # 5-minute
        # hrv:      (batch, 12, 128, 128)
        # non-hrv:  (batch, 12, 128, 128, 11)
        # sat_img:  (batch, 12, 11 + 1, 128, 128)
        # sat_img = torch.cat((hrv, nonhrv), dim=2)
        sat_img = torch.cat((hrv.unsqueeze(2), nonhrv.permute(0, 1, 4, 2, 3)), dim=2)
        # sat_img:  (batch * 12, 11 + 1, 128, 128)
        sat_img = sat_img.reshape(batch_size * self.T_sat, self.num_nonhrv_channel + 1, *sat_img.shape[-2:])
        # sat_fea:  (batch * 12, sat_cnn.output_size)
        sat_fea = self.sat_cnn(sat_img)
        # sat_fea_pv: (batch * 12, sat_cnn.output_size + 1)
        sat_fea_pv = torch.cat((sat_fea.reshape(batch_size, self.T_sat, sat_fea.shape[-1]), pv.unsqueeze(2)), dim=-1)

        # Hourly
        # weather:  (batch, N_weather, 6, 128, 128)
        # wea_img:  (batch, 6, N_weather, 128, 128)
        wea_img = weather.transpose(1, 2)
        # wea_fea:  (batch * 6, N_weather, 128, 128)
        wea_img = wea_img.reshape(batch_size * self.T_wea, self.num_weather_channel, *weather.shape[-2:])
        # wea_fea:  (batch * 6, wea_cnn.output_size)
        wea_fea = self.wea_cnn(wea_img)
        # wea_fea:  (batch, 6, wea_cnn.output_size)
        wea_fea = wea_fea.reshape(batch_size, self.T_wea, wea_fea.shape[-1])
        
        o_sat_pv = self.sat_pv_rnn(sat_fea_pv)
        o_wea = self.wea_rnn(wea_fea)

        x = torch.concat((o_sat_pv, o_wea), dim=-1)
        # x = torch.concat((self.flatten(sat_fea_pv), self.flatten(wea_fea)), dim=-1)
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))

        return x
    
    
class PureWeatherModel(torch.nn.Module):
    
    def __init__(self, num_weather_channel, device, dropout=0, T_sat=12, T_wea=6) -> None:
        super().__init__()
        
        self.num_nonhrv_channel = 11
        self.num_weather_channel = num_weather_channel
        
        self.T_sat = T_sat
        self.T_wea = T_wea
        
        self.sat_cnn = DeepCNN(self.num_nonhrv_channel + 1, dropout=dropout)
        # self.wea_cnn = DeepCNN(self.num_weather_channel, dropout=dropout)
        
        self.sat_pv_rnn = RNN(self.T_sat, dropout=dropout, input_size=self.sat_cnn.output_size+1, hidden_size=256, device=device)
        # self.pv_rnn = RNN(self.T_sat, dropout=dropout, input_size=1, hidden_size=128, device=device)
        self.wea_rnn = RNN(self.T_wea, dropout=dropout, input_size=self.num_weather_channel, hidden_size=256, device=device)
        
        # output_size = self.wea_rnn.hidden_size + self.pv_rnn.hidden_size
        output_size = self.sat_pv_rnn.hidden_size + self.wea_rnn.hidden_size

        # self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(output_size, int(output_size / 2))
        self.linear2 = nn.Linear(int(output_size / 2), 48)

    def forward(self, pv, hrv, nonhrv, weather):
        # pv:       (batch, 12)
        batch_size = pv.shape[0]
        
        # 5-minute
        # hrv:      (batch, 12, 128, 128)
        # non-hrv:  (batch, 12, 128, 128, 11)
        # sat_img:  (batch, 12, 11 + 1, 128, 128)
        # sat_img = torch.cat((hrv, nonhrv), dim=2)
        sat_img = torch.cat((hrv.unsqueeze(2), nonhrv.permute(0, 1, 4, 2, 3)), dim=2)
        
        # sat_img:  (batch * 12, 11 + 1, 128, 128)
        sat_img = sat_img.reshape(batch_size * self.T_sat, self.num_nonhrv_channel + 1, *sat_img.shape[-2:])
        
        # sat_fea:  (batch * 12, sat_cnn.output_size)
        sat_fea = self.sat_cnn(sat_img)
        
        # sat_fea_pv: (batch * 12, sat_cnn.output_size + 1)
        sat_fea_pv = torch.cat((sat_fea.reshape(batch_size, self.T_sat, sat_fea.shape[-1]), pv.unsqueeze(2)), dim=-1)

        # Hourly
        # weather:  (batch, N_weather, 6)
        # wea_img:  (batch, 6, N_weather, 128, 128)
        # wea_img = weather.transpose(1, 2)
        
        # wea_fea:  (batch * 6, N_weather, 128, 128)
        # wea_img = wea_img.reshape(batch_size * self.T_wea, self.num_weather_channel, *weather.shape[-2:])
        # wea_fea:  (batch * 6, wea_cnn.output_size)
        # wea_fea = self.wea_cnn(wea_img)
        # wea_fea:  (batch, 6, wea_cnn.output_size)
        # wea_fea = wea_fea.reshape(batch_size, self.T_wea, wea_fea.shape[-1])
        
        o_sat_pv = self.sat_pv_rnn(sat_fea_pv)
        # o_pv = self.pv_rnn(pv.unsqueeze(-1))
        o_wea = self.wea_rnn(weather.transpose(1, 2))

        x = torch.concat((o_sat_pv, o_wea), dim=-1)
        # x = torch.concat((self.flatten(sat_fea_pv), self.flatten(wea_fea)), dim=-1)
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))

        return x