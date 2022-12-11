
import torch
import torch.nn as nn
import numpy as np


class ExoMinerRaw(nn.Module):
    def __init__(self):
        super(ExoMinerRaw, self).__init__()
        def DetermineMaxPoolOutputSize(input_size , stride,ceil_mode=False):
            """
            ceil_mode = False in line with pytorch documentation for maxpooling layers
            https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html
            """
            if(ceil_mode==False):
                return int(np.floor(((input_size - 1) / stride) +1))
            else:
                return int(np.ceil(((input_size - 1) / stride) + 1))

        raw_dimension = 6000


        #### Feature 1: Full Orbit Flux [1x301]
        self.f1_LSTM = nn.LSTM(input_size=raw_dimension, hidden_size=301,batch_first=True)
        self.f1_conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=6, stride=1, padding=0)
        self.f1_maxp1 =nn.MaxPool1d(kernel_size=8,stride=1)
        f1out_size_maxp1 = DetermineMaxPoolOutputSize(16 , 1,ceil_mode=False)
        self.f1_conv2 = nn.Conv1d(in_channels=f1out_size_maxp1, out_channels=32, kernel_size=6, stride=1, padding=0)
        self.f1_maxp2 = nn.MaxPool1d(kernel_size=8,stride=1)
        f1out_size_maxp2 = DetermineMaxPoolOutputSize(input_size=32, stride=1, ceil_mode=False)
        self.f1_conv3 = nn.Conv1d(in_channels=f1out_size_maxp2, out_channels=64, kernel_size=6, stride=1, padding=0)
        self.f1_maxp3 = nn.MaxPool1d(kernel_size=8,stride=1)
        f1out_size_maxp3 = DetermineMaxPoolOutputSize(input_size=64, stride=1, ceil_mode=False)
        self.f1_lin1 = nn.Linear(in_features=f1out_size_maxp3*265,out_features=4)

        self.F1_conv_layers = nn.Sequential(
            self.f1_conv1,
            self.f1_maxp1,
            self.f1_conv2,
            self.f1_maxp2,
            self.f1_conv3,
            self.f1_maxp3
        )

        #### Feature 2: Transit Veiw Flux [1x31]
        self.f2_LSTM = nn.LSTM(input_size=raw_dimension, hidden_size=31,batch_first=True)
        self.f2_conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=6, stride=1, padding=0)
        self.f2_maxp1 = nn.MaxPool1d(kernel_size=6, stride=1)
        f2out_size_maxp1 = DetermineMaxPoolOutputSize(16, 1, ceil_mode=False)
        self.f2_conv2 = nn.Conv1d(in_channels=f2out_size_maxp1, out_channels=32, kernel_size=6, stride=1, padding=0)
        self.f2_maxp2 = nn.MaxPool1d(kernel_size=6, stride=1) # add [1x1] transit depth to output of this layer
        f2out_size_maxp2 = DetermineMaxPoolOutputSize(32, 1, ceil_mode=False)
        self.f2_lin1 = nn.Linear(in_features= (f2out_size_maxp2*11)+1, out_features=4) # 1 for transit depth


        self.F2_part1 = nn.Sequential(
            self.f2_conv1,
            self.f2_maxp1,
            self.f2_conv2,
            self.f2_maxp2
        )
        self.F2_part2 = self.f2_lin1

        #### Feature 3: Full Orbit Centroid [1x301]
        self.f3_LSTM = nn.LSTM(input_size=raw_dimension+1, hidden_size=301,batch_first=True)
        self.f3_conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=6, stride=1, padding=0)
        self.f3_maxp1 = nn.MaxPool1d(kernel_size=8, stride=1)
        f3out_size_maxp1 = DetermineMaxPoolOutputSize(16, 1, ceil_mode=False)
        self.f3_conv2 = nn.Conv1d(in_channels=f3out_size_maxp1, out_channels=32, kernel_size=6, stride=1, padding=0)
        self.f3_maxp2 = nn.MaxPool1d(kernel_size=8, stride=1)
        f3out_size_maxp2 = DetermineMaxPoolOutputSize(input_size=32, stride=1, ceil_mode=False)
        self.f3_conv3 = nn.Conv1d(in_channels=f3out_size_maxp2, out_channels=64, kernel_size=6, stride=1, padding=0)
        self.f3_maxp3 = nn.MaxPool1d(kernel_size=8, stride=1)
        f3out_size_maxp3 = DetermineMaxPoolOutputSize(input_size=64, stride=1, ceil_mode=False)
        self.f3_lin1 = nn.Linear(in_features=f3out_size_maxp3*265, out_features=4)

        self.F3_conv_layers = nn.Sequential(
            self.f3_conv1,
            self.f3_maxp1,
            self.f3_conv2,
            self.f3_maxp2,
            self.f3_conv3,
            self.f3_maxp3
        )

        #### Feature 4: Transit Veiw Centroid [1x31]
        self.f4_LSTM = nn.LSTM(input_size=raw_dimension+1, hidden_size=31,batch_first=True)
        self.f4_conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=6, stride=1, padding=0)
        self.f4_maxp1 = nn.MaxPool1d(kernel_size=6, stride=1)
        f4out_size_maxp1 = DetermineMaxPoolOutputSize(16, 1, ceil_mode=False)
        self.f4_conv2 = nn.Conv1d(in_channels=f4out_size_maxp1, out_channels=32, kernel_size=6, stride=1, padding=0)
        self.f4_maxp2 = nn.MaxPool1d(kernel_size=6, stride=1)  # add [1x5] Centroid Scalars to output of this layer
        f4out_size_maxp2 = DetermineMaxPoolOutputSize(32, 1, ceil_mode=False)
        self.f4_lin1 = nn.Linear(in_features=(f4out_size_maxp2*11) + 5, out_features=4)  # +5 for Centroid Scalars

        self.F4_part1 = nn.Sequential(
            self.f4_conv1,
            self.f4_maxp1,
            self.f4_conv2,
            self.f4_maxp2
        )
        self.F4_part2 = self.f4_lin1

        #### Feature 5: Transit Veiw Odd Even 2x[1x31]
        self.f5_LSTM = nn.LSTM(input_size=raw_dimension, hidden_size=31,batch_first=True)
        self.f5_conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=6, stride=1, padding=0)
        self.f5_maxp1 = nn.MaxPool1d(kernel_size=6, stride=1)
        f5out_size_maxp1 = DetermineMaxPoolOutputSize(16, 1, ceil_mode=False)
        self.f5_conv2 = nn.Conv1d(in_channels=f5out_size_maxp1, out_channels=32, kernel_size=6, stride=1, padding=0)
        self.f5_maxp2 = nn.MaxPool1d(kernel_size=6, stride=1)  # subtract odd and even output from each other in this layer before lin layer
        f5out_size_maxp2 = DetermineMaxPoolOutputSize(32, 1, ceil_mode=False)
        self.f5_lin1 = nn.Linear(in_features=(f5out_size_maxp2*11) , out_features=4)

        self.F5_part1 = nn.Sequential(
            self.f5_conv1,
            self.f5_maxp1,
            self.f5_conv2,
            self.f5_maxp2
        )
        self.F5_part2 = self.f5_lin1


        #### Feature 6: Transit Veiw Secondary Eclipse [1x31]
        self.f6_LSTM = nn.LSTM(input_size=raw_dimension, hidden_size=31,batch_first=True)
        self.f6_conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=6, stride=1, padding=0)
        self.f6_maxp1 = nn.MaxPool1d(kernel_size=6, stride=1)
        f6out_size_maxp1 = DetermineMaxPoolOutputSize(16, 1, ceil_mode=False)
        self.f6_conv2 = nn.Conv1d(in_channels=f6out_size_maxp1, out_channels=32, kernel_size=6, stride=1, padding=0)
        self.f6_maxp2 = nn.MaxPool1d(kernel_size=6,
                                     stride=1)  #Append in secondary scalars here [1x4]
        f6out_size_maxp2 = DetermineMaxPoolOutputSize(32, 1, ceil_mode=False)
        self.f6_lin1 = nn.Linear(in_features=(f6out_size_maxp2*11)+4, out_features=4)

        self.F6_part1 = nn.Sequential(
            self.f6_conv1,
            self.f6_maxp1,
            self.f6_conv2,
            self.f6_maxp2
        )
        self.F6_part2 = self.f6_lin1

        #### Feature 7:
        #append on stellar params [1x6]

        #### Feature 8:
        # append on DV Diagnostic Tests [1x6]

        #Final Segment:
        #input = 6*4 +12
        self.FinalLin1= nn.Linear(in_features=(6*4) +12 , out_features=128)
        self.Dropout1 = nn.Dropout(p=.3)
        self.FinalLin2 = nn.Linear(in_features=128, out_features=128)
        self.Dropout2 = nn.Dropout(p=.3)
        self.FinalLin3 = nn.Linear(in_features=128, out_features=128)
        self.Dropout3 = nn.Dropout(p=.3)
        self.FinalLin4 = nn.Linear(in_features=128, out_features=1)
        self.Dropout4 = nn.Dropout(p=.3)
        self.Sigmoid = nn.Sigmoid()

        self.Final = nn.Sequential(
            self.FinalLin1,
            self.Dropout1,
            self.FinalLin2,
            self.Dropout2,
            self.FinalLin3,
            self.Dropout3,
            self.FinalLin4,
            self.Dropout4,
            self.Sigmoid
        )
    def forward(self, flux, centroid, f2_2,f4_2,f6_2,f7,f8): #f1,f2,f2_2,f3,f4,f4_2,f5_odd,f5_even,f6,f6_2,f7,f8):
        out = None

        # print('\nflux: ', flux.shape)
        # print('\ncentroid: ', centroid.shape)

        #Feature 1
        output , f1_out = self.f1_LSTM(flux)
        f1_out = f1_out[0]
        f1_out = f1_out.reshape((f1_out.shape[1] , f1_out.shape[0],f1_out.shape[2]))
        f1_out = self.F1_conv_layers(f1_out)
        f1_out = f1_out.reshape(f1_out.shape[0], f1_out.shape[1]*f1_out.shape[2])
        f1_out = self.f1_lin1(f1_out)
        # Feature 2
        output , f2_out = self.f2_LSTM(flux)
        f2_out = f2_out[0]
        f2_out = f2_out.reshape((f2_out.shape[1], f2_out.shape[0], f2_out.shape[2]))
        f2_part1 = self.F2_part1(f2_out)
        f2_total = f2_part1.reshape(f2_part1.shape[0],f2_part1.shape[1]*f2_part1.shape[2])
        f2_total = torch.cat((f2_total,f2_2),dim=1)
        f2_out = self.F2_part2(f2_total)
        # Feature 3

        output, f3_out = self.f3_LSTM(centroid)
        f3_out = f3_out[0]
        f3_out = f3_out.reshape((f3_out.shape[1], f3_out.shape[0], f3_out.shape[2]))
        f3_out = self.F3_conv_layers(f3_out)
        f3_out = f3_out.reshape(f3_out.shape[0], f3_out.shape[1] * f3_out.shape[2])
        f3_out = self.f3_lin1(f3_out)

        # Feature 4

        output, f4_out = self.f4_LSTM(centroid)
        f4_out = f4_out[0]
        f4_out = f4_out.reshape((f4_out.shape[1], f4_out.shape[0], f4_out.shape[2]))
        f4_part1 = self.F4_part1(f4_out)
        f4_total = f4_part1.reshape(f4_out.shape[0],f4_part1.shape[1]*f4_part1.shape[2])
        f4_total = torch.cat((f4_total, f4_2), dim=1)
        f4_out   = self.F4_part2(f4_total)
        # Feature 5
        output , f5_out = self.f5_LSTM(flux)
        f5_out = f5_out[0]
        f5_out = f5_out.reshape((f5_out.shape[1], f5_out.shape[0], f5_out.shape[2]))
        f5_part1  = self.F5_part1(f5_out)
        # f5_even_out = self.F5_part1(f5_even)
        # f5_total    = torch.sub(f5_odd_out ,f5_even_out)
        f5_total    = f5_part1.reshape(f5_out.shape[0], f5_part1.shape[1] * f5_part1.shape[2])
        f5_out      = self.F5_part2(f5_total)

        #Feature 6
        output , f6_out = self.f6_LSTM(flux)
        f6_out = f6_out[0]
        f6_out = f6_out.reshape((f6_out.shape[1], f6_out.shape[0], f6_out.shape[2]))
        f6_part1 = self.F6_part1(f6_out)
        f6_total = f6_part1.reshape(f6_out.shape[0], f6_part1.shape[1]*f6_part1.shape[2])
        f6_total = torch.cat((f6_total, f6_2),dim=1)
        f6_out = self.F6_part2(f6_total)

        #Feature 7
        f7_out = f7

        #Feature 8
        f8_out =f8
        All_output = [f1_out,f2_out,f3_out,f4_out,f5_out,f6_out,f7_out,f8_out]
        input_into_final_layer = torch.cat((All_output) , dim=1)
        #Final
        out = self.Final(input_into_final_layer)
        return out