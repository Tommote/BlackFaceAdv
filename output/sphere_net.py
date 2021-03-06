import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as P


class sphere20a(nn.Cell):
    def __init__(self,classnum=10574,feature=False):
        super(sphere20a, self).__init__()
        self.classnum = classnum
        self.feature = feature
        #input = B*3*112*96
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, pad_mode='pad', padding=1, has_bias=True) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(channel=64)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True)
        self.relu1_2 = nn.PReLU(channel=64)
        self.conv1_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True)
        self.relu1_3 = nn.PReLU(channel=64)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, pad_mode='pad', padding=1, has_bias=True) #=>B*128*28*24
        self.relu2_1 = nn.PReLU(channel=128)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True)
        self.relu2_2 = nn.PReLU(channel=128)
        self.conv2_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True)
        self.relu2_3 = nn.PReLU(channel=128)

        self.conv2_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True) #=>B*128*28*24
        self.relu2_4 = nn.PReLU(channel=128)
        self.conv2_5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True)
        self.relu2_5 = nn.PReLU(channel=128)


        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, pad_mode='pad', padding=1, has_bias=True) #=>B*256*14*12
        self.relu3_1 = nn.PReLU(channel=256)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True)
        self.relu3_2 = nn.PReLU(channel=256)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True)
        self.relu3_3 = nn.PReLU(channel=256)

        self.conv3_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True) #=>B*256*14*12
        self.relu3_4 = nn.PReLU(channel=256)
        self.conv3_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True)
        self.relu3_5 = nn.PReLU(channel=256)

        self.conv3_6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True) #=>B*256*14*12
        self.relu3_6 = nn.PReLU(channel=256)
        self.conv3_7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True)
        self.relu3_7 = nn.PReLU(channel=256)

        self.conv3_8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True) #=>B*256*14*12
        self.relu3_8 = nn.PReLU(channel=256)
        self.conv3_9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True)
        self.relu3_9 = nn.PReLU(channel=256)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, pad_mode='pad', padding=1, has_bias=True) #=>B*512*7*6
        self.relu4_1 = nn.PReLU(channel=512)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True)
        self.relu4_2 = nn.PReLU(channel=512)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True)
        self.relu4_3 = nn.PReLU(channel=512)

        self.fc5 = nn.Dense(in_channels=512 * 7 * 6, out_channels=512)
        # self.fc6 = AngleLinear(512,self.classnum)


    def construct(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

        x = P.Reshape()(x, (P.Shape()(x)[0],-1,))
        x = self.fc5(x)
        # print(x.shape)
        if self.feature: 
            return x

        # x = self.fc6(x)
        # return x
