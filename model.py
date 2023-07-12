import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

class CNNEncoder(nn.Module):
    def __init__(self, cnn_out_dim=256, drop_prob=0.3, bn_momentum=0.01):
        '''
        使用pytorch提供的预训练模型作为encoder
        Use the pre-trained model provided by pytorch as an encoder
        '''
        super(CNNEncoder, self).__init__()

        self.cnn_out_dim = cnn_out_dim
        self.drop_prob = drop_prob
        self.bn_momentum = bn_momentum

        # 使用resnet预训练模型来提取特征，去掉最后一层分类器
        # Use the resnet pre-training model to extract features and remove the last layer of classifiers
        pretrained_cnn = models.resnet34(pretrained=True)
        cnn_layers = list(pretrained_cnn.children())[:-1]

        # 把resnet的最后一层fc层去掉，用来提取特征
        # Remove the last fc layer of resnet to extract features

        self.cnn = nn.Sequential(*cnn_layers)
        # 将特征embed成cnn_out_dim维向量
        # Embed the features into a cnn_out_dim-dimensional vector
        self.fc = nn.Sequential(
            *[
                self._build_fc(pretrained_cnn.fc.in_features, 512, True),
                nn.ReLU(),
                self._build_fc(512, 512, True),
                nn.ReLU(),
                nn.Dropout(p=self.drop_prob),
                self._build_fc(512, self.cnn_out_dim, False)
            ]
        )

    def _build_fc(self, in_features, out_features, with_bn=True):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features, momentum=self.bn_momentum)
        ) if with_bn else nn.Linear(in_features, out_features)

    def forward(self, x_3d):
        '''
        输入的是T帧图像, shape = (batch_size, t, h, w, 3)
        The input is T frame images, shape = (batch_size, t, h, w, 3)
        '''
        cnn_embedding_out = []
        for t in range(x_3d.size(1)):
            # 使用cnn提取特征
            # Use cnn to extract features
            # 为什么要用到no_grad()？
            # Why use no_grad()?
            # -- 因为我们使用的预训练模型，防止后续的层训练时反向传播而影响前面的层
            # -- Because the pre-training model we use prevents the backpropagation of subsequent layers from affecting the previous layers during training
            with torch.no_grad():
                x = self.cnn(x_3d[:, t, :, :, :])
                x = torch.flatten(x, start_dim=1)

            # 处理fc层
            # Process fc layer
            x = self.fc(x)

            cnn_embedding_out.append(x)

        cnn_embedding_out = torch.stack(cnn_embedding_out, dim=0).transpose(0, 1)

        return cnn_embedding_out

class RNNDecoder(nn.Module):
    def __init__(self, use_gru=True, cnn_out_dim=256, rnn_hidden_layers=3, rnn_hidden_nodes=256,
            num_classes=10, drop_prob=0.3):
        super(RNNDecoder, self).__init__()

        self.rnn_input_features = cnn_out_dim
        self.rnn_hidden_layers = rnn_hidden_layers
        self.rnn_hidden_nodes = rnn_hidden_nodes

        self.drop_prob = drop_prob
        self.num_classes = num_classes # 这里调整分类数目 # Adjust the number of categories here

        # rnn配置参数 # rnn configuration parameters
        rnn_params = {
            'input_size': self.rnn_input_features,
            'hidden_size': self.rnn_hidden_nodes,
            'num_layers': self.rnn_hidden_layers,
            'batch_first': True
        }

        # 使用lstm或者gru作为rnn层 # Use lstm or gru as rnn layer
        self.rnn = (nn.GRU if use_gru else nn.LSTM)(**rnn_params)

        # rnn层输出到线性分类器 # Rnn layer output to linear classifier
        self.fc = nn.Sequential(
            nn.Linear(self.rnn_hidden_nodes, 128),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x_rnn):
        self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(x_rnn, None)
        # 注意，前面定义rnn模块时，batch_first=True保证了以下结构：
        # Note that when defining the rnn module earlier, batch_first=True guarantees the following structure:
        # rnn_out shape: (batch, timestep, output_size)
        # h_n and h_c shape: (n_layers, batch, hidden_size)

        x = self.fc(rnn_out[:, -1, :]) # 只抽取最后一层做输出 # Only extract the last layer for output

        return x
