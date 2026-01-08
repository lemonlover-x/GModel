import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import grad
import torch.nn.functional as F
import math
from dataread import read_data


class sample_data(Dataset):
    def __init__(self, energy, tem):
        self.energy = energy
        self.tem = tem

    def __len__(self):
        return len(self.energy)

    def __getitem__(self, idx):
        real_energy_data = self.energy[idx]
        real_tem_data = self.tem[idx]
        return real_energy_data, real_tem_data


# 多尺度周期注意力定义
class MultiScalePeriodicAttention(nn.Module):
    def __init__(self, embed_dim, device, num_scales=[24, 720]):
        super(MultiScalePeriodicAttention, self).__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.num_scales = num_scales
        self.sqrt_d = math.sqrt(embed_dim)
        # 可学习权重 beta_n
        self.beta = nn.Parameter(torch.ones(len(num_scales))).to(self.device)
        # 权重系数 omega_n
        self.omega = nn.Parameter(torch.ones(len(num_scales))).to(self.device)
        # 线性变换层
        self.W_q = nn.Linear(embed_dim, embed_dim).to(self.device)
        self.W_k = nn.Linear(embed_dim, embed_dim).to(self.device)
        self.W_v = nn.Linear(embed_dim, embed_dim).to(self.device)

    def compute_periodic_bias(self, seq_len, T):
        """ 计算周期偏置矩阵 P[i, j] """
        i = torch.arange(seq_len).view(-1, 1).to(self.device)
        j = torch.arange(seq_len).view(1, -1).to(self.device)
        P = torch.cos((2 * math.pi * torch.abs(i - j)) / T)
        return P.to(dtype=torch.float32, device=self.device)

    def forward(self, x):
        """ 计算多尺度周期注意力 """
        x = x.to(self.device)
        seq_len = x.shape[1]
        # 计算 Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.sqrt_d
        # 计算不同尺度的周期偏置矩阵，并加权
        periodic_bias = torch.zeros_like(attn).to(self.device)
        for i, T in enumerate(self.num_scales):
            P_n = self.compute_periodic_bias(seq_len, T)
            periodic_bias += self.beta[i] * P_n
        # 计算注意力分数
        attn_scores = F.softmax(attn + periodic_bias, dim=-1)
        # 计算注意力输出
        output = torch.matmul(attn_scores, V).to(self.device)
        return output


# 时间序列位置编码
class SequenceEncoder(nn.Module):
    def __init__(self, hidden_dim, device):
        super(SequenceEncoder, self).__init__()
        # 输入维度到隐空间的线性映射
        self.hidden_dim = hidden_dim
        self.device = device
        self.linear = nn.Linear(2, hidden_dim).to(device)

    # 定义位置编码方式
    def create_positional_encoding(self, seq_len, hidden_dim):
        # 生成位置编码
        position = torch.arange(0, seq_len).unsqueeze(1).float().to(self.device)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * -(math.log(10000.0) / hidden_dim))
        div_term = div_term.to(self.device)
        positional_encoding = torch.zeros(seq_len, hidden_dim).float().to(self.device)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        return positional_encoding.unsqueeze(0)  # 扩展为 [1, seq_len, hidden_dim]

    def forward(self, z, c):
        """
       z: [batch_size, seq_len, noise_dim]  (噪声)
       c: [batch_size, seq_len, cond_dim]   (条件信息)
       """
        noise_dim = z.shape[2]
        # print(' zbatch_size:', z.shape)
        cond_dim = c.shape[2]
        # print(' cbatch_size:',c.shape)
        seq_len = z.shape[1]
        # print(' seq_len:', seq_len)
        input_z_c = torch.cat([z, c], dim=-1).to(self.device)
        # linear = nn.Linear(noise_dim + cond_dim, self.hidden_dim).to(self.device)
        encoded_input = self.linear(input_z_c)
        # 将位置编码添加到噪声和条件序列上
        positional_encoding = self.create_positional_encoding(seq_len, self.hidden_dim)
        encoded_input += positional_encoding.to(encoded_input.device)
        return encoded_input


# Transformer子模块
class Transformer_block(nn.Module):
    def __init__(self, hidden_dim, device, dropout=0.1):
        super(Transformer_block, self).__init__()
        # 周期注意力
        self.device = device
        self.attention = MultiScalePeriodicAttention(hidden_dim, device, num_scales=[24, 720])
        # LayerNorm
        self.layer_norm_1 = nn.LayerNorm(hidden_dim).to(self.device)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim).to(self.device)
        # Feed Forward Network (FFN)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        ).to(device)
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 注意力计算
        x = x.to(self.device)
        x_norm = self.layer_norm_1(x)  # norm
        attention = self.attention(x_norm)
        attention_out = self.dropout(attention)  # dropout
        x = self.layer_norm_1(x + attention_out)  # Add & Norm
        # FNN
        x_norm = self.layer_norm_2(x)
        feedforward = self.feed_forward(x_norm)
        feedforward_out = self.dropout(feedforward)  # dropout
        x = x + feedforward_out  # Add & Norm
        return x


# Transformer生成器
class TransformerGenerator(nn.Module):
    def __init__(self, hidden_dim, num_layers, output_dim, device):
        super(TransformerGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        # 定义Transformer编码器层
        # self.transformer_block=Transformer_block(hidden_dim,device,dropout=0.1)
        self.transformer_layers = nn.ModuleList([
            Transformer_block(hidden_dim, device, dropout=0.1) for _ in range(num_layers)
        ])
        # 输出层
        self.out = nn.Linear(hidden_dim, output_dim).to(device)
        self.Gseq = SequenceEncoder(hidden_dim, device)

    def forward(self, z, c):
        z = z.to(self.device)
        c = c.to(self.device)
        sequence = self.Gseq(z, c)
        for block in self.transformer_layers:
            sequence = block(sequence)
        G_out = self.out(sequence)  # 最终输出
        return G_out.to(self.device)


# 3. Transformer判别器
class TransformerDiscriminator(nn.Module):
    def __init__(self, hidden_dim, num_layers, device):
        super(TransformerDiscriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        # 定义Transformer编码器层
        self.transformer_block = Transformer_block(hidden_dim, device, dropout=0.1)
        self.transformer_layers = nn.ModuleList([
            self.transformer_block
            for _ in range(num_layers)
        ])
        self.Dseq = SequenceEncoder(hidden_dim, device)
        self.classification_token = nn.Parameter(torch.randn(1, 1, hidden_dim)).to(device)
        self.out = nn.Linear(hidden_dim, 2).to(device)

    def forward(self, x, c):
        x = x.to(self.device)
        c = c.to(self.device)
        sequence = self.Dseq(x, c)
        for block in self.transformer_layers:
            sequence = block(sequence)
        batch_size = sequence.size(0)
        cls_token = self.classification_token.expand(batch_size, -1, -1)  # [batch_size, 1, hidden_dim]
        judge = torch.cat((cls_token, sequence), dim=1)
        out = self.out(judge[:, 0, :])  # [batch_size, input_dim] -> [batch_size, num_classes]
        return out.to(self.device)


def gradient_penalty(discriminator, real_data, fake_data, c, device, lambda_gp):
    """
     计算梯度惩罚项，适用于WGAN-GP。

     参数：
     - real_data: 真实数据，形状为 [batch_size, ...]
     - fake_data: 生成数据，形状与 real_data 相同
     - device: 运算设备，如 "cpu" 或 "cuda"
     - lambda_gp: 梯度惩罚系数，通常设置为10

     返回：
     - gp: 梯度惩罚项
     """
    batch_size = real_data.size(0)
    real_data = real_data.to(device)
    fake_data = fake_data.to(device)
    alpha = torch.rand(batch_size, *([1] * (real_data.dim() - 1)), device=device)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True).to(device)
    disc_interpolates = discriminator(interpolates, c)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    # 计算每个样本梯度的 L2 范数
    gradients = gradients.view(batch_size, -1)
    gradients_norm = gradients.norm(2, dim=1)
    # 计算梯度惩罚项
    gp = ((gradients_norm - 1) ** 2).mean()
    return gp


# 训练函数
def train(generator, discriminator, dataloader, window_size, step_size, hidden_dim, num_layers, num_epochs, device,
          lambda_gp):
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for energy_data, c_data in dataloader:
            # print(energy_data.shape)
            real_data = energy_data.float().to(device)
            c = c_data.float().to(device)
            real_data = real_data.unsqueeze(-1).to(device)
            c = c.unsqueeze(-1).to(device)
            # print(real_data.shape)

            batch_size = real_data.size(0)
            seq_len = real_data.size(1)
            # Train discriminator
            optimizer_D.zero_grad()
            z = torch.randn(batch_size, seq_len, 1).float().to(device)  # 随机噪声输入，大小为(batch_size, 8760,1)
            fake_data = generator(z, c)  # 传入噪声和温度数据
            # print('fake_data:', fake_data)
            real_pred = discriminator(real_data, c)  # 传入真实数据和温度数据
            # print('real_pred:', real_pred)
            fake_pred = discriminator(fake_data, c)  # 传入生成的数据和温度数据
            # print('fake_pred:', fake_pred)
            grad_penalty = gradient_penalty(discriminator, real_data, fake_data, c, device, lambda_gp)
            # print('grad_penalty:', grad_penalty)
            d_loss = torch.mean(real_pred) - torch.mean(fake_pred) + lambda_gp * grad_penalty

            discriminator.zero_grad()
            d_loss.backward(retain_graph=True)
            optimizer_D.step()

            # Train generator every 5 steps
            if epoch % 3 == 0:
                fake_pred = discriminator(fake_data, c)
                g_loss = -torch.mean(fake_pred)
                generator.zero_grad()
                g_loss.backward(retain_graph=True)
                optimizer_G.step()

        print(f'Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')


if __name__ == '__main__':
    torch.cuda.set_device(6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # batch_size=1
    hidden_dim = 64
    num_layers = 6
    num_epochs = 100
    output_dim = 1
    # seq_len=87600
    window_size = 720
    step_size = 600
    generator = TransformerGenerator(hidden_dim, num_layers, output_dim, device).to(device)
    discriminator = TransformerDiscriminator(hidden_dim, num_layers, device).to(device)
    energy_sliding, tem_sliding = read_data()
    dataloader = DataLoader(
        dataset=sample_data(energy_sliding, tem_sliding),
        batch_size=32,
        shuffle=False,
    )

    # 训练模型
    train(generator, discriminator, dataloader, window_size, step_size, hidden_dim, num_layers, num_epochs, device,
          lambda_gp=5)
