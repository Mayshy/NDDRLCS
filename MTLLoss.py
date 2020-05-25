import torch
from torch import nn

# mtl_loss_layer = MultiLossLayer(loss_list)
# mtl_loss = mtl_loss_layer()
class MultiLossLayer(nn.Module):
    def __init__(self, list_length):
        super(MultiLossLayer, self).__init__()
        self._sigmas_sq = nn.ParameterList([nn.Parameter(torch.empty(())) for i in range(list_length)])
        for p in self.parameters():
            nn.init.uniform_(p,0.2,1)
            # 初始化采用和原论文一样的方法......可能需要调整
        
    # def get_loss(self):
    #     factor = tf.div(1.0, tf.multiply(2.0, self._sigmas_sq[0]))
    #     loss = tf.add(tf.multiply(factor, self._loss_list[0]), tf.log(self._sigmas_sq[0]))
    #     for i in range(1, len(self._sigmas_sq)):
    #         factor = tf.div(1.0, tf.multiply(2.0, self._sigmas_sq[i]))
    #         loss = tf.add(loss, tf.add(tf.multiply(factor, self._loss_list[i]), tf.log(self._sigmas_sq[i])))
    #     return loss

    def forward(self,  loss0, loss1):
        factor0 = torch.div(1.0,torch.mul(self._sigmas_sq[0], 2.0))
        loss = torch.add(torch.mul(factor0, loss0), 0.5*torch.log(self._sigmas_sq[0]))
        
        factor1 = torch.div(1.0,torch.mul(self._sigmas_sq[1], 2.0))
        loss = torch.add(loss, torch.add(torch.mul(factor1, loss1), 0.5*torch.log(self._sigmas_sq[1])))

        return loss

    # def forward(self,  loss_list): Maybe need modulelist
    #     factor = torch.div(1.0,torch.mul(self._sigmas_sq[0], 2.0))
    #     loss = torch.add(torch.mul(factor, loss_list[0]), torch.log(self._sigmas_sq[0]))
    #     for i in range(1, len(self._sigmas_sq)):
    #         factor = torch.div(1.0,torch.mul(self._sigmas_sq[i], 2.0)) #TODO
    #         loss = torch.add(loss, torch.add(torch.mul(factor, loss_list[i]), torch.log(self._sigmas_sq[i])))

    #     return loss

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


class XTanhLoss(torch.nn.Module):
    def __init__(self):
        super(XTanhLoss,self).__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(ey_t * torch.tanh(ey_t))


class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super(XSigmoidLoss,self).__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t)