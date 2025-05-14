from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from losses.gmef_ssim import clamp_min

from .box_filter import BoxFilter

class FastGuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(FastGuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)


    def forward(self, lr_x, lr_y, hr_x):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2*self.r+1 and w_lrx > 2*self.r+1

        ## N
        N = self.boxfilter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        ## mean_x
        mean_x = self.boxfilter(lr_x) / clamp_min(N, self.eps)
        ## mean_y
        mean_y = self.boxfilter(lr_y) / clamp_min(N, self.eps)
        ## cov_xy
        cov_xy = self.boxfilter(lr_x * lr_y) / clamp_min(N - mean_x * mean_y, self.eps)
        ## var_x
        var_x = self.boxfilter(lr_x * lr_x) / clamp_min(N - mean_x * mean_x, self.eps)

        ## A
        A = cov_xy / clamp_min(var_x, self.eps)
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear')
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear')

        return mean_A*hr_x+mean_b


class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)


    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N
        N = clamp_min(self.boxfilter(Variable(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0))), self.eps)

        # mean_x
        mean_x = self.boxfilter(x) / clamp_min(N, self.eps)
        # mean_y
        mean_y = self.boxfilter(y) / clamp_min(N, self.eps)
        # cov_xy
        cov_xy = self.boxfilter(x * y) / clamp_min(N - mean_x * mean_y, self.eps)
        # var_x
        var_x = self.boxfilter(x * x) / clamp_min(N - mean_x * mean_x, self.eps)

        # A
        A = cov_xy / clamp_min(var_x, self.eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        return mean_A * x + mean_b