import gluoncv as gcv
from mxnet.gluon import loss


class CenterNetLoss(loss.Loss):
    def __init__(self, center_weight=1, wh_weight=0.5, reg_weight=0.1,
                 from_logits=True, batch_axis=0, weight=None, **kwargs):

        super(CenterNetLoss, self).__init__(weight, batch_axis, **kwargs)

        self.center_weight = center_weight
        self.wh_weight = wh_weight
        self.reg_weight = reg_weight

        self.center_loss = gcv.loss.HeatmapFocalLoss(
            from_logits=from_logits, batch_axis=batch_axis, weight=weight)
        self.wh_loss = gcv.loss.MaskedL1Loss(
            batch_axis=batch_axis, weight=weight)
        self.reg_loss = gcv.loss.MaskedL1Loss(
            batch_axis=batch_axis, weight=weight)

    def hybrid_forward(self, F, heatmap_pred, heatmap_label, wh_pred, wh_label, 
                       wh_mask, reg_pred, reg_label, reg_mask):
                       
        center = self.center_loss(heatmap_pred, heatmap_label)
        wh = self.wh_loss(wh_pred, wh_label, wh_mask)
        reg = self.reg_loss(reg_pred, reg_label, reg_mask)

        loss = self.center_weight*center + self.wh_weight*wh + self.reg_weight*reg

        return loss
