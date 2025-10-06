import torch

class Evaluator:
    def __init__(self, mode="threshold", theta=0.5, tau=0.2):
        """
        mode: "threshold" or "distance"
        theta: 阈值分类模式下的分界点 (默认0.5)
        tau: 距离模式下的误差容忍度 (默认0.2)
        """
        assert mode in ["threshold", "distance"], "mode must be 'threshold' or 'distance'"
        self.mode = mode
        self.theta = theta
        self.tau = tau

    def evaluate(self, y_target: torch.Tensor, y_cf: torch.Tensor):
        """
        输入:
          y_target: [N] torch.Tensor, 可以是0/1或概率
          y_cf:     [N] torch.Tensor, 模型输出（0~1 概率）
        输出:
          R_mask: [N] torch.BoolTensor, True表示失败(需要回收)
          G_mask: [N] torch.BoolTensor, True表示成功(不用回收)
        """
        if self.mode == "threshold":
            # 先把概率转为 hard label
            target_label = (y_target >= self.theta).long()
            outcome_label = (y_cf >= self.theta).long()
            R_mask = (target_label != outcome_label)
        else:  # distance 模式
            delta = torch.abs(y_cf - y_target)
            R_mask = (delta > self.tau)

        G_mask = ~R_mask
        return R_mask, G_mask
