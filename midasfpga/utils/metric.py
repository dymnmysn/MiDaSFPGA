import torch
from midasfpga.utils.infer_utils import compute_scale_and_shift


class BadPixelMetric:
    def __init__(self, threshold=1.25, depth_cap=10):
        self.__threshold = threshold
        self.__depth_cap = depth_cap
        self.scaleshifts = []


    def __call__(self, prediction_disparity, target_depth, mask):
        # transform predicted disparity to aligned depth
        target_disparity = torch.zeros_like(target_depth)
        target_disparity[mask == 1] = 1.0 / target_depth[mask == 1]

        scale, shift = compute_scale_and_shift(prediction_disparity, target_disparity, mask)
        self.scaleshifts.append((scale,shift)) #-shift/scale
        prediction_aligned = scale.view(-1, 1, 1) * prediction_disparity + shift.view(-1, 1, 1)

        disparity_cap = 1.0 / self.__depth_cap
        prediction_aligned[prediction_aligned < disparity_cap] = disparity_cap

        prediction_depth = 1.0 / prediction_aligned

        # bad pixel
        pixel_ratios = torch.zeros_like(prediction_depth, dtype=torch.float)
        delta1 = torch.zeros_like(prediction_depth, dtype=torch.bool)
        delta2 = torch.zeros_like(prediction_depth, dtype=torch.bool)
        delta3 = torch.zeros_like(prediction_depth, dtype=torch.bool)

        pixel_ratios[mask == 1] = torch.max(
            prediction_depth[mask == 1] / target_depth[mask == 1],
            target_depth[mask == 1] / prediction_depth[mask == 1],
        )

        delta1[mask == 1] = (pixel_ratios[mask == 1] > self.__threshold)
        delta1_avg = torch.sum(delta1, (1, 2)) / torch.sum(mask, (1, 2)) #delta average over pixels

        delta2[mask == 1] = (pixel_ratios[mask == 1] > (self.__threshold)**2)
        delta2_avg = torch.sum(delta2, (1, 2)) / torch.sum(mask, (1, 2))

        delta3[mask == 1] = (pixel_ratios[mask == 1] > (self.__threshold)**3)
        delta3_avg = torch.sum(delta3, (1, 2)) / torch.sum(mask, (1, 2))

        d1_err_percent = 100 * torch.mean(delta1_avg) # delta average over batch
        d2_err_percent = 100 * torch.mean(delta2_avg)
        d3_err_percent = 100 * torch.mean(delta3_avg)

        return d1_err_percent, d2_err_percent, d3_err_percent, prediction_depth, target_depth, pixel_ratios