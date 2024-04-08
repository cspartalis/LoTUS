import torch


def _custom_scaling(x):
    """
    Custom minmax scaling, where the positives are scaled to [0, 1]
    and the negatives are scaled to [-1, 0], and zeros remain zeros.
    """
    pos_mask = x > 0
    neg_mask = x < 0
    zero_mask = ~(pos_mask | neg_mask)

    # Calculate scaling factors for positive and negative values
    pos_scale = 1 / torch.max(x[pos_mask])
    neg_scale = 1 / torch.abs(torch.min(x[neg_mask]))

    scaled_x = x.clone()
    scaled_x[pos_mask] = x[pos_mask] * pos_scale
    scaled_x[neg_mask] = x[neg_mask] * neg_scale
    scaled_x[zero_mask] = 0.0  # Ensure zeros remain zeros

    return scaled_x


# Example usage
x = torch.tensor([-0.3, -0.2, -0.1, 0, 0, 0, 0.3, 0.3, 0.4, 0.5, 0.6, 1.2])
scaled_x = _custom_scaling(x)

print(scaled_x)
