import torch


def block_batch_norm(
    dualres: "DualResTensor",
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    training: bool,
    momentum: float,
    eps: float,
):
    hr, lr = dualres.highres, dualres.lowres

    if training:
        exponential_average_factor = 0.0 if momentum is None else momentum

        def get_var_mean(num_blocks, x):
            if num_blocks == 0:
                return 0, 0, 0
            n = x.numel() / x.shape[1]
            var, mean = torch.var_mean(hr.detach(), [0, 2, 3], unbiased=False)
            return n, var, mean

        n_hr, var_hr, mean_hr = get_var_mean(dualres.metadata.nhighres, hr)
        n_lr, var_lr, mean_lr = get_var_mean(dualres.metadata.nlowres, lr)

        n = n_hr + n_lr
        var = var_hr * (n_hr / n) + var_lr * (n_lr / n)
        mean = mean_hr * (n_hr / n) + mean_lr * (n_lr / n)

        with torch.no_grad():
            running_mean[:] = exponential_average_factor * mean + (1 - exponential_average_factor) * running_mean
            running_var[:] = (
                exponential_average_factor * var * n / (n - 1) + (1 - exponential_average_factor) * running_var
            )

    if dualres.metadata.nhighres > 0:
        hr = torch.nn.functional.batch_norm(
            hr, running_mean, running_var, weight, bias, training=training, momentum=0, eps=eps
        )

    if dualres.metadata.nlowres > 0:
        lr = torch.nn.functional.batch_norm(
            lr, running_mean, running_var, weight, bias, training=training, momentum=0, eps=eps
        )

    return hr, lr
