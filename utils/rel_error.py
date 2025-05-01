def rel_err(x, y, eps=1e-12):
    """
    rel_error = (max_i (x_i - y_i)) / (max_i (|x_i| + |y_i|) + eps)

    input:
    - x, y: two tensors of the same shape
    - eps: a small number to avoid division by zero

    output:
    - rel_error: a scalar representing the relative error between x and y
    """
    top = (x - y).abs().max().item()
    bot = (x.abs() + y.abs()).clamp(min=eps).max().item()
    return top / bot
