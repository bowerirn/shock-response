import torch
import srs_ext


class Iso_SRS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, As, Bs):
        """
        X:  (B, T) or (B, 1, T)
        As: (F, 2), (2,), or (1, 2)
        Bs: (F, 3), (3,), or (1, 3)

        Returns:
            srs: (B, F)
        """
        if X.dim() == 3:
            if X.size(1) != 1:
                raise ValueError(f"Expected X with shape (B, 1, T) when dim=3, got {tuple(X.shape)}")
            X_in = X[:, 0, :]
        elif X.dim() == 2:
            X_in = X
        else:
            raise ValueError(f"Expected X with shape (B, T) or (B, 1, T), got {tuple(X.shape)}")

        X_in = X_in.contiguous()
        As = As.contiguous()
        Bs = Bs.contiguous()

        srs, argmax, sign = srs_ext.forward(X_in, As, Bs)

        ctx.save_for_backward(argmax, sign, As, Bs)
        ctx.x_shape = tuple(X.shape)
        ctx.T = X_in.size(-1)

        return srs

    @staticmethod
    def backward(ctx, grad_srs):
        """
        grad_srs: (B, F)

        Returns gradients for:
            X, As, Bs

        Only grad wrt X is implemented.
        """
        argmax, sign, As, Bs = ctx.saved_tensors
        grad_srs = grad_srs.contiguous()

        grad_x_2d = srs_ext.backward(
            grad_srs,
            argmax,
            sign,
            As,
            Bs,
            ctx.T,
        )

        # Restore original input shape
        if len(ctx.x_shape) == 3:
            grad_x = grad_x_2d.unsqueeze(1)
        else:
            grad_x = grad_x_2d

        return grad_x, None, None, None