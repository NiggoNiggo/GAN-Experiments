import torch
from core.registries import LOSSES

@LOSSES.registry("wgan")
class WGANLoss():
    def __init__(self):
        pass
    def disc_loss(self,real_preds,fake_preds):
        return fake_preds.mean() - real_preds.mean()
    
    def gen_loss(self,fake_preds):
        return - fake_preds.mean()


@LOSSES.registry("wgan_gp")
class WGANGradientPenalty(WGANLoss):
    def __init__(self):
        super().__init__()

    def disc_loss(self, 
                  real_preds,
                  fake_preds,
                  x_hat,
                  x_hat_pred,
                  lambd):
        #compute wgan loss
        wgan_loss  = fake_preds.mean() - real_preds.mean()
        #compute gradients of x_hat with respect to x_hat_preds
        gradients = torch.autograd.grad(
            outputs=x_hat_pred,
            inputs=x_hat,
            grad_outputs=torch.ones_like(x_hat_pred),
            create_graph=True,
            retain_graph=True
        )[0]
        # prepeare gradients
        gradients = gradients.view(
            gradients.size(0), -1
        )
        #compute the norm of the gradients
        gradient_norm = torch.norm(
            gradients,
            p=2,
            dim=1
        )
        #compute the gradient norm to ensure lipschitz 1 
        gradient_penalty = (gradient_norm - 1).pow(2).mean()
        #return the entire loss
        return (wgan_loss + lambd * gradient_penalty)
        