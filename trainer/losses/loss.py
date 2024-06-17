import torch
import torch.nn as nn
import torch.nn.functional as F

def swish(x):
    return x.mul(torch.sigmoid(x))

def log_norm_pdf(x, mu, logvar):
    return -0.5 * (
        logvar + torch.log(2 * torch.tensor(torch.pi)) +
        (x - mu).pow(2) / logvar.exp()
    )

class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()

    def forward(self, embeddings, positives):
        scores = F.binary_cross_entropy_with_logits(
            embeddings, positives.float(), reduction="none"
        )
        return scores.mean(dim=1).mean()


class BinaryCrossEntropyLossSoftmax(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLossSoftmax, self).__init__()

    def forward(self, recon_x, x, mu=None, logvar=None, anneal=1.0):

        if logvar is not None:
            BCE = -torch.mean(torch.mean(F.log_softmax(recon_x, 1) * x, -1))

            KLD = -0.5 * torch.mean(
                torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            )

        else:
            BCE = -torch.mean(torch.mean(F.log_softmax(recon_x, 1) * x, -1))
            KLD = torch.zeros_like(BCE)

        return BCE + anneal * KLD

class OT_loss(nn.Module):
    def __init__(self, args):
        
        super(OT_loss, self).__init__()
        self.args = args
        self.kfac = args.kfac
        self.split = args.emb_dim // self.kfac
        
    def bures_wasserstein_distance_vectorized(self, means1, covs1, means2, covs2):
        # Compute squared L2 norm of differences in means
        mean_diff = means1 - means2
        # Sum over columns to get a vector of shape (b,)
        mean_dist_squared = torch.sum(mean_diff**2, dim=1)

        # Cholesky decomposition of covs1 in batch
        sqrt_covs1 = torch.linalg.cholesky(covs1)

        # Batch matrix product: sqrt_covs1 * covs2 * sqrt_covs1
        # First part of the product: intermediate = sqrt_covs1 * covs2
        intermediate = torch.bmm(sqrt_covs1, covs2)
        # Second part of the product: product_matrix = intermediate * sqrt_covs1.transpose(1, 2)
        product_matrix = torch.bmm(intermediate, sqrt_covs1.transpose(1, 2))

        # Cholesky decomposition of the product_matrix in batch
        sqrt_middle = torch.linalg.cholesky(product_matrix)

        # Trace of sqrt_middle
        trace_sqrt_middle = torch.diagonal(
            sqrt_middle, dim1=-2, dim2=-1).sum(-1)

        # Trace of cov1 + cov2
        trace_covs1 = torch.diagonal(covs1, dim1=-2, dim2=-1).sum(-1)
        trace_covs2 = torch.diagonal(covs2, dim1=-2, dim2=-1).sum(-1)
        trace_cov_sum = trace_covs1 + trace_covs2

        # Total trace term
        trace_term = trace_cov_sum - 2 * trace_sqrt_middle

        # Total Bures-Wasserstein distance
        distance = mean_dist_squared + trace_term
        return distance
    def forward(
        self,
        recon_x,
        logits_text,
        logits_rec,
        x,
        mu,
        logvar,
        rec_mu,
        rec_logvar,
        anneal,
        epsilon,
    ):
            
        
        BCE_merged = - \
            torch.mean(torch.mean(F.log_softmax(recon_x, 1) * x, -1))
        BCE_text = - \
            torch.mean(torch.mean(F.log_softmax(logits_text, 1) * x, -1))
        BCE_rec = - \
            torch.mean(torch.mean(F.log_softmax(logits_rec, 1) * x, -1))

        BCE = (1 / 3) * (
            BCE_merged * self.args.recon_tau
            + BCE_text * self.args.text_tau
            + BCE_rec * self.args.rec_tau
        )

        if self.args.embedding_module == 'TearsMacrid':
            #make sure to aply KLD across factor dimensions
            #Can use simplified KLD since embeddings are normalized across factors
            logvar = logvar.view(-1, self.kfac, self.split)
            KLD = -0.5 * torch.mean(torch.mean(1 + logvar  - logvar.exp(), dim=-1))
            logvar = logvar.view(-1, self.args.emb_dim)
        else: 
            KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            

        sigma2 = torch.exp(logvar)
        sigma2_rec = torch.exp(rec_logvar)
        sigma2_diag = torch.stack([torch.diag(v) for v in sigma2])
        sigma2_rec_diag = torch.stack([torch.diag(v) for v in sigma2_rec])

        wasserstein_loss = self.bures_wasserstein_distance_vectorized(
            mu, sigma2_diag, rec_mu, sigma2_rec_diag
        ).mean()
        
        l = BCE + anneal  * KLD  + epsilon * wasserstein_loss

        return l, BCE, wasserstein_loss, BCE_rec, BCE_text, BCE_merged


class RecVAE_loss(OT_loss):
    def __init__(self, args):
        super( self).__init__()
        self.args = args

    def forward(
        self,
        recon_x,
        x,
        z,
        mu,
        logvar,
        prior,
        gamma,
        train_items,
        **kwargs
    ):
        norm = train_items.sum(dim=-1)
        kl_weight = gamma * norm
        mll = (F.log_softmax(recon_x, dim=-1) * x).sum(dim=-1).mean()
        kld = (
            (log_norm_pdf(z, mu, logvar) - prior(x, z))
            .sum(dim=-1)
            .mul(kl_weight)
            .mean()
        )
        negative_elbo = -(mll - kld)
        return negative_elbo

class PriorBCE(OT_loss):
    def __init__(self):
        super( self).__init__()

    def forward(
        self,
        recon_x,
        logits_rec,
        logits_text,
        x,
        mu=None,
        logvar=None,
        prior_mu=None,
        prior_logvar=None,
        P=None,
        S=None,
        anneal=1.0,
        epsilon=1,
        regularization_type="none",
    ):

        BCE_merged = -torch.mean(torch.mean(F.log_softmax(recon_x, 1) * x, -1))
        BCE_text = - \
            torch.mean(torch.mean(F.log_softmax(logits_text, 1) * x, -1))
        BCE_rec = -torch.mean(torch.mean(F.log_softmax(logits_rec, 1) * x, -1))

        BCE = (1 / 3) * (BCE_merged + BCE_text + BCE_rec)

        KLD1 = -0.5 * torch.mean(
            torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )

        KLD2 = -0.5 * torch.mean(
            torch.mean(1 + prior_logvar - prior_mu.pow(2) -
                       prior_logvar.exp(), dim=1)
        )

        sigma2 = torch.exp(logvar)
        sigma2_prior = torch.exp(prior_logvar)
        sigma2_diag = torch.stack([torch.diag(v) for v in sigma2])
        sigma2_prior_diag = torch.stack([torch.diag(v) for v in sigma2_prior])

        wasserstein_loss = self.bures_wasserstein_distance_vectorized(
            mu, sigma2_diag, prior_mu, sigma2_prior_diag
        ).mean()
        if regularization_type == "none":
            l = BCE + (anneal / 2) * (KLD1 + KLD2)
        if regularization_type == "OT":
            l = BCE + (anneal / 2) * (KLD1 + KLD2) + epsilon * wasserstein_loss

        return l, BCE, wasserstein_loss, BCE_rec, BCE_text, BCE_merged


class KLDivergenceLoss(nn.Module):
    def __init__(self, args, beta=0.5):
        super(KLDivergenceLoss, self).__init__()
        self.temp = args.temp
        # setup anneal schedule
        self.BCE = BinaryCrossEntropyLossSoftmax()
        self.beta = beta
        self.anneal = args.anneal
        self.min_beta = 0.0

    def forward(self, embeddings, positives, beta=0.5):

        probs = F.log_softmax(embeddings / self.temp, dim=1)
        targets = F.softmax(positives.float() / self.temp, dim=1)

        scores = F.kl_div(probs, targets, reduction="none")
        bce_loss = self.BCE(embeddings, positives)

        total_loss = self.beta * \
            scores.sum(dim=1).mean() + (1 - self.beta) * bce_loss
        return total_loss, scores.mean(dim=1).mean(), bce_loss

    def anneal_beta(self, step, max_step):
        if self.anneal:
            return min(self.min_beta, self.beta * (step / max_step))
        else:
            return self.beta


def get_loss(loss_name, args=None):
    if loss_name == "bce":
        return BinaryCrossEntropyLoss()
    elif loss_name == "bce_softmax":
        return BinaryCrossEntropyLossSoftmax()
    elif loss_name == "kl":

        return KLDivergenceLoss(args)
    elif loss_name == "ot_loss":
        return OT_loss(args)
    elif loss_name == "RecVAE_loss":
        return RecVAE_loss(args)
    elif loss_name == "Macrid_loss":
        return MacridTEARSLLoss(args)
    else:
        raise ValueError(f"loss {loss_name} not supported")
