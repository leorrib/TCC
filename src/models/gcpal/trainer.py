import math
import torch
from torch import optim
from torch.cuda.amp import autocast, GradScaler

from .augmentations import make_random_views
from .contrastive_loss import contrastive_loss_tiled

class GINPretrainer:
    """
    Runs the exact pretraining loop you wrote (with AMP, early-stopping style counter).
    You pass in: data_train_global, edge_index_knn, encoder, proj_head, and hyperparams.
    """

    def __init__(
        self,
        device=None,
        lambda_mix: float = 0.5,
        tau: float = 0.5,
        lr: float = 1e-3,
        max_epochs: int = 20,
        anchor_bs: int = 2048,
        target_bs: int = 32768,
        patience: int = 999,
        drop_p_edge: float = 0.3,
        drop_p_feat: float = 0.3,
    ):
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.lambda_mix = lambda_mix
        self.tau = tau
        self.lr = lr
        self.max_epochs = max_epochs
        self.anchor_bs = anchor_bs
        self.target_bs = target_bs
        self.patience = patience
        self.drop_p_edge = drop_p_edge
        self.drop_p_feat = drop_p_feat

    def fit(self, data_train_global, edge_index_knn, encoder, proj_head, pos_lists_struct, pos_lists_knn):
        optimizer = optim.Adam(list(encoder.parameters()) + list(proj_head.parameters()), lr=self.lr)
        scaler = GradScaler()

        best_loss = math.inf
        epochs_no_improve = 0

        encoder.train()
        proj_head.train()

        for epoch in range(1, self.max_epochs + 1):
            # random views (same logic)
            (x1, e1), (x2, e2) = make_random_views(data_train_global, self.drop_p_edge, self.drop_p_feat)
            e_knn = edge_index_knn

            with autocast():
                h1 = encoder(x1.to(self.device), e1.to(self.device))
                h2 = encoder(x2.to(self.device), e2.to(self.device))
                h_knn = encoder(x2.to(self.device), e_knn.to(self.device))

                z1 = proj_head(h1)
                z2 = proj_head(h2)
                z_knn = proj_head(h_knn)

            loss_rand = contrastive_loss_tiled(
                z1, z2, pos_lists=pos_lists_struct, tau=self.tau,
                anchor_bs=self.anchor_bs, target_bs=self.target_bs, device=self.device
            )
            loss_knn = contrastive_loss_tiled(
                z1, z_knn, pos_lists=pos_lists_knn, tau=self.tau,
                anchor_bs=self.anchor_bs, target_bs=self.target_bs, device=self.device
            )

            loss = self.lambda_mix * loss_rand + (1 - self.lambda_mix) * loss_knn

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if loss.item() < best_loss - 1e-4:
                best_loss = loss.item()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            print(f"[{epoch:02d}/{self.max_epochs}] "
                  f"loss_rand={loss_rand.item():.4f} | "
                  f"loss_knn={loss_knn.item():.4f} | "
                  f"loss={loss.item():.4f} | "
                  f"best={best_loss:.4f} | no_improve={epochs_no_improve}")

            if epochs_no_improve >= self.patience:
                print(f"⏹ Early stopping em epoch {epoch} (sem melhora por {self.patience} épocas).")
                break

        print("✅ Pré-treinamento contrastivo finalizado.")
        return {"best_loss": best_loss}
