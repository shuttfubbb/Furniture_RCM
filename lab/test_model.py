import torch
import torch.nn as nn
import torch.distributions as D

class HybridPolicy(nn.Module):
    def __init__(self, state_dim, num_discrete):
        super().__init__()
        hidden_dim = 64
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Gaussian head cho (x, y)
        self.mean = nn.Linear(hidden_dim, 2)  # output (mu_x, mu_y)
        self.log_std = nn.Parameter(torch.zeros(2))

        # Categorical head cho k
        self.logits = nn.Linear(hidden_dim, num_discrete)

    def forward(self, state):
        x = self.fc(state)
        mu = self.mean(x)
        std = self.log_std.exp()
        logits = self.logits(x)
        return mu, std, logits

    def sample_action(self, state):
        mu, std, logits = self.forward(state)

        # Continuous part (x, y)
        dist_cont = D.Normal(mu, std)
        xy = dist_cont.rsample()   # (batch, 2)
        log_prob_xy = dist_cont.log_prob(xy).sum(dim=-1)

        # Discrete part (k)
        dist_disc = D.Categorical(logits=logits)
        k = dist_disc.sample()     # (batch,)
        log_prob_k = dist_disc.log_prob(k)

        # Gộp lại
        action = torch.cat([xy, k.unsqueeze(-1).float()], dim=-1)
        log_prob = log_prob_xy + log_prob_k

        return action, log_prob


if __name__ == "__main__":
    state_dim = 200
    num_discrete = 5   # ví dụ k có 5 lựa chọn

    policy = HybridPolicy(state_dim, num_discrete)

    state = torch.randn(1, state_dim)

    action, log_prob = policy.sample_action(state)

    print("State input:", state.shape)
    print("Sampled action [x, y, k]:", action)
    print("Log prob of action:", log_prob)
