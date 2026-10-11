import time
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Load data
data_path = Path(__file__).parent / "SNLS80mV.mat"
data = loadmat(data_path)

# Inspect keys for input (V1) and output (V2)
keys = {k.lower(): k for k in data.keys() if not k.startswith("__")}
if "v1" in keys and "v2" in keys:
    u_raw = np.asarray(data[keys["v1"]], dtype=np.float64).squeeze()
    y_raw = np.asarray(data[keys["v2"]], dtype=np.float64).squeeze()
else:
    non_meta = [k for k in data.keys() if not k.startswith("__")]
    u_raw = np.asarray(data[non_meta[0]], dtype=np.float64).squeeze()
    y_raw = np.asarray(data[non_meta[1]], dtype=np.float64).squeeze()

# Excerpt [42650:46746]
u_excerpt = u_raw[42650:46746]
y_excerpt = y_raw[42650:46746]

# Subtract mean of excerpt
u = u_excerpt - np.mean(u_excerpt)
y = y_excerpt - np.mean(y_excerpt)

total_len = len(y)  # 4096
train_len = 3072
test_len = 1024

fs = 610.35
dt = 1.0 / fs

# Compute scaling factors from training data only
u_train = u[:train_len]
y_train = y[:train_len]

Y_S = float(np.std(y_train))
U_S = float(np.std(u_train))

v_fd_train = (y_train[1:] - y_train[:-1]) / dt
W_0 = float(np.std(v_fd_train) / Y_S)
h = float(W_0 * dt)

# Scale signals
y_scaled = y / Y_S
u_scaled = u / U_S

v_fd_scaled = np.zeros_like(y_scaled)
v_fd_scaled[:-1] = (y_scaled[1:] - y_scaled[:-1]) / h
v_fd_scaled[-1] = v_fd_scaled[-2]

# Convert to PyTorch tensors for training
y_train_t = torch.tensor(y_scaled[:train_len], dtype=torch.float32)
u_train_t = torch.tensor(u_scaled[:train_len], dtype=torch.float32)
v_train_t = torch.tensor(v_fd_scaled[:train_len], dtype=torch.float32)

# 2. Neural ODE Network Architecture
class NeuralODEFunc(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        # Small weights on output layer for stable initial integration
        nn.init.uniform_(self.net[-1].weight, -0.05, 0.05)

    def forward(self, x):
        return self.net(x)

def rk4_step(func, y_curr, v_curr, u_curr, u_next, step_size):
    # k1
    k1_y = v_curr
    inp1 = torch.stack([y_curr, v_curr, u_curr], dim=-1)
    k1_v = func(inp1).squeeze(-1)

    # k2
    y2 = y_curr + 0.5 * step_size * k1_y
    v2 = v_curr + 0.5 * step_size * k1_v
    u_mid = 0.5 * (u_curr + u_next)
    k2_y = v2
    inp2 = torch.stack([y2, v2, u_mid], dim=-1)
    k2_v = func(inp2).squeeze(-1)

    # k3
    y3 = y_curr + 0.5 * step_size * k2_y
    v3 = v_curr + 0.5 * step_size * k2_v
    k3_y = v3
    inp3 = torch.stack([y3, v3, u_mid], dim=-1)
    k3_v = func(inp3).squeeze(-1)

    # k4
    y4 = y_curr + step_size * k3_y
    v4 = v_curr + step_size * k3_v
    k4_y = v4
    inp4 = torch.stack([y4, v4, u_next], dim=-1)
    k4_v = func(inp4).squeeze(-1)

    y_next = y_curr + (step_size / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y)
    v_next = v_curr + (step_size / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return y_next, v_next

# 3. Training setup
model = NeuralODEFunc(hidden_dim=64)
lr = 2e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-4)

window_len = 40
batch_size = 32
num_epochs = 80
batches_per_epoch = 30

start_time = time.time()
final_loss = 0.0

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for _ in range(batches_per_epoch):
        optimizer.zero_grad()
        idx = torch.randint(0, train_len - window_len, (batch_size,))
        y_curr = y_train_t[idx]
        v_curr = v_train_t[idx]

        y_preds = []
        v_preds = []
        y_targs = []
        v_targs = []

        for step in range(window_len - 1):
            u_k = u_train_t[idx + step]
            u_kp1 = u_train_t[idx + step + 1]
            y_curr, v_curr = rk4_step(model, y_curr, v_curr, u_k, u_kp1, h)
            y_preds.append(y_curr)
            v_preds.append(v_curr)
            y_targs.append(y_train_t[idx + step + 1])
            v_targs.append(v_train_t[idx + step + 1])

        y_preds_stack = torch.stack(y_preds, dim=1)
        y_targs_stack = torch.stack(y_targs, dim=1)
        v_preds_stack = torch.stack(v_preds, dim=1)
        v_targs_stack = torch.stack(v_targs, dim=1)

        loss = nn.functional.mse_loss(y_preds_stack, y_targs_stack) + 0.1 * nn.functional.mse_loss(v_preds_stack, v_targs_stack)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()
    epoch_loss /= batches_per_epoch
    final_loss = epoch_loss

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}: Training loss = {epoch_loss:.6e}")

training_time = time.time() - start_time

# 4. Full Free-Run Evaluation
v0_scaled = (y_scaled[1] - y_scaled[0]) / h
u_seq_tensor = torch.tensor(u_scaled, dtype=torch.float32)

y_sim_scaled = torch.zeros(total_len, dtype=torch.float32)
y_sim_scaled[0] = torch.tensor(y_scaled[0], dtype=torch.float32)

y_curr = torch.tensor([y_scaled[0]], dtype=torch.float32)
v_curr = torch.tensor([v0_scaled], dtype=torch.float32)

model.eval()
with torch.no_grad():
    for k in range(total_len - 1):
        u_k = u_seq_tensor[k:k+1]
        u_kp1 = u_seq_tensor[k+1:k+2]
        y_curr, v_curr = rk4_step(model, y_curr, v_curr, u_k, u_kp1, h)
        y_sim_scaled[k + 1] = y_curr[0]

# Denormalize predicted output to physical volts
y_pred = (y_sim_scaled.numpy()) * Y_S

# 5. Compute Error Metrics in Volts
train_rmse = float(np.sqrt(np.mean((y[:train_len] - y_pred[:train_len]) ** 2)))
test_rmse = float(np.sqrt(np.mean((y[train_len:] - y_pred[train_len:]) ** 2)))

# 6. Plot and save results
time_vec = np.arange(total_len) * dt
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time_vec, y, label="Measured", color="black", alpha=0.7)
plt.plot(time_vec, y_pred, label="Predicted (Free Run)", color="tab:red", linestyle="--", alpha=0.8)
plt.axvline(x=train_len * dt, color="tab:blue", linestyle=":", label="Train/Test Split")
plt.ylabel("Output (V)")
plt.title("Silverbox Neural ODE Free-Run Simulation")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(time_vec, y - y_pred, label="Simulation Error", color="tab:gray")
plt.axvline(x=train_len * dt, color="tab:blue", linestyle=":")
plt.xlabel("Time (s)")
plt.ylabel("Error (V)")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / "silverbox_free_run.png", dpi=150)
plt.close()

# 7. Print summary and required outputs
print("Network: 3 layers (2 hidden), 64 hidden units, Tanh activation")
print(f"Optimizer: Adam, initial learning rate = {lr}")
print(f"Window length: {window_len}, batch size: {batch_size}")
print(f"Epochs completed: {num_epochs}")
print(f"Final training loss: {final_loss:.6e}")
print(f"Training time: {training_time:.2f} seconds")
print(f"Train RMSE: {train_rmse:.6f}")
print(f"Test RMSE: {test_rmse:.6f}")