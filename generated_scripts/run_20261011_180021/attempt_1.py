import time
from pathlib import Path
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

data_path = Path(__file__).parent / "SNLS80mV.mat"
data = scipy.io.loadmat(data_path)

u_key = None
y_key = None
for k in data.keys():
    if k.startswith('__'):
        continue
    if 'V1' in k:
        u_key = k
    elif 'V2' in k:
        y_key = k

if u_key is None or y_key is None:
    data_vars = [k for k in data.keys() if not k.startswith('__')]
    for k in data_vars:
        arr = np.squeeze(data[k])
        if arr.ndim == 1 and len(arr) > 46746:
            if u_key is None:
                u_key = k
            elif y_key is None:
                y_key = k

u_raw = np.squeeze(data[u_key]).astype(np.float64)
y_raw = np.squeeze(data[y_key]).astype(np.float64)

fs = 610.35
dt = 1.0 / fs

u_excerpt = u_raw[42650:46746]
y_excerpt = y_raw[42650:46746]

u = u_excerpt - np.mean(u_excerpt)
y = y_excerpt - np.mean(y_excerpt)

n_train = 3072
n_test = 1024
total_samples = len(y)

u_train = u[:n_train]
y_train = y[:n_train]

y_scale = np.std(y_train)
u_scale = np.std(u_train)

v_train_fd = (y_train[1:] - y_train[:-1]) / dt
v_scale = np.std(v_train_fd)
w0 = v_scale / y_scale
H = w0 * dt

y_norm = torch.tensor(y / y_scale, dtype=torch.float32)
u_norm = torch.tensor(u / u_scale, dtype=torch.float32)

v_init = (y[1] - y[0]) / dt
v0_norm = torch.tensor(v_init / (w0 * y_scale), dtype=torch.float32)
y0_norm = y_norm[0]

class AccelerationNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, y_state, v_state, u_val):
        x = torch.stack([y_state, v_state, u_val], dim=-1)
        return self.net(x).squeeze(-1)

model = AccelerationNet(hidden_dim=64)

def rk4_step(model, y_curr, v_curr, u_curr, u_next, h):
    u_mid = 0.5 * (u_curr + u_next)
    
    k1_y = v_curr
    k1_v = model(y_curr, v_curr, u_curr)
    
    k2_y = v_curr + 0.5 * h * k1_v
    k2_v = model(y_curr + 0.5 * h * k1_y, k2_y, u_mid)
    
    k3_y = v_curr + 0.5 * h * k2_v
    k3_v = model(y_curr + 0.5 * h * k2_y, k3_y, u_mid)
    
    k4_y = v_curr + h * k3_v
    k4_v = model(y_curr + h * k3_y, k4_y, u_next)
    
    y_next = y_curr + (h / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y)
    v_next = v_curr + (h / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return y_next, v_next

window_len = 30
stride = 10

starts = list(range(0, n_train - window_len, stride))
num_windows = len(starts)

windows_u = []
windows_y = []
windows_v0 = []

for s in starts:
    windows_u.append(u_norm[s:s + window_len + 1])
    windows_y.append(y_norm[s:s + window_len + 1])
    v_start = (y_norm[s + 1] - y_norm[s]) / H
    windows_v0.append(v_start)

windows_u = torch.stack(windows_u)
windows_y = torch.stack(windows_y)
windows_v0 = torch.stack(windows_v0)

batch_size = 64
dataset = torch.utils.data.TensorDataset(windows_u, windows_y, windows_v0)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

max_epochs = 60
start_time = time.time()
final_loss = 0.0
epochs_completed = 0

for epoch in range(1, max_epochs + 1):
    model.train()
    total_loss = 0.0
    for batch_u, batch_y, batch_v0 in loader:
        optimizer.zero_grad()
        B = batch_u.size(0)
        curr_y = batch_y[:, 0]
        curr_v = batch_v0
        pred_y_list = [curr_y]
        
        for step in range(window_len):
            u_c = batch_u[:, step]
            u_n = batch_u[:, step + 1]
            curr_y, curr_v = rk4_step(model, curr_y, curr_v, u_c, u_n, H)
            pred_y_list.append(curr_y)
            
        pred_y_seq = torch.stack(pred_y_list, dim=1)
        loss = criterion(pred_y_seq, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * B
        
    epochs_completed = epoch
    final_loss = total_loss / num_windows
    if epoch % 10 == 0 or epoch == max_epochs:
        print(f"Epoch {epoch}/{max_epochs} - Training Loss: {final_loss:.6f}")
    if time.time() - start_time > 200:
        break

train_duration = time.time() - start_time

model.eval()
with torch.no_grad():
    sim_y = [y0_norm]
    curr_y = y0_norm.unsqueeze(0)
    curr_v = v0_norm.unsqueeze(0)
    
    for t in range(total_samples - 1):
        u_c = u_norm[t:t+1]
        u_n = u_norm[t+1:t+2]
        curr_y, curr_v = rk4_step(model, curr_y, curr_v, u_c, u_n, H)
        sim_y.append(curr_y.squeeze(0))
        
    sim_y_norm = torch.stack(sim_y).numpy()
    pred_y = sim_y_norm * y_scale

train_err = pred_y[:n_train] - y[:n_train]
test_err = pred_y[n_train:] - y[n_train:]

train_rmse_val = np.sqrt(np.mean(train_err ** 2))
test_rmse_val = np.sqrt(np.mean(test_err ** 2))

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
time_vec = np.arange(total_samples) * dt

axes[0].plot(time_vec, y, label="Measured", color="black", alpha=0.7)
axes[0].plot(time_vec, pred_y, label="Predicted", color="tab:blue", linestyle="--")
axes[0].axvline(n_train * dt, color="red", linestyle=":", label="Train/Test Split")
axes[0].set_ylabel("Output [V]")
axes[0].legend(loc="upper right")
axes[0].set_title("Silverbox Free-Run Neural ODE Rollout")

axes[1].plot(time_vec, pred_y - y, label="Error", color="tab:red")
axes[1].axvline(n_train * dt, color="red", linestyle=":")
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Error [V]")
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.savefig(Path(__file__).parent / "silverbox_neural_ode_result.png", dpi=150)
plt.close()

print(f"Network: 2 hidden layers, 64 units each, Tanh activation")
print(f"Optimizer: Adam, learning rate: {learning_rate}")
print(f"Window length: {window_len}, batch size: {batch_size}")
print(f"Epochs completed: {epochs_completed}")
print(f"Final training loss: {final_loss:.6f}")
print(f"Training time: {train_duration:.2f} seconds")
print(f"Train RMSE: {train_rmse_val:.6f}")
print(f"Test RMSE: {test_rmse_val:.6f}")