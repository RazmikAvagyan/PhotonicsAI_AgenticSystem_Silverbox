import time
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Load data
    data_path = Path(__file__).parent / "SNLS80mV.mat"
    mat = sio.loadmat(data_path)

    # Find V1 and V2 keys dynamically
    v1_key = None
    v2_key = None
    for k in mat.keys():
        if k.startswith('__'):
            continue
        if 'V1' in k:
            v1_key = k
        elif 'V2' in k:
            v2_key = k

    if v1_key is None or v2_key is None:
        # Fallback to key matching by inspection
        candidates = [k for k in mat.keys() if not k.startswith('__')]
        for k in candidates:
            if '1' in k:
                v1_key = k
            elif '2' in k:
                v2_key = k

    u_full = np.squeeze(mat[v1_key])
    y_full = np.squeeze(mat[v2_key])

    # Excerpt [42650:46746] -> length 4096
    start_idx = 42650
    end_idx = 46746
    u_excerpt = u_full[start_idx:end_idx].astype(np.float32)
    y_excerpt = y_full[start_idx:end_idx].astype(np.float32)

    # Center input and output over excerpt
    u_excerpt = u_excerpt - np.mean(u_excerpt)
    y_excerpt = y_excerpt - np.mean(y_excerpt)

    n_total = len(y_excerpt) # 4096
    n_train = 3072
    n_test = 1024

    fs = 610.35
    dt = 1.0 / fs

    # 2. Rescaling
    u_train = u_excerpt[:n_train]
    y_train = y_excerpt[:n_train]

    y_scale = float(np.std(y_train))
    u_scale = float(np.std(u_train))

    v_train_raw = (y_train[1:] - y_train[:-1]) / dt
    v_scale = float(np.std(v_train_raw))
    omega_0 = v_scale / y_scale
    h = omega_0 * dt

    u_norm = torch.tensor(u_excerpt / u_scale, dtype=torch.float32)
    y_norm = torch.tensor(y_excerpt / y_scale, dtype=torch.float32)

    # 3. Model Definition
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
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, y, v, u):
            x = torch.stack([y, v, u], dim=-1)
            return self.net(x).squeeze(-1)

    model = AccelerationNet(hidden_dim=64)

    def rk4_step(y, v, u_curr, u_next, model_fn, step_h):
        u_mid = 0.5 * (u_curr + u_next)

        k1_y = v
        k1_v = model_fn(y, v, u_curr)

        y2 = y + 0.5 * step_h * k1_y
        v2 = v + 0.5 * step_h * k1_v
        k2_y = v2
        k2_v = model_fn(y2, v2, u_mid)

        y3 = y + 0.5 * step_h * k2_y
        v3 = v + 0.5 * step_h * k2_v
        k3_y = v3
        k3_v = model_fn(y3, v3, u_mid)

        y4 = y + step_h * k3_y
        v4 = v + step_h * k3_v
        k4_y = v4
        k4_v = model_fn(y4, v4, u_next)

        y_next = y + (step_h / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y)
        v_next = v + (step_h / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        return y_next, v_next

    # 4. Trajectory Dataset for Training
    L = 24
    class TrajectoryDataset(Dataset):
        def __init__(self, y_all, u_all, n_samples, window_len, step_h):
            self.y = y_all
            self.u = u_all
            self.L = window_len
            self.h = step_h
            self.num_windows = n_samples - window_len

        def __len__(self):
            return self.num_windows

        def __getitem__(self, idx):
            y_init = self.y[idx]
            v_init = (self.y[idx + 1] - self.y[idx]) / self.h
            u_seq = self.u[idx : idx + self.L + 1]
            y_target = self.y[idx + 1 : idx + self.L + 1]
            return y_init, v_init, u_seq, y_target

    train_dataset = TrajectoryDataset(y_norm, u_norm, n_train, L, h)
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    lr = 2e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 5. Training Loop
    start_time = time.time()
    max_time_sec = 210.0  # Safe headroom within 5-minute (300s) CPU limit
    num_epochs = 60
    final_loss = 0.0
    epochs_completed = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for y0, v0, u_seq, y_tgt in train_loader:
            optimizer.zero_grad()
            curr_y = y0
            curr_v = v0
            pred_y_list = []

            for step in range(L):
                u_c = u_seq[:, step]
                u_n = u_seq[:, step + 1]
                curr_y, curr_v = rk4_step(curr_y, curr_v, u_c, u_n, model, h)
                pred_y_list.append(curr_y)

            pred_y_traj = torch.stack(pred_y_list, dim=1)
            loss = criterion(pred_y_traj, y_tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / batches
        final_loss = avg_loss
        epochs_completed = epoch

        if epoch % 10 == 0 or epoch == num_epochs:
            print(f"Epoch {epoch:02d} / {num_epochs} - Loss: {avg_loss:.6f}")

        if time.time() - start_time > max_time_sec:
            break

    training_time = time.time() - start_time

    # 6. Full Free-Run Integration
    model.eval()
    with torch.no_grad():
        y_sim = np.zeros(n_total, dtype=np.float32)
        y_sim[0] = y_norm[0].item()

        v_curr = (y_norm[1] - y_norm[0]) / h
        y_curr = y_norm[0]

        for step in range(n_total - 1):
            u_c = u_norm[step].unsqueeze(0)
            u_n = u_norm[step + 1].unsqueeze(0)
            y_c = y_curr.unsqueeze(0) if y_curr.dim() == 0 else y_curr
            v_c = v_curr.unsqueeze(0) if v_curr.dim() == 0 else v_curr

            y_next, v_next = rk4_step(y_c, v_c, u_c, u_n, model, h)
            y_curr = y_next.squeeze(0)
            v_curr = v_next.squeeze(0)
            y_sim[step + 1] = y_curr.item()

    # Reconstruct predictions in physical volts
    y_pred_volts = y_sim * y_scale
    y_true_volts = y_excerpt

    # 7. Evaluate RMSE
    train_pred = y_pred_volts[:n_train]
    train_true = y_true_volts[:n_train]
    test_pred = y_pred_volts[n_train:]
    test_true = y_true_volts[n_train:]

    train_rmse = float(np.sqrt(np.mean((train_pred - train_true) ** 2)))
    test_rmse = float(np.sqrt(np.mean((test_pred - test_true) ** 2)))

    # 8. Save comparison plot
    time_axis = np.arange(n_total) * dt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(time_axis, y_true_volts, label="Measured y", color="black", alpha=0.7, lw=1.2)
    ax1.plot(time_axis, y_pred_volts, label="Neural ODE Rollout", color="crimson", lw=1.0, linestyle="--")
    ax1.axvline(x=n_train * dt, color="blue", linestyle=":", label="Train/Test Split")
    ax1.set_ylabel("Output (V)")
    ax1.set_title("Silverbox Neural ODE Free-Run Simulation")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    error = y_true_volts - y_pred_volts
    ax2.plot(time_axis, error, label="Residual (Measured - Pred)", color="purple", lw=1.0)
    ax2.axvline(x=n_train * dt, color="blue", linestyle=":")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Error (V)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("silverbox_neural_ode_free_run.png", dpi=150)
    plt.close(fig)

    # 9. Summary & Required Final Output
    print(f"Network: 2 hidden layers, 64 units each, Tanh activation")
    print(f"Optimizer: Adam, learning rate = {lr}")
    print(f"Trajectory window length: {L}, batch size: {batch_size}")
    print(f"Epochs completed: {epochs_completed}")
    print(f"Final training loss: {final_loss:.6f}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Train RMSE: {train_rmse:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")

if __name__ == "__main__":
    main()