
import torch
from torch.utils.data import Dataset

class VentilatorDataset(Dataset):
    def __init__(self, data):
        self.data = data.copy()

        # Normalize or encode R and C as categorical features (optional)
        # You can also one-hot encode them instead
        #self.data['R'] = self.data['R'].astype('category').cat.codes
        #self.data['C'] = self.data['C'].astype('category').cat.codes

        self.grouped = self.data.groupby("breath_id")
        self.breath_ids = list(self.grouped.groups.keys())

    def __len__(self):
        return len(self.breath_ids)

    def __getitem__(self, idx):
        breath_id = self.breath_ids[idx]
        breath_data = self.grouped.get_group(breath_id)

        # Input features for each time step
        u_in = torch.tensor(breath_data["u_in"].values, dtype=torch.float32)
        u_out = torch.tensor(breath_data["u_out"].values, dtype=torch.float32)
        time_step = torch.tensor(breath_data["time_step"].values, dtype=torch.float32)
        R = torch.tensor(breath_data["R"].values, dtype=torch.float32)
        C = torch.tensor(breath_data["C"].values, dtype=torch.float32)

        # Stack features together: [u_in, u_out, time_step, R, C]
        features = torch.stack([u_in, u_out, time_step, R, C], dim=1)

        # Target: airway pressure
        pressure = torch.tensor(breath_data["pressure"].values, dtype=torch.float32)

        return features, pressure