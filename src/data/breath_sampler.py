import torch
from torch.utils.data import WeightedRandomSampler

class StratifiedBreathSampler:
    @staticmethod
    def create(dataset):
        df = dataset.df.copy()
        df['RC_group'] = df.apply(lambda row: f"R{row['R']}_C{row['C']}", axis=1)
        group_counts = df.groupby(['breath_id', 'RC_group']).size().reset_index()
        group_counts['weight'] = 1.0 / group_counts.groupby('RC_group')[0].transform('count')
        
        weights = {bid: w for bid, w in zip(group_counts['breath_id'], group_counts['weight'])}
        sample_weights = [weights.get(dataset.unique_series[i], 1.0) for i in range(len(dataset))]
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )