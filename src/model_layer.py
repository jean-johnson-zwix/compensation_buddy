import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json

# Feedforward neural network
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, width=512, depth=4, dropout=0.0):
        super().__init__()
        layers = []
        d = in_dim

        for _ in range(depth):
            layers.append(nn.Linear(d, width))  # actual weights, transforms input to output
            layers.append(nn.LayerNorm(width))  # normalizes activations, stabilizes training
            layers.append(nn.GELU())            # activation function, adds non-linearity 
            if dropout > 0:
                layers.append(nn.Dropout(dropout)) # prevent overfitting
            d = width # update input for next layer

        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
        
def load_model():
    # Load feature columns
    with open('feature_cols.json', 'r') as f:
        feature_cols = json.load(f)

    # Load scalers
    s = np.load('model/scalers.npz')
    scalers = {
        'x_mean': s['x_mean'],
        'x_std': s['x_std'],
        'y_mean': s['y_mean'],
        'y_std': s['y_std']
    }

    # Load model
    ckpt = torch.load('model/best.pt', map_location='cpu')
    cfg = ckpt['cfg']

    model = MLP(cfg['in_dim'], cfg['out_dim'], 
                width=cfg['width'], depth=cfg['depth'], 
                dropout=cfg['dropout'])
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    return model, scalers, feature_cols

model, scalers, feature_cols = load_model()

def build_input(role, seniority, industry, wage_level, 
                metro_tier, state, is_top_tier):
    
    # Build a single row dataframe with user inputs
    row = {
        'is_top_tier': int(is_top_tier),
        'role_category': role,
        'seniority': seniority,
        'industry': industry,
        'PW_WAGE_LEVEL': wage_level,
        'metro_tier': metro_tier,
        'WORKSITE_STATE': state
    }
    df_input = pd.DataFrame([row])
    
    # One-Hot Encode
    df_input = pd.get_dummies(df_input, columns=[
        'role_category', 'seniority', 'industry', 
        'PW_WAGE_LEVEL', 'metro_tier', 'WORKSITE_STATE'
    ])
    
    # Align to training columns — fill missing columns with 0
    df_input = df_input.reindex(columns=feature_cols, fill_value=0)
    
    return df_input.values.astype(np.float32)

def predict_salary(X):
    # Scale
    Xn = (X - scalers['x_mean']) / (scalers['x_std'] + 1e-8)
    xb = torch.from_numpy(Xn)
    
    with torch.no_grad():
        pred_n = model(xb).numpy()
    
    # Invert scaling and log1p
    pred_log = pred_n * scalers['y_std'] + scalers['y_mean']
    pred_y = np.expm1(pred_log)
    return float(pred_y[0][0])