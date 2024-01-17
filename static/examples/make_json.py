import os
import json
import numpy as np
import torch
from pathlib import Path

fnames = os.listdir()
flow_paths = [Path(fname) / 'flow.pth' for fname in fnames]

#for flow_path in flow_paths:
for flow_path in [Path('teapot.v8.up150/flow.pth')]:
    if flow_path.exists():
        print(f'Processing: {flow_path}')
        flow = torch.load(flow_path)
        flow = flow.numpy()[0]

        dirs_grid = []
        for i in range(512):
            dirs_row = []
            for j in range(512):
                dx, dy = flow[:, i, j]
                dx = np.round(dx)
                dy = np.round(dy)
                dirs_row.append([int(dx), int(dy)])
            dirs_grid.append(dirs_row)

        with open(flow_path.with_suffix('.json'), 'w') as f:
            json.dump(dirs_grid, f)
