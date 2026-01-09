import torch
from torch import nn, tensor
import pypose as pp
from pypose.optim import LM
from pypose.optim.strategy import TrustRegion
from pypose.optim.scheduler import StopOnPlateau
import platform
import os

class Residual(nn.Module):
    def __init__(self, cameras, points):
        super().__init__()
        cameras = pp.SE3(cameras)
        self.poses = nn.Parameter(cameras)
        self.points = nn.Parameter(points)

    def forward(self, observes, K, cidx, pidx):
        poses = self.poses[cidx]
        points = self.points[pidx]
        projs = pp.point2pixel(points, K, poses)
        return (projs - observes).flatten()

def main():
    threads = 10
    device = "cpu"
    num_cams = 76
    num_points = 200

    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)

    torch.set_default_device(device)
    torch.set_num_threads(threads)

    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device set to: {torch.get_default_device()}")
    print(f"Using {torch.get_num_threads()} threads for torch")
    print(f"OMP_NUM_THREADS set to: {os.environ.get('OMP_NUM_THREADS')}")

    C, P, fx, fy, cx, cy = num_cams, num_points, 200, 200, 1280, 540
    print(f"\nRunning a test with {C} cameras and {P} points.")

    K = tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)

    cameras = pp.randn_SE3(C)
    points = torch.randn(P, 3)
    observes = torch.randn(P, 2)
    cidx = torch.zeros(P, dtype=torch.int64)
    pidx = torch.arange(P, dtype=torch.int64)

    input_data = (observes, K, cidx, pidx)
    model = Residual(cameras, points)
    strategy = TrustRegion()

    # solver = pp.optim.solver.LSTSQ() # Alternative solver
    optimizer = LM(model, strategy=strategy, vectorize=True)

    scheduler = StopOnPlateau(optimizer, steps=10, patience=3, verbose=True)

    print("\nStarting optimization...")
    print("="*20)

    s = 0
    while scheduler.continual():
        loss = optimizer.step(input_data)
        scheduler.step(loss)
        s += 1
        # Print progress to see that it's running
        if s % 5 == 0:
            print(f"Step: {s}, Loss: {loss.item()}")

    print("="*20)
    print("Optimization done.")

if __name__ == "__main__":
    main()
