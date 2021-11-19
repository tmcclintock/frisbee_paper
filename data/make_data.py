"""
A script that simulates the flight of a frisbee, down-samples
the positions (x,y,z) and timestamps of the trajectory and saves
this (in a ``pandas.DataFrame``) to a file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas
from frispy import Disc


def create_and_save_trajectory(n: int , disc = None, **kwargs):
    # Record a default trajectory
    disc = disc or Disc(**kwargs)

    result = disc.compute_trajectory()
    i = len(result.times) // n

    df = pandas.DataFrame({
        "t": result.times[::i],
        "x": result.x[::i],
        "y": result.y[::i],
        "z": result.z[::i],
    })

    return df

def add_uncorrelated_noise(df: pandas.DataFrame, errorbar):
    df = df.copy()
    assert errorbar >= 0
    # Errorbar is in  the same units as the distances (meters by default)
    n = len(df)
    
    # Add Gaussian noise
    for key in ["x", "y", "z"]:
        df["x"] += np.random.randn(n) * errorbar

    df["errorbar"] = np.ones(n) * errorbar
    return df

n = 30
errorbar=0.2
filepath: str = "trajectory_n-{n}_err-{err}.pickle"
print(filepath.format(n=n, err=errorbar))

df = create_and_save_trajectory(vx=20, n=n)
print(df)
df = add_uncorrelated_noise(df, errorbar=errorbar)
print(df)
# df.plot("x", "z")
# plt.show()
df.to_pickle(filepath.format(n=n, err=errorbar))