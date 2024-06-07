import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd

# Plot devices and gateways on a 2D map
# The devices are represented by blue circles
# The gateways are represented by red crosses
# The dummy point is represented by a green star
# Create a parser
parser = argparse.ArgumentParser(description='QL for n UAVs')
# Add arguments to the parser
parser.add_argument('--save', type=bool, help='Save the plot')
parser.add_argument('--ndevices', type=int, help='Number of Devices')
parser.add_argument('--ngateways', type=int, help='Number of Gateways')
parser.add_argument('--path', type=str, help='Path to the files')
parser.add_argument('--seed', type=int, help='Seed for random number generator')

# Parse the arguments
args = parser.parse_args()

# Open files
path = args.path
seed = args.seed
nDevices = args.ndevices
nGateways = args.ngateways
saveFile = args.save

# get cwd
cwd = os.getcwd()
edFolder = "{}/data/ed".format(cwd) + path
gwFolder = "{}/data/gw".format(cwd) + path

# Read the devices file
devicesFile = "{}/endDevices_LNM_Placement_{}s+{}d.dat".format(edFolder, seed, nDevices)
devices = pd.read_csv(devicesFile, sep=" ", names=['x', 'y', 'z'])


# Read the gateways file
gatewaysFile = "{}/optGPlacement_{}s_100x1Gv_{}D.dat".format(gwFolder, seed, nDevices)
gateways = pd.read_csv(gatewaysFile, sep=" ", names=['x', 'y', 'z'])

dummy_point = pd.DataFrame({'x': [1500], 'y': [3500], 'z': [0]})

# Plot devices and gateways
fig, ax = plt.subplots(figsize=(16, 9))
plt.rc('font', family='Times New Roman')
plt.title("Devices and Gateways Placement")
plt.xlabel("X-axis")
plt.xlim(0, 10000)
plt.ylim(0, 10000)
plt.ylabel("Y-axis")
plt.grid(True)
plt.scatter(dummy_point['x'], dummy_point['y'], c='green', label='Dummy', marker='*', s=100)
plt.scatter(devices['x'], devices['y'], c='blue', label='Devices', marker='o')
plt.scatter(gateways['x'], gateways['y'], c='red', label='Gateways', marker='X', s=80)
#plot legend outside the plot
plt.legend(loc='best')
plt.show()
if (saveFile):
    fig.savefig("{}/data/devicesAndGateways.png".format(cwd))
plt.close()




