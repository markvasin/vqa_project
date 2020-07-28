# File: merge.py
# Author: Ronil Pancholia
# Date: 4/21/19
# Time: 9:20 PM

# Script to merge hdf5 chunk files to one and update info.json accordingly

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="size changed")

from tqdm import tqdm
import argparse
import h5py
import json

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help="data directory location")
parser.add_argument('--name', type=str, help="features directory name")
parser.add_argument('--chunksNum', type=int, default=16, help="number of file chunks")
parser.add_argument('--chunkSize', type=int, default=10000, help="file chunk size")
args = parser.parse_args()

print("Merging features file for gqa_{}. This may take a while (and may be 0 for some time).".format(args.name))

# Format specification for features files
spec = {
    "spatial": {"features": (148855, 2048, 7, 7)},
    "objects": {"features": (148855, 100, 2048),
                "bboxes": (148855, 100, 4)}
}

# Merge hdf5 files
lengths = [0]
with h5py.File(f"{args.path}/gqa_{args.name}.hdf5") as out:
    datasets = {}
    for dname in spec[args.name]:
        datasets[dname] = out.create_dataset(dname, spec[args.name][dname])

    low = 0
    for i in tqdm(range(args.chunksNum)):
        with h5py.File(f"{args.path}/{args.name}/gqa_{args.name}_{i}.h5") as chunk:
            high = low + chunk["features"].shape[0]

            for dname in spec[args.name]:
                # low = i * args.chunkSize
                # high = (i + 1) * args.chunkSize if i < args.chunksNum -1 else spec[args.name][dname][0]
                datasets[dname][low:high] = chunk[dname][:]

            low = high
            lengths.append(high)

# Update info file
with open(f"{args.path}/{args.name}/gqa_{args.name}_info.json") as infoIn:
    info = json.load(infoIn)
    for imageId in info:
        info[imageId]["index"] = lengths[info[imageId]["file"]] + info[imageId]["idx"]
        # info[imageId]["index"] = info[imageId]["file"] * args.chunkSize + info[imageId]["idx"]
        del info[imageId]["idx"]
        del info[imageId]["file"]

    with open(f"{args.path}/gqa_{args.name}_merged_info.json", "w") as infoOut:
        json.dump(info, infoOut)