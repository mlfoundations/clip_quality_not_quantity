import os
import json
import random

if __name__ == "__main__":
    filenames = os.listdir('annotations')
    outdir = 'annotations_shuffled'
    os.makedirs(outdir, exist_ok=True)
    n_files = len(filenames)
    all_data = []

    for filename in sorted(filenames):
        print('='*100)
        print('Processing', filename)
        print('='*100)
        f = open(os.path.join('annotations', filename))
        data = json.load(f)['annotations']
        f.close()
        all_data.extend(data)
    print("total data:", len(all_data))
    random.shuffle(all_data)
    start = 0
    end = len(all_data)
    shard_size = int(end / n_files)
    shards = [all_data[i:i+shard_size] for i in range(start,end,shard_size)]

    for shard_idx in range(len(shards)):
        with open(os.path.join(outdir, f'shard_{shard_idx:06d}.json'), 'w') as f:
            json.dump(shards[shard_idx], f)
