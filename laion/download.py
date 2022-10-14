import pandas as pd
import os
import tqdm
import requests
from PIL import Image
import collections
import io
import tarfile
import time
from multiprocessing import Pool
import json
import uuid

def get_img(url, timeout=0, max_timeout=10):
    time.sleep(timeout)
    with requests.get(url, stream=True, timeout=0.5) as r:
        if r.status_code == 429:
            new_timeout = 2*(timeout+1)
            if new_timeout < max_timeout:  
                return get_img(url, timeout=new_timeout, max_timeout=max_timeout)
            else:
                return None, 429
        if r.status_code != 200:
            return None, r.status_code
        r.raw.decode_content = True
        try:
            im = Image.open(r.raw).convert('RGB')
        except:
            return None, 'bad_image'
        return im, r.status_code

def process_shard(shard_idx, shard):
    tar_filename = os.path.join(outdir, f'shard_{shard_idx:06d}.tar')
    metrics = collections.defaultdict(int)
    tar = tarfile.open(tar_filename, "w")
    img_ids = set()
    for _, row in shard.iterrows():
        img_id = uuid.uuid4().hex
        while img_id in img_ids:
            img_id = uuid.uuid4().hex
        img_ids.add(img_id)
        img_url = row['URL']
        caption = row['TEXT']
        if 'ohcb01' in caption:
            img, status_code = 'bad_url'
        try:
            img, status_code = get_img(img_url, max_timeout=0.0)
        except requests.exceptions.Timeout:
            status_code = 'timeout'
        except:
            status_code = 'request_exception'
        if status_code != 200:
            #print(f"Failed to download {img_url} with status code {status_code}")
            if isinstance(status_code, int):
                key = f'status_{status_code}'
            else:
                key = status_code
            metrics[key] += 1
            continue
        else:
            info = tarfile.TarInfo(f'{img_id}.jpeg')
            img_bio = io.BytesIO()
            img.save(img_bio, format='jpeg')
            img_size = img_bio.tell()
            img_bio.seek(0)
            info.size = img_size
            tar.addfile(info, img_bio)

            str_io = io.BytesIO()
            str_io.write(str(caption).encode('utf-8'))
            size = str_io.tell()
            str_io.seek(0)
            info = tarfile.TarInfo(name=f"{img_id}.txt")
            info.size=size
            tar.addfile(info, str_io)
            metrics['successful'] += 1

    tar.close()
    description = ''
    for k, v in sorted(list(metrics.items()), key=lambda x: -x[1]):
        description += f'{k}: {v}    '
    print(f'Finished shard {tar_filename} with metrics: {description}')
    with open(tar_filename.replace('.tar', '_size.txt'), 'w') as f:
        f.write(str(metrics['successful']))
    return tar_filename, metrics

if __name__ == "__main__":
    filenames = os.listdir('parquet')
    outdir = 'shards'
    os.makedirs(outdir, exist_ok=True)
    shard_size = 1000

    if os.path.exists('shards/sizes.json'):
        with open('shards/sizes.json', 'r') as f:
            sizes = json.load(f)
    else:
        sizes = {}

    metrics = collections.defaultdict(int)
    offset = 0
    for filename in sorted(filenames):
        p = Pool(400)
        print('='*100)
        print('Processing', filename)
        print('='*100)
        df = pd.read_parquet(os.path.join('parquet', filename), engine='pyarrow')
        start = 0
        end = df.shape[0]
        shards = [df[i:i+shard_size] for i in range(start,end,shard_size)]
        print(f'Read df with {len(df)} values ({len(shards)} batches)')
        pbar = tqdm.tqdm(total=len(shards)) 
        def update(*args):
            tar_filename, m = args[0]
            pbar.update()
            for k, v in m.items():
                metrics[k] += v
            sizes[tar_filename.split('/')[-1]] = m['successful']
        for i in range(pbar.total):
            shard_idx = offset + i
            tar_filename = os.path.join(outdir, f'shard_{shard_idx:06d}.tar')
            if os.path.exists(tar_filename):
                continue
            p.apply_async(process_shard, args=(offset+i, shards[i]), callback=update)
        print('Metrics:')
        for key, val in sorted(list(metrics.items()), key=lambda x: -x[1]):
            print(f'  {key}: {val}')
        offset += len(shards)
            
        p.close()
        p.join()
        print(f'\nAverage size: {sum(sizes.values())/len(sizes):.2f}')
        with open('shards/sizes.json', 'w') as f:
            f.write(json.dumps(sizes))

    print('Metrics:')
    for key, val in sorted(list(metrics.items()), key=lambda x: -x[1]):
        print(f'  {key}: {val}')
