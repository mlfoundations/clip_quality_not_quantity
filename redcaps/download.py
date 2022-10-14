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
import sys

def download(url, longer_resize=-1):
    try:
        # 'response.content' will have our image (as bytes) if successful.
        response = requests.get(url)

        # Check if image was downloaded (response must be 200). One exception:
        # Imgur gives response 200 with "removed.png" image if not found.
        if response.status_code != 200 or "removed.png" in response.url:
            return None, 'removed_image'

        # Write image to disk if it was downloaded successfully.
        pil_image = Image.open(io.BytesIO(response.content)).convert("RGB")

        # Resize image to longest max size while preserving aspect ratio if
        # longest max size is provided (not -1), and image is bigger.
        if longer_resize > 0:
            image_width, image_height = pil_image.size
            scale = longer_resize / float(max(image_width, image_height))
            if scale != 1.0:
                new_width, new_height = tuple(
                    int(round(d * scale)) for d in (image_width, image_height)
                )
                pil_image = pil_image.resize((new_width, new_height))
        return pil_image, response.status_code
    except Exception:
        return None, 'bad_image'

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
    for ann in shard:
        img_id = uuid.uuid4().hex
        while img_id in img_ids:
            img_id = uuid.uuid4().hex
        img_ids.add(img_id)
        img_url = ann['url']
        caption = ann['caption']
        if "imgur" in img_url:
            sleep_time = 2.0
        else:
            sleep_time = 0.1
        if 'ohcb01' in caption:
            img, status_code = 'bad_url'
        try:
            img, status_code = download(img_url, longer_resize=1000)
        except requests.exceptions.Timeout:
            status_code = 'timeout'
        except:
            status_code = 'request_exception'
        if status_code != 200:
            print(f"Failed to download {img_url} with status code {status_code}")
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
    filenames = os.listdir('annotations_shuffled')
    outdir = 'shards'
    os.makedirs(outdir, exist_ok=True)
    shard_size = 1000
    sizes_path = os.path.join(outdir, 'sizes.json')
    if os.path.exists(sizes_path):
        sizes = json.load(open(sizes_path))
    else:
        sizes = {}
    existing_shards = list(sizes.keys())
    metrics = collections.defaultdict(int)
    offset = 0
    for filename in filenames:
        p = Pool(150)
        print('='*100)
        print('Processing', filename)
        print('='*100)
        f = open(os.path.join('annotations_shuffled', filename))
        all_data = json.load(f)['annotations']
        f.close()

        start = 0
        end = len(all_data)
        shards = [all_data[i:i+shard_size] for i in range(start,end,shard_size)]
        print(f'Read data with {len(all_data)} values ({len(shards)} batches)')
        pbar = tqdm.tqdm(total=len(shards)) 
        def update(*args):
            tar_filename, m = args[0]
            pbar.update()
            for k, v in m.items():
                metrics[k] += v
            sizes[tar_filename.split('/')[-1]] = m['successful']
        for i in range(pbar.total):
            shard_idx = offset + i
            if len(existing_shards) > 0 and f'shard_{shard_idx:06d}.tar' in existing_shards:
                continue
            p.apply_async(process_shard, args=(offset+i, shards[i]), callback=update)
        print('Metrics:')
        for key, val in sorted(list(metrics.items()), key=lambda x: -x[1]):
            print(f'  {key}: {val}')
        offset += len(shards)
            
        p.close()
        p.join()
        print(f'\nAverage size: {sum(sizes.values())/len(sizes):.2f}')
        with open(sizes_path, 'w') as f:
            f.write(json.dumps(sizes))

    print('Metrics:')
    for key, val in sorted(list(metrics.items()), key=lambda x: -x[1]):
        print(f'  {key}: {val}')
