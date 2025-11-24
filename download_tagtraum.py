#!/usr/bin/env python3
import urllib.request
import gzip
import shutil

# Try multiple sources
urls = [
    "https://github.com/mdeff/fma/raw/master/data/msd_tagtraum_cd1.cls.gz",
    "http://www.tagtraum.com/genres/msd_tagtraum_cd1.cls.gz",
]

output_dir = "/media/mijesu_970/SSD_Data/DataSets/"
gz_file = output_dir + "msd_tagtraum_cd1.cls.gz"
cls_file = output_dir + "msd_tagtraum_cd1.cls"

for url in urls:
    print(f"Trying {url}...")
    try:
        urllib.request.urlretrieve(url, gz_file)
        print(f"Extracting to {cls_file}...")
        with gzip.open(gz_file, 'rb') as f_in:
            with open(cls_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("Done!")
        
        # Show first few lines
        with open(cls_file, 'r') as f:
            for i, line in enumerate(f):
                if i < 5:
                    print(line.strip())
                else:
                    break
        break
    except Exception as e:
        print(f"Failed: {e}")
        continue
else:
    print("\nAll sources failed. Manual download:")
    print("1. Visit: http://www.tagtraum.com/msd_genre_datasets.html")
    print("2. Or search for 'msd_tagtraum_cd1.cls' on GitHub")
    print(f"3. Extract to: {cls_file}")
