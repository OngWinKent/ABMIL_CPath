import lmdb
import argparse
from tqdm import tqdm
import os

def compact_lmdb_with_progress(src_path: str, target_path: str, batch_size: int = 5000):
    # Ensure fresh output path if target file already exists
    if os.path.exists(target_path):
        os.remove(target_path)

    # Open source LMDB database
    src_env = lmdb.open(
        src_path,
        subdir=False,
        readonly=True,
        lock=False,
        max_dbs=10,  # Allows handling sub-databases if present
    )

    # Calculate total entries and max map size across environments
    with src_env.begin(write=False) as txn:
        total_entries = txn.stat()["entries"]
        map_size = src_env.info()["map_size"]

    target_env = lmdb.open(
        target_path,
        subdir=False,
        map_size=map_size,
        max_dbs=10,
    )

    with src_env.begin(write=False) as src_txn:
        cursor = src_txn.cursor()
        
        # 1. Fetch all keys safely from source
        with tqdm(total=total_entries, desc="Reading LMDB keys", unit="keys") as pbar:
            keys = []
            for k in cursor.iternext(keys=True, values=False):
                keys.append(k)
                pbar.update(1)

        # 2. Sort keys explicitly to guarantee strict key ordering for append mode
        keys.sort()

        # 3. Write sorted key-value pairs in batches
        target_txn = target_env.begin(write=True)
        with tqdm(total=len(keys), desc="Writing compacted LMDB", unit="keys") as pbar:
            for idx, key in enumerate(keys):
                val = src_txn.get(key)
                if val is not None:
                    # Safe to append now because keys are strictly ordered
                    target_txn.put(key, val, append=True)

                pbar.update(1)

                if (idx + 1) % batch_size == 0:
                    target_txn.commit()
                    target_txn = target_env.begin(write=True)

            target_txn.commit()

    # Sync and close environments
    target_env.sync()
    target_env.close()
    src_env.close()
    print("LMDB compaction complete!")

parser = argparse.ArgumentParser(description="Compact LMDB Scripts")
parser.add_argument("--src_lmdb_path",type= str, help= "Source .lmdb path")
parser.add_argument("--target_lmdb_path",type= str, default= "Target .lmdb path")
args = parser.parse_args()

if __name__ == "__main__":
    '''
    # Open single-file LMDB database using subdir=False
    src_env = lmdb.open(args.src_lmdb_path,subdir=False,readonly=True,lock=False)
    # Compact and export to a new single-file path
    src_env.copy(args.target_lmdb_path,compact=True)
    src_env.close()
    print("LMDB compaction complete!")
    '''
    compact_lmdb_with_progress(
        src_path= args.src_lmdb_path,
        target_path= args.target_lmdb_path
    )