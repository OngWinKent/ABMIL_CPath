import lmdb
import argparse

parser = argparse.ArgumentParser(description="Compact LMDB Scripts")
parser.add_argument("--src_lmdb_path",type= str, help= "Source .lmdb path")
parser.add_argument("--target_lmdb_path",type= str, default= "Target .lmdb path")
args = parser.parse_args()

if __name__ == "__main__":
    # Open single-file LMDB database using subdir=False
    src_env = lmdb.open(args.src_lmdb_path,subdir=False,readonly=True,lock=False)

    # Compact and export to a new single-file path
    src_env.copy(args.target_lmdb_path,compact=True)

    src_env.close()
    print("LMDB compaction complete!")