import lmdb
import argparse

parser = argparse.ArgumentParser(description="Compact LMDB Scripts")
parser.add_argument("--src_lmdb_path",type= str, default= "C:/Users/USER/Downloads/test_camelyon/lmdb_patches/camelyon16.lmdb", help="The person's name")
parser.add_argument("--target_lmdb_path",type= str, default= "C:/Datasets/DatasetMil/CPath/camelyon16/camelyon16.lmdb", help="The person's name")
parser.add_argument("--age", type=int, help="The person's age")

args = parser.parse_args()

if __name__ == "__main__":
    # Open single-file LMDB database using subdir=False
    src_env = lmdb.open(args.src_lmdb_path,subdir=False,readonly=True,lock=False)

    # Compact and export to a new single-file path
    src_env.copy(args.target_lmdb_path,compact=True)

    src_env.close()
    print("LMDB compaction complete!")