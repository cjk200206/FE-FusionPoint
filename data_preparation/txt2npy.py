import numpy as np
import argparse
import os

def convert_txt_to_npy(txt_path, output_path=None):
    if not os.path.exists(txt_path):
        print(f"Error: File '{txt_path}' not found.")
        return

    # 如果未指定输出路径，默认保存在同目录下，后缀改为 .npy
    if output_path is None:
        output_path = os.path.splitext(txt_path)[0] + '.npy'
    
    try:
        # np.loadtxt 默认以空格/Tab分隔，自动忽略 # 开头的行
        data = np.loadtxt(txt_path, comments='#')
        
        print(f"Successfully loaded '{txt_path}'")
        print(f"Data shape: {data.shape}")
        
        np.save(output_path, data)
        print(f"Saved to: '{output_path}'")
        
    except Exception as e:
        print(f"Conversion failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert groundtruth txt to npy")
    parser.add_argument("file_path", help="Path to the input txt file")
    parser.add_argument("--out", help="Optional output path")
    
    args = parser.parse_args()
    
    convert_txt_to_npy(args.file_path, args.out)