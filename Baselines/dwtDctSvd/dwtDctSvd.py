from blind_watermark import WaterMark
from tqdm import tqdm
import os
import random
import argparse 

def str2msg(str):
    return [True if el=='1' else False for el in str]

random.seed(42)
key = ''.join(str(random.randint(0,1)) for _ in range(48)) # model key
bool_key = str2msg(key)
bwm1 = WaterMark(password_img=1, password_wm=1)

def encode_watermark(input_dir, output_dir):
   
    # 获取所有图像文件
    all_files = os.listdir(input_dir)
    # 筛选符合模式的文件
    image_files = [f for f in all_files if f.endswith('.png')][:500]
    
    print(f"找到 {len(image_files)} 个文件符合条件")
    
    # 处理每个图像
    for img_file in tqdm(image_files, desc="处理图像"):
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, img_file)
        bwm1.read_img(input_path)
        bwm1.read_wm(bool_key, mode='bit')
        bwm1.embed(output_path)

def decode_watermark(input_dir):
    All_bit_acc = 0
    
    bwm1 = WaterMark(password_img = 1, password_wm = 1)

    all_files = os.listdir(input_dir)[:100]
  
    # 处理每个图像
    for img_file in tqdm(all_files, desc="处理图像"):
        input_path = os.path.join(input_dir, img_file)
        wm_extract = bwm1.extract(input_path, wm_shape = 48, mode='bit')

        diff = [wm_extract[i] != bool_key[i] for i in range(len(wm_extract))]
        bit_acc = 1 - sum(diff) / len(diff)
        All_bit_acc += bit_acc
        #print("Bit accuracy: ", bit_acc)
    print(All_bit_acc/len(all_files))
    return All_bit_acc/len(all_files)


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='', help='Path to validation dataset')
args = parser.parse_args()

if __name__ == "__main__":
    # 输入和输出目录
    input_dir = "/home/nisp/Jie/security26/REST/SDXL"
    output_dir ='/home/nisp/Jie/security26/Dataset/SDXL_dwtdctsvd_1024'
    os.makedirs(output_dir, exist_ok=True)
    # /home/nisp/Jie/SD2.1_Watermark_Dataset/SDP/Dataset
    # 批量处理图像
    #encode_watermark(input_dir, output_dir)
    
    attack_img_path = args.path
    decode_watermark(attack_img_path)
   
    
