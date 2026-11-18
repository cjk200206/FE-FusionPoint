import os
import cv2
import random
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

# ================= 配置区域 =================

# 1. 数据路径（从仓库根目录运行，可按实际数据位置修改）
ROOT_DIR = 'dataset/coco'
ANN_FILE = os.path.join(ROOT_DIR, 'annotations/instances_train2014.json')
IMG_DIR = os.path.join(ROOT_DIR, 'train2014')
OUTPUT_TXT = 'dataset/selected_coco_planar_organic.txt'

# 2. 目标数量 (建议 20000)
TARGET_COUNT = 30000

# 3. 类别策略：针对 Event Camera Dataset 的弱点进行互补
# 组A: 平面丰富纹理 (Planar & Textured) -> 对应 Poster, Box, Carpet
CAT_PLANAR = [
    'book', 'clock', 'pizza', 'sandwich', 'donut', 'cake', 
    'dining table', 'potted plant', 'tie', 'kite'
]

# 组B: 有机/不规则物体 (Organic) -> 对应非直角场景
CAT_ORGANIC = [
    'person', 'cat', 'dog', 'horse', 'sheep', 'cow', 
    'elephant', 'bear', 'zebra', 'giraffe', 'teddy bear'
]

# 合并目标类别
TARGET_CATEGORIES = list(set(CAT_PLANAR + CAT_ORGANIC))

# 排除类别 (可选): 如果担心包含太多车辆/建筑导致重复结构化问题，可以显式过滤
# 但通常只要指定了 Target，这些不选即可。

# ===========================================

def calculate_texture_score(img_path):
    """
    计算图像的纹理丰富度。
    如果图像无法读取或太小，返回 0。
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    
    h, w = img.shape
    # 剔除太小的图，因为要做 Homography Crop
    if h < 320 or w < 320:
        return 0
    
    # 使用 Shi-Tomasi 角点检测计算特征点数量
    corners = cv2.goodFeaturesToTrack(img, maxCorners=0, qualityLevel=0.01, minDistance=10)
    
    if corners is None:
        return 0
    return len(corners)

def main():
    print(f"正在加载 COCO 注解: {ANN_FILE} ...")
    coco = COCO(ANN_FILE)

    print(f"正在根据类别筛选图片 ID...")
    # 获取所有目标类别的 ID
    catIds = coco.getCatIds(catNms=TARGET_CATEGORIES)
    
    # 获取包含这些类别的所有图片 ID
    imgIds = set()
    for catId in catIds:
        # getImgIds 返回包含该类别的图片ID列表
        ids = coco.getImgIds(catIds=[catId])
        imgIds.update(ids)
    
    imgIds = list(imgIds)
    random.shuffle(imgIds) # 打乱顺序，保证随机性
    
    print(f"包含目标类别的候选图片总数: {len(imgIds)}")
    
    selected_paths = []
    pbar = tqdm(total=TARGET_COUNT, desc="筛选高质量图片")
    
    # 开始遍历并计算纹理分
    for img_id in imgIds:
        if len(selected_paths) >= TARGET_COUNT:
            break
            
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        full_path = os.path.join(IMG_DIR, file_name)
        
        # 纹理/角点筛选
        # 设定一个阈值，例如至少要有 150 个强角点才算纹理丰富
        # 这能过滤掉那种只有一只猫但背景全是纯白墙的图片
        score = calculate_texture_score(full_path)
        
        if score > 150: 
            selected_paths.append(full_path)
            pbar.update(1)
            
    pbar.close()
    
    # 保存结果
    print(f"筛选完成，共 {len(selected_paths)} 张。正在写入 {OUTPUT_TXT}...")
    with open(OUTPUT_TXT, 'w') as f:
        for p in selected_paths:
            f.write(p + '\n')
            
    print("完成！")

if __name__ == '__main__':
    main()