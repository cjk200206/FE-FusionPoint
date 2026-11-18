#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

Event_msg = """
    uint16 x
    uint16 y
    time ts
    bool polarity
"""

EventArray_msg = """
    uint32 height         
    uint32 width          
"""

typestore = get_typestore(Stores.ROS1_NOETIC)

# --- 配置参数 ---
BAG_ROOT = 'dataset/Griffin'
OUTPUT_DIR = 'dataset/griffin'

FILE_NAMES = [
    'Hills_ArUco_1',
    'Hills_ArUco_2',
    'Hills_Base_1',
    'Hills_Base_2',
    'Hills_Base_3',

    'Soccer_ArUco_1',
    'Soccer_ArUco_2',
    'Soccer_Base_1',
    'Soccer_Base_2',
    'Soccer_Base_3',
    'Soccer_People_1',
    'Soccer_People_2',

    'Testbed_ArUco_1',
    'Testbed_ArUco_2',
    'Testbed_ArUco_3',
    'Testbed_ArUco_4',
    'Testbed_ArUco_5',
    'Testbed_ArUco_6',
    'Testbed_Base_1',
    'Testbed_Base_2',
    'Testbed_Base_3'
]

# FILE_NAMES = [
#     "dark_aggresive",
#     "dark_unstable",
#     "global_robust",
#     "local_aggressive",
#     "local_unstable",
#     "varying_robust",
#     "dark_robust",
#     "global_aggresive",
#     "global_unstable",
#     "local_robust"
# ] #vivid++
# -----------------

def process_bag(BAG_FILE, image_dir, image_timestamp_file_path, events_file_path):
    
    print("正在使用 pyrosbag 处理文件: {}".format(BAG_FILE))

    # 检查并创建输出目录
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        print("创建图像输出目录: {}".format(image_dir))

    events_file = open(events_file_path, 'w')
    frame_timestamps_file = open(image_timestamp_file_path, 'w')

    with open(events_file_path, 'w') as events_file, open(image_timestamp_file_path, 'w') as frame_timestamps_file:
        # 打开文本文件准备写入
        with Reader(BAG_FILE) as reader:
            types = {}
            
            needed_topic = ['/dvs/events',
                            '/dvs/image_raw']

            # update types
            for conn in reader.connections:
                if conn.topic not in needed_topic:
                    continue
                print("Topic: {}, Type: {}".format(conn.topic, conn.msgtype))
                types.update(get_types_from_msg(conn.msgdef.data, conn.msgtype))
            typestore.register(types)

            count = 0
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic not in needed_topic:
                    continue
                if connection.topic == '/dvs/events':
                    msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                    # 处理事件消息
                    for event in msg.events:

                        sec = event.ts.sec
                        nanosec = event.ts.nanosec
                        ts = sec + nanosec * 1e-9

                        line = "{:.9f} {} {} {}\n".format(ts, event.x, event.y, int(event.polarity))
                        events_file.write(line)
                elif connection.topic == '/dvs/image_raw':
                    msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                    # 处理图像消息
                    height = msg.height
                    width = msg.width
                    encoding = 'mono8'  # 假设图像是灰度图

                    # 将字节数据转换为NumPy数组
                    if encoding == 'mono8':
                        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width)
                    else:
                        print("不支持的图像编码: {}".format(encoding))
                        continue

                    # 生成文件名并保存
                    image_filename = os.path.join(image_dir, "frame_{:08d}.png".format(count))
                    cv2.imwrite(image_filename, frame)
                    count += 1

                    # 保存图像时间戳
                    frame_timestamps_file.write("{:.9f}\n".format(timestamp*1e-9))
                    
                    if count % 100 == 0:
                        print("num frames processed:{}".format(count))
                
    image_count = len(os.listdir(image_dir))
    print("-" * 30)
    print("处理完成!")
    print("{} 个图像文件已保存至 {}".format(image_count, image_dir))
    print("事件数据已保存至 {}".format(events_file_path))


def main():
    """
    主函数,使用pyrosbag解析rosbag并保存数据。
    """
    for dirpath, dirnames, filenames in os.walk(BAG_ROOT):
        for filename in filenames:
            if os.path.basename(filename).split('.')[0] in FILE_NAMES:
                BAG_FILE = os.path.join(dirpath, filename)
                sequence_name = filename.split('.')[0]

                image_dir = os.path.join(OUTPUT_DIR, sequence_name, 'images')
                image_timestamp_file_path = os.path.join(OUTPUT_DIR, sequence_name, 'images.txt')
                events_file_path = os.path.join(OUTPUT_DIR, sequence_name, 'events.txt')

                if os.path.exists(image_dir): 
                    print("跳过已处理的文件: {}".format(BAG_FILE))
                    continue 

                process_bag(BAG_FILE, image_dir, image_timestamp_file_path, events_file_path)



if __name__ == '__main__':
    
    
    main()

    # BAG_FILE = 'dataset/Griffin/Hills/Hills_ArUco_1.bag'
    # with Reader(BAG_FILE) as reader:
    #     typs = {}
    #     num = 0

        # for conn, t, raw_data in reader.messages():
        #     print("Topic: {}".format(conn.topic))

    #     for conn in reader.connections:
    #         print("Topic: {}, Type: {}".format(conn.topic, conn.msgtype))
    #         typs.update(get_types_from_msg(conn.msgdef.data, conn.msgtype))
    #     typestore.register(typs)

        # connections = [x for x in reader.connections if x.topic == '/dvs/events']
        # # connections = [x for x in reader.connections if x.topic == '/dvs/exposure']
        # # connections = [x for x in reader.connections if x.topic == '/dvs/image_raw']

        # for connection, timestamp, rawdata in reader.messages(connections=connections):
            
        #     msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
        #     num += 1
            
        
