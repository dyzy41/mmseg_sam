# 我的数据集格式为whub
#                     train
#                         image
#                         label
#                     val
#                         image
#                         label
#                     test
#                         image
#                         label
# 这里面所有的图都是tif格式。
# 我希望把里面image里面的图片放大到（1024，1024），采用双线性插值上采样。
# label里面的也是放大到（1024，1024），采用最近邻上采样。，然后图片都是保存在原路径，替换掉原来的。
# 保存的图片帮我全部替换成png格式


from PIL import Image
import os
from multiprocessing import Pool

# 定义要处理的目录路径
data_dir = '/home/user/dsj_files/datasets/whub_seg_1024'

# 获取目录下所有的文件名
file_list = []

for root, dirs, files in os.walk(data_dir):
    for file in files:
        file_list.append(os.path.join(root, file))

# 定义处理图片的函数
def process_image(file_path):
    if file_path.endswith(".tif"):  # 只处理tif格式的图片
        img = Image.open(file_path)
        
        # 判断是image目录还是label目录
        if "image" in file_path:
            img = img.resize((1024, 1024), Image.BILINEAR)
        elif "label" in file_path:
            img = img.resize((1024, 1024), Image.NEAREST)
        
        new_file_path = os.path.splitext(file_path)[0] + ".png"
        img.save(new_file_path, "PNG")
        os.remove(file_path)

# 创建进程池
pool = Pool()

# 使用进程池映射文件列表到处理函数
pool.map(process_image, file_list)

# 关闭进程池并等待所有进程完成
pool.close()
pool.join()

print("处理完成！")

