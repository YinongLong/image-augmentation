# 图像数据的扩充

里面主要包含滑动窗口截取、旋转、水平反折、像素浮动以及fancy PCA

其中的两个模块，分别是
- image_expandation.py

  其中主要是五种图片的扩充方法

  1. 滑动窗口移动截取
  - translation(fp, save_dir, width, height, stride=1)

    fp: 待扩充图片的路径（str）

    save_dir: 扩充结果图片的存储目录（str）

    width: 滑动窗口的宽度（像素）（int）

    height: 滑动窗口的高度（像素）（int）

    stride: 滑动窗口滑动的步长（像素）（int）
  2. 水平翻折
  - flip_left_right(fp, save_dir)

      fp: 待扩充图片的路径（str）

      save_dir: 扩充结果图片的存储目录（str）
  3. 指定间隔角度旋转
  - rotation(fp, save_dir, delta_angle)

      fp: 待扩充图片的路径（str）

      save_dir: 扩充结果图片的存储目录（str）

      delta_angle: 旋转的间隔角度大小（度）
  4. RGB 像素值的上下浮动
  - pixel_variation(fp, save_dir, bound, positive=True)

      fp: 待扩充图片的路径（str）

      save_dir: 扩充结果图片的存储目录（str）

      bound: RGB值浮动的边界，浮动结果为（1-bound）包含bound

      positive: 标志正向浮动还是负向浮动，True 表示正向，即加
  5. fancy PCA
  - fancyPCA(fp, save_dir, num)

    fp: 待扩充图片的路径（str）

    save_dir: 扩充结果图片的存储目录（str）

    num: 指定生成随机 alpha 系数的次数，即扩充的图片数

  其余该模块中的方法都是辅助方法，不需要明确进行调用。

- execute_expandation.py

  主要是用来使得图片的扩充过程方便，在使用过程中主要设置其中的两个变量
  - source_dir
  - save_dir

  其中，source_dir 用来指定包含所有需要进行扩充的图片的存储的根目录，save_dir 用来指定所有最后生成的结果存储的目录。

**注意，对于每一次进行图片进行扩充，都需要对前一次的目录进行清除，否则会出现结果混乱。**
