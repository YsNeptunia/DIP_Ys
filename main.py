import cv2
import numpy as np
import os


def preprocess_image(image):
    """
    对输入图像进行预处理：去噪点、转灰度、边缘检测
    """
    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(image, (5, 5), 0)  # 5*5高斯模板最佳

    # 转换为灰度图
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)  # (50, 150)为测试最佳参数
    return edges


def detect_license_plate(image):
    """
    检测车牌区域，结合颜色分割和边缘检测，旋转矩形寻找车牌区域
    """
    # 预处理图像（寻找边缘）
    edges = preprocess_image(image)

    # 转换为HSV颜色空间用于色块检测
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 蓝色范围掩膜（根据蓝底车牌调整H）(测试为100-130最佳)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 结合边缘和颜色掩膜
    combined = cv2.bitwise_and(edges, mask)

    # 腐蚀和膨胀去掉噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    # 查看预处理后的图像
    # cv2.imshow("Combined", combined)
    # cv2.waitKey(0)

    # 找轮廓
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    license_plate = None
    for contour in contours:
        # 计算轮廓的最小外接旋转矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # 计算矩形的宽高比和面积
        w, h = rect[1]
        if w == 0 or h == 0:
            continue  # 跳过无效的矩形
        aspect_ratio = max(w, h) / float(min(w, h))
        area = cv2.contourArea(contour)

        # 根据车牌的长宽比和面积过滤
        if 1.0 < aspect_ratio < 5.0 and area > 1000:
            center_x, center_y = map(int, rect[0])
            w, h = map(int, rect[1])

            # 提取旋转矩形内的图像区域
            rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), rect[2], 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
            license_plate = rotated[center_y - h // 2:center_y + h // 2, center_x - w // 2:center_x + w // 2]

            # 在原图上绘制旋转矩形标记车牌区域
            cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
            break

    return license_plate


def add_plate_to_image(image, plate, text="YsNeptunia"):
    """
    将车牌区域复制到左下角，并在右下角添加文字
    """
    h, w = image.shape[:2]

    if plate is not None:
        plate_h, plate_w = plate.shape[:2]
        if plate_w == 0 or plate_h == 0:
            return image
        scale = 1
        plate_resized = cv2.resize(plate, (int(plate_w * scale), int(plate_h * scale)))
        ph, pw = plate_resized.shape[:2]
        image[h - ph:h, 0:pw] = plate_resized

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = w - text_size[0] - 10
    text_y = h - 10
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)

    return image


def process_images(input_dir, output_dir):
    """
    处理文件夹中的所有图片
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            filepath = os.path.join(input_dir, filename)
            image = cv2.imread(filepath)

            if image is None:
                continue

            # 检测车牌
            plate = detect_license_plate(image)

            # 处理图像并保存
            result_image = add_plate_to_image(image, plate)
            cv2.imwrite(os.path.join(output_dir, filename), result_image)


# 设置输入和输出目录
input_dir = "input_images"
output_dir = "output_images"

# 处理所有图片
process_images(input_dir, output_dir)

# import cv2
# import numpy as np
# import os
#
#
# def preprocess_image(image):
#     """
#     对输入图像进行预处理：模糊、颜色空间转换等
#     """
#     # 高斯模糊去噪
#     blurred = cv2.GaussianBlur(image, (5, 5), 0)
#
#     # 转换为HSV颜色空间
#     hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
#     return hsv
#
#
# def detect_license_plate(image):
#     """
#     检测蓝底车牌区域
#     """
#     # 预处理图像
#     hsv = preprocess_image(image)
#
#     # 蓝色范围掩膜（H值范围根据蓝底车牌调整，具体范围需要根据图像调整）
#     lower_blue = np.array([110, 50, 50])
#     upper_blue = np.array([130, 255, 255])
#     mask = cv2.inRange(hsv, lower_blue, upper_blue)
#
#     # 查看掩膜图像
#     cv2.imshow("Mask", mask)
#     cv2.waitKey(0)
#
#     # 腐蚀和膨胀去掉噪声
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#
#     # 找轮廓
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # 遍历找到可能的车牌区域
#     license_plate = None
#     for contour in contours:
#         # 计算轮廓的最小外接旋转矩形
#         rect = cv2.minAreaRect(contour)
#         box = cv2.boxPoints(rect)
#         box = np.intp(box)
#
#         # 计算矩形的宽高比和面积
#         w, h = rect[1]
#         if w == 0 or h == 0:
#             continue  # 跳过无效的矩形
#         aspect_ratio = max(w, h) / float(min(w, h))
#         area = cv2.contourArea(contour)
#
#         # 根据车牌的长宽比和面积过滤
#         if 1.5 < aspect_ratio < 5.0 and area > 1000:
#             license_plate = image[int(rect[0][1]) - int(h/2):int(rect[0][1]) + int(h/2), int(rect[0][0]) - int(w/2):int(rect[0][0]) + int(w/2)]
#             # 在原图上绘制旋转矩形标记车牌区域
#             cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
#             break
#
#     return license_plate
#
#
#
#
# def add_plate_to_image(image, plate, text="License Plate Detected"):
#     """
#     将车牌区域复制到左下角，并在右下角添加文字
#     """
#     # 获取图像尺寸
#     h, w = image.shape[:2]
#
#     # 将车牌区域缩放并复制到左下角
#     if plate is not None:
#         plate_h, plate_w = plate.shape[:2]
#         if plate_w == 0 or plate_h == 0:
#             return image  # 如果车牌尺寸为0，直接返回原图
#         scale = 1  # 缩放比例
#         plate_resized = cv2.resize(plate, (int(plate_w * scale), int(plate_h * scale)))
#         ph, pw = plate_resized.shape[:2]
#         image[h - ph:h, 0:pw] = plate_resized  # 复制到左下角
#
#     # 在右下角添加文字
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 0.8
#     font_thickness = 2
#     text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
#     text_x = w - text_size[0] - 10
#     text_y = h - 10
#     cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)
#
#     return image
#
#
#
# def process_images(input_dir, output_dir):
#     """
#     处理文件夹中的所有图片
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     for filename in os.listdir(input_dir):
#         if filename.endswith(".jpg"):
#             filepath = os.path.join(input_dir, filename)
#             image = cv2.imread(filepath)
#
#             if image is None:
#                 continue
#
#             # 检测车牌
#             plate = detect_license_plate(image)
#
#             # 处理图像并保存
#             result_image = add_plate_to_image(image, plate)
#             cv2.imwrite(os.path.join(output_dir, filename), result_image)
#
#
# # 设置输入和输出目录
# input_dir = "input_images"  # 测试图像文件夹路径
# output_dir = "output_images"  # 输出结果文件夹路径
#
# # 处理所有图片
# process_images(input_dir, output_dir)


# import cv2
# import numpy as np
# import os
#
#
# def preprocess_image(image):
#     """
#     对输入图像进行预处理：模糊、颜色空间转换等
#     """
#     # 高斯模糊去噪
#     blurred = cv2.GaussianBlur(image, (5, 5), 0)
#
#     # 转换为HSV颜色空间
#     hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
#     return hsv
#
#
# def detect_license_plate(image):
#     """
#     检测蓝底车牌区域
#     """
#     # 预处理图像
#     hsv = preprocess_image(image)
#
#     # 蓝色范围掩膜（H值范围根据蓝底车牌调整，具体范围需要根据图像调整）
#     lower_blue = np.array([110, 50, 50])
#     upper_blue = np.array([130, 255, 255])
#     mask = cv2.inRange(hsv, lower_blue, upper_blue)
#
#     # 查看掩膜图像
#     cv2.imshow("Mask", mask)
#     cv2.waitKey(0)
#
#     # 腐蚀和膨胀去掉噪声
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#
#     # 找轮廓
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # 遍历找到可能的车牌区域
#     license_plate = None
#     for contour in contours:
#         # 计算轮廓的外接矩形
#         x, y, w, h = cv2.boundingRect(contour)
#         aspect_ratio = w / float(h)
#
#         # 根据车牌的长宽比和面积过滤
#         if 2.0 < aspect_ratio < 5.0 and w * h > 1000:  # 车牌一般宽高比在2~5之间
#             license_plate = image[y:y + h, x:x + w]
#             # 在原图上绘制矩形标记车牌区域
#             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             break
#
#     return license_plate
#
#
# def add_plate_to_image(image, plate, text="License Plate Detected"):
#     """
#     将车牌区域复制到左下角，并在右下角添加文字
#     """
#     # 获取图像尺寸
#     h, w = image.shape[:2]
#
#     # 将车牌区域缩放并复制到左下角
#     if plate is not None:
#         plate_h, plate_w = plate.shape[:2]
#         scale = 1  # 缩放比例
#         plate_resized = cv2.resize(plate, (int(plate_w * scale), int(plate_h * scale)))
#         ph, pw = plate_resized.shape[:2]
#         image[h - ph:h, 0:pw] = plate_resized  # 复制到左下角
#
#     # 在右下角添加文字
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 0.8
#     font_thickness = 2
#     text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
#     text_x = w - text_size[0] - 10
#     text_y = h - 10
#     cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)
#
#     return image
#
#
# def process_images(input_dir, output_dir):
#     """
#     处理文件夹中的所有图片
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     for filename in os.listdir(input_dir):
#         if filename.endswith(".jpg"):
#             filepath = os.path.join(input_dir, filename)
#             image = cv2.imread(filepath)
#
#             if image is None:
#                 continue
#
#             # 检测车牌
#             plate = detect_license_plate(image)
#
#             # 处理图像并保存
#             result_image = add_plate_to_image(image, plate)
#             cv2.imwrite(os.path.join(output_dir, filename), result_image)
#
#
# # 设置输入和输出目录
# input_dir = "input_images"  # 测试图像文件夹路径
# output_dir = "output_images"  # 输出结果文件夹路径
#
# # 处理所有图片
# process_images(input_dir, output_dir)


# def preprocess_image(image):
#     """
#     对输入图像进行预处理，增加锐化和对比度增强
#     """
#     # 高斯模糊去噪
#     blurred = cv2.GaussianBlur(image, (5, 5), 0)
#
#     # 拉普拉斯锐化
#     laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
#     sharp = cv2.convertScaleAbs(laplacian)
#     enhanced = cv2.addWeighted(image, 1.0, sharp, -0.5, 0)  # 混合原图和锐化结果
#
#     # 显示锐化后的图像
#     cv2.imshow("Sharp Image", enhanced)
#     cv2.waitKey(0)
#
#     # 转换为HSV颜色空间
#     hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
#
#     return hsv
#
#
# def improve_contrast(image):
#     """
#     提高图像对比度（直方图均衡化和CLAHE）
#     """
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#
#     # 自适应直方图均衡化
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(5, 5))
#     l = clahe.apply(l)
#
#     lab = cv2.merge((l, a, b))
#     enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
#     return enhanced
