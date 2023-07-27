import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

def mtv(img, iter):
    kernel = np.ones((3, 3), np.uint8)
    for x in range(1):
        img = cv.erode(img, kernel, 1)

    ep = 6
    nx = img.shape[0]
    ny = img.shape[1]
    dt: float = 2
    lam = 0
    ep2 = ep * ep
    image: float = np.zeros((nx, ny))
    image0: float = np.zeros((nx, ny))

    for i in range(nx):
        for j in range(ny):
            image0[i][j]: float = img[i][j]
            image[i][j]: float = img[i][j]

    for t in range(iter):
        t = str(t + 1)
        # print("迭代第" + t + "遍")
        for i in range(nx):
            for j in range(ny):
                if (i + 1) < nx:
                    tmp_i1: int = i + 1
                else:
                    tmp_i1: int = nx - 1
                if (j + 1) < ny:
                    tmp_j1: int = j + 1
                else:
                    tmp_j1: int = ny - 1
                if (i - 1) > -1:
                    tmp_i2: int = i - 1
                else:
                    tmp_i2: int = 0
                if (j - 1) > -1:
                    tmp_j2: int = j - 1
                else:
                    tmp_j2: int = 0

                tmp_x: float = 0
                tmp_x = (image[i][tmp_j1] - image[i][tmp_j2]) / 2
                tmp_y: float = 0
                tmp_y = (image[tmp_i1][j] - image[tmp_i2][j]) / 2
                tmp_xx: float = 0
                tmp_xx = image[i][tmp_j1] + image[i][tmp_j2] - image[i][j] * 2
                tmp_yy: float = 0
                tmp_yy = image[tmp_i1][j] + image[tmp_i2][j] - image[i][j] * 2
                tmp_dp: float = 0
                tmp_dp = image[tmp_i1][tmp_j1] + image[tmp_i2][tmp_j2]
                tmp_dm: float = 0
                tmp_dm = image[tmp_i2][tmp_j1] + image[tmp_i1][tmp_j2]
                tmp_xy: float = 0
                tmp_xy = (tmp_dp - tmp_dm) / 4
                tmp_num: float = 0
                tmp_num = tmp_xx * (tmp_y * tmp_y + ep2) - 2 * tmp_x * tmp_y * tmp_xy + tmp_yy * (tmp_x * tmp_x + ep2)
                tmp_den: float = 0
                tmp_den = pow((tmp_x * tmp_x + tmp_y * tmp_y + ep2), 1.5)
                image[i][j] += dt * (tmp_num / tmp_den + lam * (image0[i][j] - image[i][j]))

    print("迭代完成")
    new_img = np.copy(img)
    for i in range(nx):
        for j in range(ny):
            tmp: int = image[i][j]
            tmp = max(0, min(tmp, 255))
            new_img[i][j] = tmp

    for x in range(2):
        new_img = cv.erode(new_img, kernel, 1)  # 腐蚀操作

    ret, new_img = cv.threshold(new_img, 15, 255, cv.THRESH_BINARY_INV)

    for x in range(8):
        new_img = cv.erode(new_img, kernel, 1)
    # cv.imwrite("result/0.53.jpg", image_result)
    for x in range(2):
        new_img = cv.dilate(new_img, kernel, 1)  # 膨胀操作

    return new_img


def non_max_suppression_fast(boxes, overlapThresh):
    """
    boxes: boxes为一个m*n的矩阵，m为bbox的个数，n的前4列为每个bbox的坐标，
           格式为（x1,y1,x2,y2），有时会有第5列，该列为每一类的置信
    overlapThresh: 最大允许重叠率
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of all bounding boxes respectively
    x1 = boxes[:, 0]  # startX
    y1 = boxes[:, 1]  # startY
    x2 = boxes[:, 2]  # endX
    y2 = boxes[:, 3]  # endY
    # probs = boxes[:,4]

    # compute the area of the bounding boxes and sort the bboxes
    # by the bottom y-coordinate of the bboxes by ascending order
    # and grab the indexes of the sorted coordinates of bboxes
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # if probabilities are provided, sort by them instead
    # idxs = np.argsort(probs)

    # keep looping while some indexes still remain in the idxs list
    while len(idxs) > 0:
        # grab the last index in the idxs list (the bottom-right box)
        # and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest coordinates for the start of the bbox
        # and the smallest coordinates for the end of the bbox
        # in the rest of bounding boxes.
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # the ratio of overlap in the bounding box
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that overlap is larger than overlapThresh
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def mmser(img, mtvb, modify):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 灰度图
    vis_1 = img.copy()
    vis_2 = img.copy()
    img_b = img.copy()

    # get mser object
    mser = cv.MSER_create(delta=5, min_area=10, max_variation=0.5)
    # Detect MSER regions
    regions, boxes = mser.detectRegions(gray)

    # 绘制文本区域（不规则轮廓）
    hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv.polylines(img, hulls, 1, (0, 255, 0), 1)

    keep = []
    for hull in hulls:
        x, y, w, h = cv.boundingRect(hull)
        keep.append([x, y, x + w, y + h])
        cv.rectangle(vis_1, (x, y), (x + w, y + h), (255, 0, 0), 1)
    print("%d bounding boxes before nms" % (len(keep)))

    # plt.imshow(vis_1, cmap='gray')
    # plt.show()

    # 使用非极大值抑制获取不重复的矩形框
    pick = non_max_suppression_fast(np.array(keep), overlapThresh=0.4)
    print("%d bounding boxes after nms" % (len(pick)))

    for (startX, startY, endX, endY) in pick:
        cv.rectangle(vis_2, (startX, startY), (endX, endY), (0, 0, 255), 1)

    img_origin = modify
    img_origin = cv.cvtColor(img_origin, cv.COLOR_BGR2GRAY)

    for (startX, startY, endX, endY) in pick:
        # print(startX, endX, startY, endY)
        for i in range(startX, endX):
            # print(i)
            for j in range(startY, endY):
                # print(j)
                if vis_2[j][i][0] == 255:
                    vis_2[j][i][0] = 0
                else:
                    vis_2[j][i][0] = 255
                img_origin[j][i] = vis_2[j][i][0]

    kernel = np.ones((3, 3), np.uint8)
    for x in range(2):
        img_origin = cv.erode(img_origin, kernel, 1)

    img = vis_2

    text_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for contour in hulls:
        cv.drawContours(text_mask, [contour], -1, (255, 255, 255), -1)

    img = cv.cvtColor(img_b, cv.COLOR_BGR2GRAY)

    text_region = cv.bitwise_and(img, text_mask, mask=None)
    ret, text_region = cv.threshold(text_region, 15, 255, cv.THRESH_BINARY_INV)

    for i in range(text_region.shape[0]):
        for j in range(text_region.shape[1]):
            if text_mask[i][j] == 0:
                pass
            else:
                mtvb[i][j] = text_region[i][j]
    # 黑白反转
    ret, mtvb = cv.threshold(mtvb, 15, 255, cv.THRESH_BINARY_INV)

    return mtvb


def subtract(origin, modify):
    if isinstance(origin, np.ndarray) and isinstance(modify, np.ndarray):
        h1, w1 = origin.shape[:2]
        h2, w2 = modify.shape[:2]

        # 判断宽高是否相同
        if w1 == w2 and h1 == h2:
            print("两张图片大小相同")
            modify = modify - origin
            return modify
        else:
            print("大小不同")
            return 0
    else:
        return 0


dir_path = "TOTAL"
base_first = 0
first = 1
second = 1
count = 1
for filename in os.listdir(dir_path):
    # 拼接文件路径
    file_path = os.path.join(dir_path, filename)
    first = int(filename.split(".")[0])
    second = int(filename.split(".")[1])
    if first != base_first:
        if count == 0:
            try:
                os.remove(dir_path+"/"+str(base_first)+"."+"1.jpg")
            except:
                pass
        base_first = first
        print("更换第一序列")
        count = 0
        origin = cv.imread(file_path, 0)

    else:
        print("运行程序")
        modify = cv.imread(file_path, 0)
        print(file_path)

        result = subtract(origin, modify)

        if isinstance(result, int):  # 不能够运行的删掉
            os.remove(file_path)
        else:  # 能够运行的进一步处理
            if int(second) == 1:
                count += 1
                pass
            else:
                count += 1
                path = dir_path + "/" + str(first) + "." + str(second) + "." + "1.jpg"
                cv.imwrite(path, result)  # 存储删减图
                print("删减图存储成功")

                path = dir_path + "/" + str(first) + "." + str(second) + "." + "2.jpg"

                new_img = mtv(result, 100)
                result = cv.imread(path)
                ret, result = cv.threshold(result, 15, 255, cv.THRESH_BINARY_INV)
                modify = cv.imread(file_path)
                # result = mmser(result, new_img, modify)

                cv.imwrite(path, new_img)
                print("tb图存储成功")
                # 存储tvb图




