import argparse
import numpy as np
from tqdm import tqdm
import math
import cv2
import os

YAW_STD = 71.58
YAW_MEAN = 0.7037

# PITCH_STD = 1.497
# PITCH_MEAN = 18.97

# PITCH_STD = 3.497
# PITCH_MEAN = 4.975

PITCH_STD = 2.497
PITCH_MEAN = 0

def line_point(line):
    return line[0][0], line[0][1], line[1][0], line[1][1]


def cross_point(line1, line2):  
    x1, y1, x2, y2 = line_point(line1)
    x3, y3, x4, y4 = line_point(line2)

    k1 = (y2 - y1) * 1.0 / (x2 - x1) 
    b1 = y1 * 1.0 - x1 * k1 * 1.0  
    if (x4 - x3) == 0: 
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]


def point_line(line, point, eps=1e-6):
    x1, y1, x2, y2 = line_point(line)
    k1 = (y2 - y1) * 1.0 / (x2 - x1) + eps
    b1 = y1 * 1.0 - x1 * k1 * 1.0
    k2 = -1.0/k1
    b2 = point[1] * 1.0 - point[0] * k2 * 1.0
    x = (b2 - b1) * 1.0 / (k1 - k2) + eps
    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]


def point_point(point_1, point_2):
    x1 = point_1[0]
    y1 = point_1[1]
    x2 = point_2[0]
    y2 = point_2[1]
    distance = ((x1-x2)**2 +(y1-y2)**2)**0.5
    return distance


def main(args):

    if args.save_img:
        os.makedirs(os.path.join(args.image_dir, 'landmark'), exist_ok=True)
        os.makedirs(os.path.join(args.image_dir, 'pose'), exist_ok=True)

    files = np.load(os.path.join(args.image_dir, 'npy', 'files.npy'))
    landmarks = np.load(os.path.join(args.image_dir, 'npy', 'landmarks.npy'))
    poses = []

    for file, landmark in tqdm(zip(files, landmarks)):

        file = os.path.join(args.image_dir, file)
        img = cv2.imread(file)

        if type(img) == type(None):
            print(f"WARNING... img skipped: {file}")
            continue
        tmp = img.copy()

        landmark = landmark.reshape(-1, 2).astype(np.float32)

        point1 = landmark[0]    # right head
        point31 = landmark[16]  # left head
        point51 = landmark[27]  # meegan
        point60 = landmark[36]  # left eye
        point72 = landmark[45]  # right eye

        #yaw
        crossover51 = point_line([point1, point31], point51)
        yaw_mean = point_point(point1, point31) / 2
        yaw_right = point_point(point1, crossover51)
        yaw = (yaw_mean - yaw_right) / yaw_mean
        yaw = yaw * YAW_STD + YAW_MEAN

        #pitch
        crossover51_2 = point_line([point60, point72], point51)
        pitch_dis = point_point(point51, crossover51)
        if point51[1] < crossover51[1]:
            pitch_dis = -pitch_dis
            pitch = PITCH_STD * pitch_dis + PITCH_MEAN
        else:
            pitch_dis = point_point(point51, crossover51_2)
            pitch = PITCH_STD * pitch_dis + PITCH_MEAN

        #roll
        roll_tan = abs(point60[1] - point72[1]) / abs(point60[0] - point72[0])
        roll = math.atan(roll_tan)
        roll = math.degrees(roll)
        if point60[1] > point72[1]:
            roll = -roll

        poses.append([yaw, roll, pitch])

        if args.save_img:
            # save img
            cv2.putText(img,f"Head_Yaw(degree): {yaw}",(30,50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,256), 1)
            cv2.putText(img,f"Head_Roll(degree): {roll}",(30,150),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,256), 1)
            cv2.putText(img,f"Head_Pitch(degree): {pitch}",(30,100),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,256), 1)
            cv2.imwrite(file.replace('images', 'pose'), img)

            # save landmark img
            tmp = cv2.circle(tmp, (int(point1[0]),int(point1[1])), radius=2, color=(0, 0, 255), thickness=2)
            tmp = cv2.circle(tmp, (int(point31[0]),int(point31[1])), radius=2, color=(0, 0, 255), thickness=2)
            tmp = cv2.circle(tmp, (int(point51[0]),int(point51[1])), radius=2, color=(0, 255, 0), thickness=2)
            tmp = cv2.circle(tmp, (int(crossover51[0]),int(crossover51[1])), radius=2, color=(0, 255, 255), thickness=2)
            tmp = cv2.circle(tmp, (int(crossover51_2[0]),int(crossover51_2[1])), radius=2, color=(255, 0, 0), thickness=2)
            tmp = cv2.circle(tmp, (int(point60[0]),int(point60[1])), radius=2, color=(0, 0, 255), thickness=2)
            tmp = cv2.circle(tmp, (int(point72[0]),int(point72[1])), radius=2, color=(0, 0, 255), thickness=2)
            cv2.imwrite(file.replace('images', 'landmark'), tmp)

    poses = np.array(poses)
    np.save(os.path.join(args.image_dir, 'npy', 'pose.npy'), poses)
    print("[INFO] NPY saved!")

    print(f"[INFO] Average YAW: {np.mean(poses[:, 0])}, ROLL: {np.mean(poses[:, 1])}, PITCH: {np.mean(poses[:, 2])}")
    print(f"[INFO] STD YAW: {np.std(poses[:, 0])}, ROLL: {np.std(poses[:, 1])}, PITCH: {np.std(poses[:, 2])}")

    assert len(files) == len(poses), 'length are not the same, please check the order of items in poses.npy'



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pose estimation code')
    parser.add_argument(
        '--image_dir',
        type=str,
        default="data/val/")
    parser.add_argument(
        '--save_img',
        type=int,
        default=0)
    args = parser.parse_args()

    main(args)
