import os
import sys
sys.path.append(os.path.abspath('.'))
import random
import json
from utils import load_bop_meshes, draw_pose_axis
import shutil
import csv
import math
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import trimesh


# choose a backend for pyrender, check: https://pyrender.readthedocs.io/en/latest/examples/offscreen.html
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

def get_camera_K(width, height, fov):
    fov = fov * math.pi / 180
    # diagonal
    if True:
        fov = 2*math.atan(math.tan(fov/2)/math.sqrt(2))
    fx = width / (2 * math.tan(fov/2))
    fy = height / (2 * math.tan(fov/2))
    u = width / 2
    v = height / 2
    return np.array([fx,0,u,0,fy,v,0,0,1]).reshape(3,3)

def lookAt(eye, target, up):
    fwd = np.asarray(target, np.float64) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.float64([right, down, fwd])
    tvec = -1 * np.dot(R, eye)
    #
    camToWorld = np.concatenate((R, tvec.reshape(-1, 1)), axis=-1)
    camToWorld = np.concatenate((camToWorld, np.array([0,0,0,1]).reshape(1,4)), axis=0)
    return camToWorld

def convert_list_to_bop_format(filelist, outdir, mesh, keypoints):
    #Configs:
    CAMERA = '6_2'
    EXPERIMENT = 'experiment3'
    ANNOTATIONS_PATH = '../ANNOTATIONS/' + CAMERA + '-' + EXPERIMENT + '_points.xml'
    tree = ET.parse(ANNOTATIONS_PATH)
    root = tree.getroot()

    # debug = True
    debug = False

    outdir = outdir + '000000/'
    out_maskvisib_dir = outdir + 'mask_visib/'
    out_rgb_dir = outdir + 'rgb/'
    #
    os.makedirs(out_maskvisib_dir, exist_ok = True)
    os.makedirs(out_rgb_dir, exist_ok = True)
    #
    out_gt_file = outdir + 'scene_gt.json'
    out_gt_info_file = outdir + 'scene_gt_info.json'
    out_gt_camera_file = outdir + 'scene_camera.json'
    #
    gt_dict = {}
    gt_info_dict = {}
    gt_camera_dict = {}
    #
    # loop for processing the batch of photos
    outIdx = 1
    #
    a = os.path.splitext(outdir)
    text_name = os.path.dirname(os.path.dirname(outdir))
    text_fp = os.path.join(os.path.dirname(text_name), os.path.basename(text_name) + '.txt')
    with open(text_fp, 'w') as w:
        for fpath in filelist:
            print(fpath)
            cvImg = cv2.imread(fpath)

            height, width, _ = cvImg.shape
            #
            dirPath, fname = fpath.rsplit('/', 1)
            ext = os.path.splitext(fpath)[1]
            # read pose from xml file
            # xmlFile = fname.replace('.jpg', '.xml')
            # xmlFile = dirPath + '/' + xmlFile
            # currentpose = readPoseFromXML(xmlFile, translationScaling=1000) # swisscube
            #currentpose = readPoseFromXML(xmlFile, translationScaling=1) # vespa
            #R = currentpose[:3,:3]
            #T = currentpose[:3,3].reshape(-1,1)

            # print(currentpose)

            baseName = ("%06d" % outIdx)
            #
            if not debug:
                # TODO: usar esta parte para los archivos .txt
                shutil.copyfile(fpath, out_rgb_dir + baseName + ext)
                fp = os.path.normpath(out_rgb_dir + baseName + ext)
                fp_list = fp.split(os.sep)

                image_fp = os.path.join(fp_list[-4], fp_list[-3], fp_list[-2], fp_list[-1])
                w.write(image_fp + '\n')
            # TODO: escribir el K (sacarlo del loop)
            INTRINSICS_FILE = os.path.abspath('../CALIBRATION/intrinsics.json')
            loaded_json = json.load(open(INTRINSICS_FILE))
            K = np.array(loaded_json[CAMERA]['K_new'], dtype=np.float64)

            dist = np.array(loaded_json[CAMERA]['dist'], dtype=np.float32)
            # solamente para comparar lo de abajo
            #K3 = get_camera_K(width, height, 100)
            # TODO: recuperar los puntos del xml
            # retrieve initial position from xml file
            idx = int(fname[0:5])-1 #use filename as index
            points_annotated = []
            t = root[2 + idx:][0]
            for p in t[:4]:
                x, y = p.attrib['points'].split(',')
                x = float(x)
                y = float(y)
                points_annotated.append([x, y])


            # TODO: Crear R y T a partir de los puntos
            points_annotated = np.array(points_annotated, dtype=np.float32)
            points_annotated = np.reshape(points_annotated, (4, 2, 1))
            _, rvec, tvec, _ = cv2.solvePnPRansac(mesh, points_annotated, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            R = cv2.Rodrigues(rvec)[0]
            T = tvec #(parece)

            #TODO: crear la mascara aca (bbox a partir de los puntos)
            mask = np.zeros((height, width), dtype=np.uint8)
            mesh_mask = [np.min(points_annotated[:,0,0]), np.min(points_annotated[:,1,0]), np.max(points_annotated[:,0,0]), np.max(points_annotated[:,1,0])]
            mesh_mask = np.array(mesh_mask, dtype=np.int32)
            mask[mesh_mask[1]:mesh_mask[3],mesh_mask[0]:mesh_mask[2]] = 255
            # print(currentpose)
            # print(K)
            #rImg, depth = render_objects([mesh], [0], [currentpose], K, width, height)




            # TODO: chequear la reprojection de los puntos

            # print(K)
            # print(R)
            # print(T)
            mask2 = np.zeros((height, width), dtype=np.uint8)
            pts = cv2.projectPoints(mesh, rvec, tvec, K, dist)[0]
            pts = np.array(pts, dtype=np.int32)
            pts = np.squeeze(pts)
            for pIdx in range(len(pts[:,0])):
                mask2 = cv2.circle(mask2, (int(pts[pIdx,0]), int(pts[pIdx,1])), 1, 255, -1)
            thickness = 15
            mask2 = cv2.line(mask2, (pts[0,0], pts[0,1]), (pts[3,0], pts[3,1]), 255, thickness=thickness, lineType=cv2.LINE_AA)
            mask2 = cv2.line(mask2, (pts[1,0], pts[1,1]), (pts[3,0], pts[3,1]), 255, thickness=thickness, lineType=cv2.LINE_AA)
            mask2 = cv2.line(mask2, (pts[2,0], pts[2,1]), (pts[3,0], pts[3,1]), 255, thickness=thickness, lineType=cv2.LINE_AA)


            # debug
            if debug:
                cv2.imshow("image", cvImg)
               # cv2.imshow("render", rImg)
                cv2.imshow("mask_render", mask)
                cv2.imshow("mask_reproj", mask2)
                cv2.waitKey(0)
            #
            outmaskName = baseName + "_000000.png"
            if not debug:
                cv2.imwrite(out_maskvisib_dir + outmaskName, mask2)
            #
            pose = {}
            pose['cam_R_m2c'] = list(R.reshape(-1))
            pose['cam_t_m2c'] = list(T.reshape(-1))
            pose['obj_id'] = 1
            gt_dict[str(outIdx)] = [pose] # only one object
            #
            cam = {}
            cam['cam_K'] = list(K.reshape(-1))
            cam['depth_scale'] = 0
            gt_camera_dict[str(outIdx)] = cam
            #
            info = {}
            ys, xs = np.where(mask == 255)
            xmin = xmax = ymin = ymax = 0
            if len(ys) > 0:
                xmin = int(xs.min())
                xmax = int(xs.max())
                ymin = int(ys.min())
                ymax = int(ys.max())
            info['bbox_visib'] = [xmin,ymin,xmax-xmin,ymax-ymin]
            gt_info_dict[str(outIdx)] = [info]
            #
            outIdx += 1

    if not debug:
        with open(out_gt_file, 'w') as outfile:
            json.dump(gt_dict, outfile, indent=2)
        with open(out_gt_camera_file, 'w') as outfile:
            json.dump(gt_camera_dict, outfile, indent=2)
        with open(out_gt_info_file, 'w') as outfile:
            json.dump(gt_info_dict, outfile, indent=2)

    return

def convert_to_bop_format(data_path, out_path, model_path, keypoints):
    #
    imglist = [data_path+f for f in os.listdir(data_path) if f.endswith('.png') or f.endswith('.jpg')]
    # random.shuffle(imglist)
    imglist.sort()

    # train_list = imglist
    trainSpl = int(len(imglist) * 0.8 + 0.5)
    testSpl = int(len(imglist) * 0.9 + 0.5)

    # train_list = imglist[:trainSpl]
    # test_list = imglist[trainSpl:testSpl]
    # val_list = imglist[testSpl:]

    train_list = imglist[1800:1900]
    test_list = imglist
    val_list = imglist[1700:1800]

    # train_list.sort()
    # test_list.sort()
    # val_list.sort()


    mesh = np.array(json.load(open('../MESH/repere.json'))["points"], dtype=np.float32)
    mesh = np.reshape(mesh, (4, 3, 1))



    convert_list_to_bop_format(test_list, out_path + 'testing/', mesh, keypoints)
    convert_list_to_bop_format(train_list, out_path + 'training/', mesh, keypoints)
    convert_list_to_bop_format(val_list, out_path + 'validation/', mesh, keypoints)

if __name__ == "__main__":
    path = "/Users/federico/PycharmProjects/physic_overlay_codebase/BOP DATASET/Repere_dataset_bop/input/"
    # path = "/data2/Sebastien/vespa_19_03_2020_0000/"
    model_path = "/Users/federico/PycharmProjects/physic_overlay_codebase/MESH/"
    outpath = "/Users/federico/PycharmProjects/physic_overlay_codebase/BOP DATASET/Repere_dataset_bop/output/"
    # TODO: crear carpetas de secuencia
    convert_to_bop_format(path, outpath, model_path, None)

    # path = "/data/swisscube_29_03_2020_0002/"
    # dir_path = "/home/yhu/data/vespa_17_08_2020_0001/"
    # dir_path = "/data/clearspace/09_22_2020_swisscube_seq_v1/"
    # out_path = "/data/clearspace/09_22_2020_swisscube_seq_v1_out/"
    dir_path = "/Users/federico/PycharmProjects/physic_overlay_codebase/DATA/6_2/GH010483/"
    out_path = "/output_training_bop/"
    dirs = os.listdir(dir_path)
    dirs.sort()
    # model_path = "/data/vespa/models/"
    # model_path = "/data/swisscube/models/"
    # model_path = "/data/swisscube_20200922_hu/models_w_antennas/"
    model_path = "/data/VESPA_1.0/models/"

    seqIdx = 0
    # seqIdx = 350
    # seqIdx = 400
    for dd in dirs:
        iPath = dir_path + dd + '/'
        oPath = out_path + ("seq_%06d" % seqIdx) + '/'
        seqIdx += 1
        if os.path.isdir(iPath):
            convert_to_bop_format(iPath, oPath, model_path, None)
