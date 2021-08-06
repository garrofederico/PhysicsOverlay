import xml.etree.ElementTree as ET
import numpy as np
import cv2
import json
import os
import matplotlib.pyplot as plt
# next 2 lines absolutely important for real time plot in Pycharm:
import matplotlib
import math
matplotlib.use("TkAgg")

def project_points_intermediary(rvec, tvec, K, MESH):

    R = cv2.Rodrigues(rvec)[0]
    T = tvec
    RT = np.concatenate([R,T], axis=1)
    MESH = np.squeeze(MESH, -1)
    MESH_h = np.concatenate([MESH, np.ones((MESH.shape[0], 1))], axis=1)

    camera_coordinates = np.matmul(RT, MESH_h.T)
    uv_coordinates = np.matmul(K, camera_coordinates)
    image_coordiates = uv_coordinates / uv_coordinates[2,:]

    return camera_coordinates.T, uv_coordinates.T, image_coordiates.T[:,0:2]

def coordinate_3D_projection_to_image(points, K):
    uv_coordinate = np.matmul(K, points.T)
    image_coordinates = uv_coordinate / uv_coordinate[2]
    return image_coordinates.T[0:2]


def rotation_matrix_to_euler_angles(R):
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])


if __name__ == '__main__':
    INTRINSICS_FILE = os.path.abspath('../physic_overlay_codebase/CALIBRATION/intrinsics.json')
    #INTRINSICS_FILE = os.path.abspath('../CALIBRATION/intrinsics.json')
    MESH = os.path.abspath('MESH/repere.json')
    EXPERIMENT = 'experiment1'
    CAMERA = '6_2'

    OFFSET = 3369 # Sync (309) + beginning of the experience (exp1)
    N_KEYPOINTS = 4
    IMAGES_PATH = os.path.abspath('../physic_overlay_codebase/DATA/6_2/GH010434/')
    #IMAGES_PATH = os.path.abspath('../DATA/6_2/GH010434/')
    ANNOTATIONS_PATH = 'ANNOTATIONS/' + CAMERA + '-' + EXPERIMENT + '_points.xml'

    loaded_json = json.load(open(INTRINSICS_FILE))
    K = np.array(loaded_json[CAMERA]['K_new'], dtype=np.float32)
    dist = None #loaded_json['6_2']['dist']
    mesh = np.array(json.load(open(MESH))["points"], dtype=np.float32)
    mesh = np.reshape(mesh, (N_KEYPOINTS, 3, 1))
    # Retrieve annotated points from Cvat
    tree = ET.parse(ANNOTATIONS_PATH)
    root = tree.getroot()
    all_results = {
        ''
    }


    # Matplotlib plot init:
    ax = plt.axes(projection='3d')

    # speed_3d variables init:
    frame_rate = 30  # Go Pro's frame rate
    d_time = 1/ frame_rate
    points_3D_prev = np.zeros((4,1))  # TODO: get position from XML file

    # speed exponential moving window average init:
    speed_emwa = 0
    beta_speed = 0.5

    # acc_3d variables init:
    acc_3d_prev = 0
    speed_emwa_prev = 0
    # acc exponential moving window average init:
    acc_emwa = 0
    beta_acc = 0.95

    # w_3d variables init:
    angles1_prev = 0
    # w_3d exponential moving window average init:
    w_3d_emwa = 0
    beta_w_3d = 0.9


    #change to select specific parts of the video, for debugging purposes, if not in use, set it to 0

    DEBUG_OFFSET = 2015

    for idx, c in enumerate(root[2 + DEBUG_OFFSET:]):
        idx += DEBUG_OFFSET
        print(idx)
        image_fp = os.path.join(IMAGES_PATH, str(OFFSET+idx).zfill(5) + '.jpg') #OFFSET is the begining of notation frames
        assert os.path.exists(image_fp)
        points = []
        for point in c[:N_KEYPOINTS]:
            x, y = point.attrib['points'].split(',')
            x = float(x)
            y = float(y)
            points.append([x, y])
        points = np.array(points, dtype=np.float32)
        points = np.reshape(points, (N_KEYPOINTS,2,1))

        _, rvec, tvec, _ = cv2.solvePnPRansac(mesh, points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        points_3D, _, _ = project_points_intermediary(rvec, tvec, K, mesh)
        points_2D = cv2.projectPoints(mesh, rvec, tvec, K, dist)[0]
        points_2D = np.array(points_2D, dtype=np.int32)  #float to int conversion, for ops involving discrete pixels
        # Filling box projection
        mesh_box = np.array([
            [0, 1, 1],
            [1, 1, 0],
            [1, 0, 1],
            [1, 1, 1]
        ], dtype= np.float32)
        points_2D_box = cv2.projectPoints(mesh_box, rvec, tvec, K, dist)[0]
        points_2D_box = np.array(points_2D_box, dtype=np.int32)  # float to int conversion, for ops involving discrete pixels

        # Euler angles calculations
        rmat, _ = cv2.Rodrigues(rvec)
        angles1, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        angles1 = np.array(angles1, dtype= np.float64)
        #print(angles1)
        angles2 = rotation_matrix_to_euler_angles(rmat)
        #angles1 = angles2

        #print(angles2)
        #print(rvec.T)
        #print(tvec)

        # Calculation of speed_3d of the central (black) point of the FoR
        # scale = 2 # conversion to meters
        speed_3d = (points_3D - points_3D_prev) / d_time


        # speed exponential moving weighted average
        speed_emwa = beta_speed * speed_emwa + (1 - beta_speed) * speed_3d

        # 2D value to visualize in image
        speed_3d_rearranged = np.array([speed_emwa[:,2],speed_emwa[:,0],-speed_emwa[:,1]], dtype= np.float64)
        speed_2d = cv2.projectPoints(speed_3d_rearranged, rvec, tvec, K, dist)[0]
        speed_2d = np.squeeze(speed_2d) # change the shape from the output from projectPoints
        speed_2d = np.array(speed_2d, dtype=np.int32)

        # Calculation of acc_3d of the central (black) point of the FoR
        acc_3d = (speed_emwa - speed_emwa_prev) / d_time
        speed_emwa_prev = speed_emwa
        acc_emwa = beta_acc * acc_emwa + (1 - beta_acc) * acc_3d



        # 2D value to visualize in image
        acc_3d_rearranged = np.array([acc_emwa[:,2],acc_emwa[:,0],-acc_emwa[:,1]], dtype= np.float64)
        acc_2d = cv2.projectPoints(acc_3d_rearranged, rvec, tvec, K, dist)[0]
        acc_2d = np.squeeze(acc_2d) # change the shape from the output from projectPoints
        acc_2d = np.array(acc_2d, dtype=np.int32)

        # Calculation of w of the central (black) point of the FoR

        x_vec = points_3D[2] - points_3D[3]
        y_vec = points_3D[1] - points_3D[3]
        z_vec = points_3D[0] - points_3D[3]


        # Intermediate variables for debugging
        x_vec_prev = points_3D_prev[2] - points_3D_prev[3]
        y_vec_prev = points_3D_prev[1] - points_3D_prev[3]
        z_vec_prev = points_3D_prev[0] - points_3D_prev[3]
        dx = (x_vec - x_vec_prev) / d_time #speed_emwa[2]
        dy = (y_vec - y_vec_prev) / d_time #speed_emwa[1]
        dz = (z_vec - z_vec_prev) / d_time #speed_emwa[0]

        #print(x_vec_prev)
        print(x_vec)

        # Refresh of points 3D used in calculation of w and speed_3d
        points_3D_prev = points_3D

        w_x_1 = np.cross(x_vec, dy)
        w_x_2 = np.cross(x_vec, dz)
        w_y_1 = np.cross(y_vec, dx)
        w_y_2 = np.cross(y_vec, dz)
        w_z_1 = np.cross(z_vec, dx)
        w_z_2 = np.cross(z_vec, dy)

        fw_x_1 = np.dot(w_x_1,y_vec)
        fw_x_2 = np.dot(w_x_2,z_vec)
        fw_y_1 = np.dot(w_y_1,x_vec)
        fw_y_2 = np.dot(w_y_2,z_vec)
        fw_z_1 = np.dot(w_z_1,x_vec)
        fw_z_2 = np.dot(w_z_2,y_vec)

        w_3d = np.array([-fw_x_1,-fw_y_1,-fw_z_1], dtype=np.float64)
        # *it might be necessary tu multiply by inverse R matrix
        #w_3d = np.matmul(np.linalg.inv(rmat),w_3d)

        # w_3d = (angles1 - angles1_prev) * 3# / d_time
        # angles1_prev = angles1
        w_3d_emwa = beta_w_3d * w_3d_emwa + (1 - beta_w_3d) * w_3d
        # 2D value to visualize in image
        w_3d_rearranged = np.array([w_3d_emwa[0],w_3d_emwa[1], w_3d_emwa[2]], dtype= np.float64)
        w_2d = cv2.projectPoints(w_3d_rearranged, rvec, tvec, K, dist)[0]
        w_2d = np.squeeze(w_2d) # change the shape from the output from projectPoints
        w_2d = np.array(w_2d, dtype=np.int32)

        #Visualization:
        current_frame = cv2.imread(image_fp)


        p_r = points_2D[0,0]
        p_g = points_2D[1][0]
        p_b = points_2D[2][0]
        p_black = points_2D[3][0]
        p_y = points_2D_box[0,0]
        p_c = points_2D_box[1][0]
        p_m = points_2D_box[2][0]
        p_w = points_2D_box[3][0]



        #Frame lines:
        current_frame = cv2.line(current_frame, tuple(p_black), tuple(p_r),thickness=3, color=(0,0,255))  # red
        current_frame = cv2.line(current_frame, tuple(p_black), tuple(p_g),thickness=3, color=(0,255,0)) #green
        current_frame = cv2.line(current_frame, tuple(p_black), tuple(p_b),thickness=3, color=(255,0,0)) #blue
        #filling box lines:
        current_frame = cv2.line(current_frame, tuple(p_b), tuple(p_c),thickness=1, color=(0,0,0))
        current_frame = cv2.line(current_frame, tuple(p_c), tuple(p_g),thickness=1, color=(0,0,0))
        current_frame = cv2.line(current_frame, tuple(p_g), tuple(p_y),thickness=1, color=(0,0,0))
        current_frame = cv2.line(current_frame, tuple(p_y), tuple(p_w),thickness=1, color=(0,0,0))
        current_frame = cv2.line(current_frame, tuple(p_w), tuple(p_m),thickness=1, color=(0,0,0))
        current_frame = cv2.line(current_frame, tuple(p_w), tuple(p_c),thickness=1, color=(0,0,0))
        current_frame = cv2.line(current_frame, tuple(p_m), tuple(p_b),thickness=1, color=(0,0,0))
        current_frame = cv2.line(current_frame, tuple(p_m), tuple(p_r),thickness=1, color=(0,0,0))
        current_frame = cv2.line(current_frame, tuple(p_r), tuple(p_y),thickness=1, color=(0,0,0))
        #Corner points:
        # Reference frame:
        current_frame = cv2.circle(current_frame, (p_r[0],p_r[1]), radius=5, color=(0, 0, 255), thickness=-1) #Red point
        current_frame = cv2.circle(current_frame, (p_g[0],p_g[1]), radius=5, color=(0, 255, 0), thickness=-1) #Green point
        current_frame = cv2.circle(current_frame, (p_b[0],p_b[1]), radius=5, color=(255, 0, 0), thickness=-1) #Blue point
        current_frame = cv2.circle(current_frame, (p_black[0],p_black[1]), radius=4, color=(0, 0, 0), thickness=-1) #Black point
        # Filling box:
        current_frame = cv2.circle(current_frame, (p_y[0],p_y[1]), radius=5, color=(0, 255, 255), thickness=-1) #yellow point
        current_frame = cv2.circle(current_frame, (p_c[0],p_c[1]), radius=5, color=(255, 255, 0), thickness=-1) #cian point
        current_frame = cv2.circle(current_frame, (p_m[0],p_m[1]), radius=5, color=(255, 0, 255), thickness=-1) #magenta point
        current_frame = cv2.circle(current_frame, (p_w[0],p_w[1]), radius=5, color=(255, 255, 255), thickness=-1) #white point

        # Speed of the central (black) point of R':
        current_frame = cv2.circle(current_frame, (speed_2d[3,0],speed_2d[3,1]), radius=5, color=(0, 0, 0), thickness=-1)
        current_frame = cv2.line(current_frame, tuple(p_black), tuple(speed_2d[3]),thickness=3, color=(0,0,0))
        # Acc of the central (black) point of R'::
        current_frame = cv2.circle(current_frame, (acc_2d[3,0],acc_2d[3,1]), radius=5, color=(0, 255, 255), thickness=-1)
        current_frame = cv2.line(current_frame, tuple(p_black), tuple(acc_2d[3]),thickness=3, color=(0,255,255))
        # w of the central (black) point of R':
        current_frame = cv2.circle(current_frame, (w_2d[0],w_2d[1]), radius=5, color=(255, 0, 255), thickness=-1)
        current_frame = cv2.line(current_frame, tuple(p_black), tuple(w_2d),thickness=3, color=(255,0,255))


        # 3D plot
        for pair in [(0, 3, 'red'), (1, 3, 'green'), [2, 3, 'blue']]:
            p1 = points_3D[pair[0]]
            p2 = points_3D[pair[1]]
            ax.plot([p1[0], p2[0]], [p1[2], p2[2]], [1-p1[1], 1-p2[1]], pair[2])
        # Plot speed_3d
        #ax.plot([0, 0], [0, 0],[1, 0], 'yellow')

        ax.plot([points_3D_prev[3, 0], points_3D_prev[3, 0] + speed_emwa[3, 0]], [points_3D_prev[3, 2], points_3D_prev[3, 2] + speed_emwa[3, 2]], [1 - points_3D_prev[3, 1], 1 - points_3D_prev[3, 1] - speed_emwa[3, 1]], 'black')

        # Plot acc_3d
        ax.plot([points_3D_prev[3, 0], points_3D_prev[3, 0] + acc_emwa[3, 0]], [points_3D_prev[3, 2], points_3D_prev[3, 2] + acc_emwa[3, 2]], [1 - points_3D_prev[3, 1], 1 - points_3D_prev[3, 1] - acc_emwa[3, 1]], 'yellow')

        # Plot w_3d
        ax.plot([points_3D_prev[3, 0], points_3D_prev[3, 0] + w_3d_emwa[1]], [points_3D_prev[3, 2], points_3D_prev[3, 2] - w_3d_emwa[0]], [1 - points_3D_prev[3, 1], 1 - points_3D_prev[3, 1] + w_3d_emwa[2]], 'magenta')


        ax.set_xlim(-3,3)
        ax.set_ylim(-1,5)
        ax.set_zlim(-2,4)


        # Moving point plt
        #p_active_frames = [2000, 2100]

        #p_moving = [(idx-p_active_frames[0])/p_active_frames[0]-1]

        plt.ion()
        plt.draw()
        plt.pause(0.0000001)




        cv2.imshow('Visualization', current_frame)
        #cv2.waitKey()
        plt.cla()
        #ax.cla()
        results = {
              'image_fp': os.path.abspath(image_fp),
            'rvec': rvec,
            'tvec': tvec,
            'points_3D': points_3D,
            'points_2D': points_2D
        }

        #print(results)
        #print(rvec)
        #print(tvec)




cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)
plt.close()









