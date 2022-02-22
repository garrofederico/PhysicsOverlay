import xml.etree.ElementTree as ET
import numpy as np
import cv2
import json
import os
import matplotlib.pyplot as plt
# next 3 lines absolutely important for real time plot in Pycharm:
import matplotlib
import math
import yaml

matplotlib.use("TkAgg")


def project_points_intermediary(rvec, tvec, K, MESH):
    R = cv2.Rodrigues(rvec)[0]
    T = tvec
    RT = np.concatenate([R, T], axis=1)
    MESH = np.squeeze(MESH, -1)
    MESH_h = np.concatenate([MESH, np.ones((MESH.shape[0], 1))], axis=1)

    camera_coordinates = np.matmul(RT, MESH_h.T)
    uv_coordinates = np.matmul(K, camera_coordinates)
    image_coordiates = uv_coordinates / uv_coordinates[2, :]

    return camera_coordinates.T, uv_coordinates.T, image_coordiates.T[:, 0:2]


def rotation_matrix_to_euler_angles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def draw_frame(image, frame_of_reference):
    """It draws the frame on a numpy image.
    For reference:  frame_of_reference[0] is the red point
                    frame_of_reference[1] is the green point
                    frame_of_reference[2] is the blue point
                    frame_of_reference[3] is the black central point
    :param image:
    :param frame_of_reference:
    :return: image
    """
    # frame lines
    image = cv2.line(image, tuple(frame_of_reference[3]), tuple(frame_of_reference[0]), thickness=3,
                     color=(0, 0, 255))  # red
    image = cv2.line(image, tuple(frame_of_reference[3]), tuple(frame_of_reference[1]), thickness=3,
                     color=(0, 255, 0))  # green
    image = cv2.line(image, tuple(frame_of_reference[3]), tuple(frame_of_reference[2]), thickness=3,
                     color=(255, 0, 0))  # blue
    # frame corner points :
    image = cv2.circle(image, (frame_of_reference[0][0], frame_of_reference[0][1]), radius=5, color=(0, 0, 255),
                       thickness=-1)  # Red point
    image = cv2.circle(image, (frame_of_reference[1][0], frame_of_reference[1][1]), radius=5, color=(0, 255, 0),
                       thickness=-1)  # Green point
    image = cv2.circle(image, (frame_of_reference[2][0], frame_of_reference[2][1]), radius=5, color=(255, 0, 0),
                       thickness=-1)  # Blue point
    image = cv2.circle(image, (frame_of_reference[3][0], frame_of_reference[3][1]), radius=4, color=(0, 0, 0),
                       thickness=-1)  # Black point
    return image


def draw_filling_box(image, frame: np.array, box):
    """Gets and prints the spreadsheet's header columns
        p_y = points_2D_box[0, 0]
        p_c = points_2D_box[1][0]
        p_m = points_2D_box[2][0]
        p_w = points_2D_box[3][0]
Args:
    :param box: a numpy array of shape (4,2)
    :param frame: a numpy array of shape (4,2)
    file_loc (str): The file location of the spreadsheet
    print_cols (bool): A flag used to print the columns to the console
        (default is False)

Returns:
    list: a list of strings representing the header columns


    """
    # filling box lines:
    image = cv2.line(image, tuple(frame[2]), tuple(box[1]), thickness=1, color=(0, 0, 0))
    image = cv2.line(image, tuple(box[1]), tuple(frame[1]), thickness=1, color=(0, 0, 0))
    image = cv2.line(image, tuple(frame[1]), tuple(box[0]), thickness=1, color=(0, 0, 0))
    image = cv2.line(image, tuple(box[0]), tuple(box[3]), thickness=1, color=(0, 0, 0))
    image = cv2.line(image, tuple(box[3]), tuple(box[2]), thickness=1, color=(0, 0, 0))
    image = cv2.line(image, tuple(box[3]), tuple(box[1]), thickness=1, color=(0, 0, 0))
    image = cv2.line(image, tuple(box[2]), tuple(frame[2]), thickness=1, color=(0, 0, 0))
    image = cv2.line(image, tuple(box)[2], tuple(frame[0]), thickness=1, color=(0, 0, 0))
    image = cv2.line(image, tuple(frame[0]), tuple(box[0]), thickness=1, color=(0, 0, 0))
    # Corner points:
    # Filling box:
    image = cv2.circle(image, (box[0][0], box[0][1]), radius=5, color=(0, 255, 255),
                       thickness=-1)  # yellow point
    image = cv2.circle(image, (box[1][0], box[1][1]), radius=5, color=(255, 255, 0),
                       thickness=-1)  # cian point
    image = cv2.circle(image, (box[2][0], box[2][1]), radius=5, color=(255, 0, 255),
                       thickness=-1)  # magenta point
    image = cv2.circle(image, (box[3][0], box[3][1]), radius=5, color=(255, 255, 255),
                       thickness=-1)  # white point
    return image


# def draw_moving_point(image, frame, box, moving_point_2D):
#
#     # black frame line
#     image = cv2.line(image, moving_point_2D, ((box[2] + box[3]) // 2), thickness=1, color=(0, 0, 0))
#     # image = cv2.line(image, moving_point_2D, ((frame[1] + box[0]) // 2), thickness=1, color=(0, 0, 0))
#     # moving point
#     image = cv2.circle(image, (moving_point_2D[0], moving_point_2D[1]), radius=8, color=(0, 255, 100),
#                        thickness=-1)
#     # moving point line
#     image = cv2.line(image, ((frame[2] + box[1]) // 2), moving_point_2D, thickness=1, color=(0, 255, 100))
#     # image = cv2.line(image, ((frame[3] + frame[0]) // 2), moving_point_2D, thickness=1, color=(0, 255, 100))
#     mp_trajectory_2D.append(moving_point_2D)
#     for p in mp_trajectory_2D:
#         image = cv2.circle(image, (p[0], p[1]), radius=4, color=(0, 255, 100), thickness=-1)
#     #moving_point_3D, _, _ = project_points_intermediary(rvec, tvec, K, np.expand_dims(moving_point, 2))
#     # moving_point_3D = np.squeeze(moving_point_3D) reshape en vez
#     #mp_trajectory_3D.append(moving_point_3D)
#     # Plot point trajectory
#     # ax.plot(mp_trajectory_3D[1], -mp_trajectory_3D[0], mp_trajectory_3D[2], 'black')
#     return image


def draw_2D_projection(image, var_3D, rvec, tvec, K, dist, color=(0, 255, 255)):
    """

    :param image:
    :param var_3D:
    :param rvec:
    :param tvec:
    :param K:
    :param dist:
    :param color:
    """
    var_3D_rearranged = np.array([var_3D[:, 2], var_3D[:, 0], -var_3D[:, 1]], dtype=np.float64)
    var_2D = cv2.projectPoints(var_3D_rearranged, rvec, tvec, K, dist)[0]
    var_2D = np.squeeze(var_2D)  # change the shape from the output from projectPoints
    var_2D = np.array(var_2D, dtype=np.int32)
    # Speed of the central (black) point of R':
    image = cv2.circle(image, (var_2D[3, 0], var_2D[3, 1]), radius=5, color=color,
                       thickness=-1)
    image = cv2.line(image, tuple(points_2D_emwa[3]), tuple(var_2D[3]), thickness=3, color=color)


def calculate_w():
    """

    :return: single 3D point representing the module of w
    """
    x_vec = (points_3D[2] - points_3D[3]) / MESH_SIZE
    y_vec = (points_3D[1] - points_3D[3]) / MESH_SIZE
    z_vec = (points_3D[0] - points_3D[3]) / MESH_SIZE
    # Intermediate variables for debugging
    x_vec_prev = (points_3D_prev[2] - points_3D_prev[3]) / MESH_SIZE
    y_vec_prev = (points_3D_prev[1] - points_3D_prev[3]) / MESH_SIZE
    z_vec_prev = (points_3D_prev[0] - points_3D_prev[3]) / MESH_SIZE
    dx = (x_vec - x_vec_prev) / d_time
    dy = (y_vec - y_vec_prev) / d_time
    dz = (z_vec - z_vec_prev) / d_time
    w_x_1 = np.cross(x_vec, dy)
    w_x_2 = np.cross(x_vec, dz)
    w_y_1 = np.cross(y_vec, dx)
    w_y_2 = np.cross(y_vec, dz)
    w_z_1 = np.cross(z_vec, dx)
    w_z_2 = np.cross(z_vec, dy)
    fw_x_1 = np.dot(w_x_1, y_vec)
    fw_x_2 = np.dot(w_x_2, z_vec)
    fw_y_1 = np.dot(w_y_1, x_vec)
    fw_y_2 = np.dot(w_y_2, z_vec)
    fw_z_1 = np.dot(w_z_1, x_vec)
    fw_z_2 = np.dot(w_z_2, y_vec)
    w_3D = np.array([-fw_x_1, -fw_y_1, -fw_z_1], dtype=np.float64)
    return w_3D


class MovingPoint:
    """

    """
    def __init__(self, active_range, start_point, moving_axis, mesh_size, color):
        self.active_range = active_range
        self.start_point = start_point
        self.moving_axis = moving_axis
        self.mesh_size = mesh_size
        self.color = color
        self.mp_trajectory_2D = []


    def draw(self, index, frame):
        n = 3
        if self.active_range[0] <= index <= self.active_range[1]:
            moving_point = np.array([
                self.start_point[0], self.start_point[1] , self.start_point[2]
            ], dtype=np.float32)
            moving_point[self.moving_axis] += (index - self.active_range[0]) / (self.active_range[1] - self.active_range[0])
            moving_point = self.mesh_size * moving_point

            moving_point_2D = cv2.projectPoints(moving_point, rvec, tvec, K, dist)[0]
            moving_point_2D = np.squeeze(moving_point_2D)
            moving_point_2D = np.array(moving_point_2D, dtype=np.int32)  # float to int conversion, for ops involving discrete pixels
            self.mp_trajectory_2D.append(moving_point_2D)
            while len(self.mp_trajectory_2D) <= n: #fill with repeated points for the delay
                self.mp_trajectory_2D.append(moving_point_2D)

            # frame = draw_moving_point(frame, points_2D_emwa, points_2D_box_emwa, moving_point_2D)

            if self.moving_axis == 2:
                if self.start_point[0] == 1:
                    # black frame line
                    frame = cv2.line(frame, self.mp_trajectory_2D[-n], ((points_2D_box_emwa[2] + points_2D_box_emwa[3]) // 2), thickness=1, color=(0, 0, 0))
                    # moving point line
                    frame = cv2.line(frame, ((points_2D_emwa[2] + points_2D_box_emwa[1]) // 2), self.mp_trajectory_2D[-n], thickness=1, color=self.color)
                else:
                    # black frame line
                    frame = cv2.line(frame, self.mp_trajectory_2D[-n], ((points_2D_emwa[0] + points_2D_box_emwa[2]) // 2), thickness=1, color=(0, 0, 0))
                    # moving point line
                    frame = cv2.line(frame, ((points_2D_emwa[3] + points_2D_emwa[2]) // 2), self.mp_trajectory_2D[-n], thickness=1, color=self.color)
            if self.moving_axis == 1:
                # black frame line
                frame = cv2.line(frame, self.mp_trajectory_2D[-n], ((points_2D_emwa[1] + points_2D_box_emwa[0]) // 2), thickness=1, color=(0, 0, 0))
                # moving point line
                frame = cv2.line(frame, ((points_2D_emwa[3] + points_2D_emwa[0]) // 2), self.mp_trajectory_2D[-n], thickness=1, color=self.color)
            if self.moving_axis == 0:
                pass # to do
            # moving point
            frame = cv2.circle(frame, (self.mp_trajectory_2D[-n][0], self.mp_trajectory_2D[-n][1]), radius=8, color=self.color,
                               thickness=-1)


            # frame = cv2.polylines(frame, np.reshape(self.mp_trajectory_2D, (-1,1,2)), isClosed=1, color=self.color, thickness =1)
            for p in self.mp_trajectory_2D[:-n]:
                frame = cv2.circle(frame, (p[0], p[1]), radius=4, color=self.color, thickness=-1)
        return frame


def get_poses_from_annotations(cfg):
    # Retrieve annotated points from Cvat xml file
    # Returns "format"
    tree = ET.parse(cfg["SETUP"]["ANNOTATIONS_PATH"])
    root = tree.getroot()
    poses_from_annotations = []
    for c in root[2:]:
        points = []
        for point in c[:N_KEYPOINTS]:
            x, y = point.attrib['points'].split(',')
            x = float(x)
            y = float(y)
            points.append([x, y])
        points = np.array(points, dtype=np.float32)
        points = np.reshape(points, (N_KEYPOINTS, 2, 1))

        _, rvec, tvec, _ = cv2.solvePnPRansac(mesh, points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        points_3D, _, _ = project_points_intermediary(rvec, tvec, K, mesh)
        points_2D = cv2.projectPoints(mesh, rvec, tvec, K, dist)[0]
        points_2D = np.squeeze(points_2D)
        points_2D = np.array(points_2D,
                             dtype=np.int32)  # float to int conversion, for ops involving discrete pixels
        poses_from_annotations.append([(rvec, tvec), points_2D, points_3D])
    return poses_from_annotations


def get_poses_from_predictions(cfg):
    poses_from_preds = []
    preds_json = json.load(open(cfg["SETUP"]["PREDS_PATH"]))
    pred_list = list(preds_json.values())
    for pred in pred_list:
        pred = pred['pred']
        pred_not_empty = len(pred) > 0
        if pred_not_empty:

            pred_confidence = pred[0][0]
            rmat = np.array(pred[0][2])
            rvec, _ = cv2.Rodrigues(rmat)
            tvec = np.array(pred[0][3])
            points_3D, _, _ = project_points_intermediary(rvec, tvec, K, mesh)
            points_2D = cv2.projectPoints(mesh, rvec, tvec, K, dist)[0]
            points_2D = np.squeeze(points_2D)
            # _, rvec, tvec, _ = cv2.solvePnPRansac(mesh, points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            poses_from_preds.append([(rvec, tvec), points_2D, points_3D, pred_confidence])
        else:
            # TODO: interpolate when empty pred
            # find next valid prediction
            # count all the empty ones
            # interpolate
            poses_from_preds.append(None)
    return poses_from_preds


def get_config_file(config_file):
    assert os.path.exists(config_file)
    with open(config_file, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)

        except yaml.YAMLError as exc:
            print(exc)
    return cfg

if __name__ == '__main__':
    config_file = 'CONFIGS/object-video_name.yaml'

    cfg = get_config_file(config_file)

    INTRINSICS_FILE = os.path.abspath('../physic_overlay_codebase/CALIBRATION/intrinsics.json')
    # INTRINSICS_FILE = os.path.abspath('../CALIBRATION/intrinsics.json')
    MESH = os.path.abspath('MESH/repere.json')
    EXPERIMENT = 'experiment3'
    # EXPERIMENT = 'experiment4'
    CAMERA = '6_2'

    OFFSET = 1  # Sync (309) + beginning of the experience (exp1)
    N_KEYPOINTS = 4
    IMAGES_PATH = os.path.abspath('../physic_overlay_codebase/DATA/6_2/GH010483/')
    # IMAGES_PATH = os.path.abspath('../physic_overlay_codebase/DATA/6_2/GH010484/')
    # IMAGES_PATH = os.path.abspath('../DATA/6_2/GH010434/')
    # ANNOTATIONS_PATH = 'ANNOTATIONS/' + CAMERA + '-' + EXPERIMENT + '_points.xml'

    #Repere Size
    MESH_SIZE = 37

    # get yaml config file





    intrinsics_json = json.load(open(os.path.abspath(cfg["SETUP"]["INTRINSICS_FILE"])))
    K = np.array(intrinsics_json[CAMERA]['K_new'], dtype=np.float32)
    dist = None  # loaded_json['6_2']['dist']
    mesh = np.array(json.load(open(MESH))["points"], dtype=np.float32)
    mesh = np.reshape(mesh, (N_KEYPOINTS, 3, 1))
    mesh *= MESH_SIZE

    mesh_box = MESH_SIZE * np.array([
        [0, 1, 1],
        [1, 1, 0],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.float32)
    ########## Start of script #####################

    # Matplotlib plot init:
    ax = plt.axes(projection='3d')

    DEBUG_OFFSET = cfg["SETUP"]["DEBUG_OFFSET"]
    # speed_3D variables init:
    frame_rate = 30  # Go Pro's frame rate
    d_time = 1 / frame_rate



    assert cfg["SETUP"]["POSES_FROM"] == 'annotations' or cfg["SETUP"]["POSES_FROM"] == 'predictions'\
            , "POSES_FROM in config file has to be 'annotations' or 'predictions' "
    if cfg["SETUP"]["POSES_FROM"] == 'predictions':
        poses = get_poses_from_predictions(cfg)
    if cfg["SETUP"]["POSES_FROM"] == 'annotations':
        poses = get_poses_from_annotations(cfg)




    # position exponential moving window average init:
    points_2D_emwa = cv2.projectPoints(mesh, poses[0 + DEBUG_OFFSET][0][0], poses[0 + DEBUG_OFFSET][0][1], K, dist)[0]
    points_2D_emwa = np.squeeze(points_2D_emwa)
    points_2D_box_emwa = cv2.projectPoints(mesh_box, poses[0 + DEBUG_OFFSET][0][0], poses[0 + DEBUG_OFFSET][0][1], K, dist)[0]
    points_2D_box_emwa = np.squeeze(points_2D_box_emwa)
    frame_filter = 3
    beta_position_2D = (frame_filter - 1 )/ float(frame_filter)

    # position exponential moving window average init 3D:
    points_3D_emwa = points_3D_prev
    points_3D_emwa_prev = points_3D_emwa
    # rmat and tvec ewma init:
    rmat_ewma = np.zeros((3,3))
    tvec_ewma = np.zeros((3,1))

    # speed exponential moving window average init:

    speed_3D_emwa = 0
    beta_speed = 0.98

    # acc_3d variables init:
    acc_3d_prev = 0
    speed_emwa_prev = 0
    # acc exponential moving window average init:
    acc_3D_emwa = 0
    beta_acc = 0.95

    # w_3d variables init:
    angles1_prev = 0
    # w_3d exponential moving window average init:
    w_3D_emwa = 0
    beta_w_3d = 0.9




    make_video = True
    if make_video:
        out = cv2.VideoWriter('test_fede.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (960, 1080))
        out2 = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (1920, 1080))

    #definition of moving points:
    moving_p1 = MovingPoint([1250, 1700],(1, 0.5, 0), 2, MESH_SIZE, color=(0, 110, 100))
    moving_p2 = MovingPoint([1850, 2250], (0.5, 0, 0), 2, MESH_SIZE, color=(100, 0, 100))
    moving_p3 = MovingPoint([2420, 2800], (0, 0, 0.5), 1, MESH_SIZE, color=(100, 50, 200))
    ######## FRAME LOOP #########
    pred_list = list(preds_json.values())
    for idx in range(len(pred_list)-DEBUG_OFFSET):
        idx += DEBUG_OFFSET
        if idx < frame_filter: # to prevent invalid image filepath
            idx = frame_filter
        pred = pred_list[idx]['pred']
        pred_not_empty = len(pred) > 0
        if pred_not_empty :

            print(idx)
            image_fp = os.path.join(IMAGES_PATH,
                                    str(OFFSET + idx - frame_filter ).zfill(5) + '.jpg')  # OFFSET is the begining of notation frames
            assert os.path.exists(image_fp)

            pred_confidence = pred[0][0]
            rmat = np.array(pred[0][2])
            # ewma for R
            rmat_ewma = beta_position_2D * rmat_ewma + (1 - beta_position_2D) * rmat
            rvec,_ = cv2.Rodrigues(rmat_ewma)
            # rvec = rodrigues(rmat)
            tvec = np.array(pred[0][3])
            # ewma for t
            tvec_ewma = beta_position_2D * tvec_ewma + (1 - beta_position_2D) * tvec
            tvec = tvec_ewma

            # _, rvec, tvec, _ = cv2.solvePnPRansac(mesh, points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            points_3D, _, _ = project_points_intermediary(rvec, tvec, K, mesh)
            points_2D = cv2.projectPoints(mesh, rvec, tvec, K, dist)[0]
            points_2D = np.squeeze(points_2D)
            # Position 2D exponential moving weighted average
            points_2D_emwa = beta_position_2D * points_2D_emwa + (1 - beta_position_2D) * points_2D
            points_2D_emwa = np.array(points_2D_emwa, dtype=np.int32)#float to int conversion, for ops involving discrete pixels
            # Position 2D exponential moving weighted average
            points_3D_emwa = beta_position_2D * points_3D_emwa + (1 - beta_position_2D) * points_3D
            points_2D_emwa = np.array(points_2D_emwa, dtype=np.int32)#float to int conversion, for ops involving discrete pixels


            # Filling box projection
            points_2D_box = cv2.projectPoints(mesh_box, rvec, tvec, K, dist)[0]
            points_2D_box = np.array(points_2D_box,
                                     dtype=np.int32)  # float to int conversion, for ops involving discrete pixels
            points_2D_box = np.squeeze(points_2D_box)
            # Position 2D exponential moving weighted average
            points_2D_box_emwa = beta_position_2D * points_2D_box_emwa + (1 - beta_position_2D) * points_2D_box
            points_2D_box_emwa = np.array(points_2D_box_emwa, dtype=np.int32)#float to int conversion, for ops involving discrete pixels

            # Visualization:
            current_frame = cv2.imread(image_fp)


            # Values for fixed FoR
            fixed_frame = np.array([[229, 401], [468, 626], [59, 665], [234, 628]])

            # fixed frame lines:
            current_frame = draw_frame(current_frame, fixed_frame)
            # moving frame lines:
            points_2D_emwa = np.squeeze(points_2D_emwa)
            current_frame = draw_frame(current_frame, points_2D_emwa)
            # filling box
            points_2D_box_emwa = np.squeeze(points_2D_box_emwa)
            current_frame = draw_filling_box(current_frame, points_2D_emwa, points_2D_box_emwa)

            # Calculation of speed_3D
            speed_3D = (points_3D - points_3D_prev) / d_time
            # speed exponential moving weighted average
            speed_3D_emwa = beta_speed * speed_3D_emwa + (1 - beta_speed) * speed_3D

            # 2D speed projection
            draw_2D_projection(current_frame, speed_3D_emwa, rvec, tvec, K, dist, color=(0, 255, 255))

            # Calculation of acc_3D
            acc_3D = (speed_3D_emwa - speed_emwa_prev) / d_time
            speed_emwa_prev = speed_3D_emwa
            # acc exponential moving weighted average
            acc_3D_emwa = beta_acc * acc_3D_emwa + (1 - beta_acc) * acc_3D

            # 2D acc projection
            draw_2D_projection(current_frame, acc_3D_emwa, rvec, tvec, K, dist, color=(0, 128, 255))

            # Calculation of w (returns a single 3D point

            w_3D = calculate_w()
            scale_w = 10
            w_3D = w_3D * scale_w
            w_3D_emwa = beta_w_3d * w_3D_emwa + (1 - beta_w_3d) * w_3D
            # 2D value to visualize in image


            w_2d = cv2.projectPoints(w_3D_emwa, rvec, tvec, K, dist)[0]
            w_2d = np.squeeze(w_2d)  # change the shape from the output from projectPoints
            w_2d = np.array(w_2d, dtype=np.int32)

            # w of the central (black) point of R':

            current_frame = cv2.circle(current_frame, (w_2d[0], w_2d[1]), radius=5, color=(255, 0, 255), thickness=-1)
            current_frame = cv2.line(current_frame, tuple(points_2D_emwa[3]), tuple(w_2d), thickness=3, color=(255, 0, 255))

            # Refresh of points 3D used in calculation of w and speed_3d
            points_3D_prev = points_3D
            # draw moving points
            current_frame = moving_p1.draw(idx, current_frame)
            current_frame = moving_p2.draw(idx, current_frame)
            current_frame = moving_p3.draw(idx, current_frame)

            # 3D plot
            for pair in [(0, 3, 'red'), (1, 3, 'green'), [2, 3, 'blue']]:
                p1 = points_3D[pair[0]]
                p2 = points_3D[pair[1]]
                ax.plot([p1[0], p2[0]], [p1[2], p2[2]], [1 - p1[1], 1 - p2[1]], pair[2])
            # Plot speed_3d
            # ax.plot([0, 0], [0, 0],[1, 0], 'yellow')

            ax.plot([points_3D_prev[3, 0], points_3D_prev[3, 0] + speed_3D_emwa[3, 0]],
                    [points_3D_prev[3, 2], points_3D_prev[3, 2] + speed_3D_emwa[3, 2]],
                    [1 - points_3D_prev[3, 1], 1 - points_3D_prev[3, 1] - speed_3D_emwa[3, 1]], 'yellow')

            # Plot acc_3d
            ax.plot([points_3D_prev[3, 0], points_3D_prev[3, 0] + acc_3D_emwa[3, 0]],
                    [points_3D_prev[3, 2], points_3D_prev[3, 2] + acc_3D_emwa[3, 2]],
                    [1 - points_3D_prev[3, 1], 1 - points_3D_prev[3, 1] - acc_3D_emwa[3, 1]], 'orange')

            # Plot w_3d
            ax.plot([points_3D_prev[3, 0], points_3D_prev[3, 0] + w_3D_emwa[1]],
                    [points_3D_prev[3, 2], points_3D_prev[3, 2] - w_3D_emwa[0]],
                    [1 - points_3D_prev[3, 1], 1 - points_3D_prev[3, 1] + w_3D_emwa[2]], 'magenta')

            ax.set_xlim(-3 * MESH_SIZE, 3 * MESH_SIZE)
            ax.set_ylim(-1 * MESH_SIZE, 5 * MESH_SIZE)
            ax.set_zlim(-2 * MESH_SIZE, 4 * MESH_SIZE)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # TODO: avoid saving image to make video https://www.codegrepper.com/code-examples/python/convert+matplotlib+plot+to+video+opencv+python
            # https://stackoverflow.com/questions/42603161/convert-an-image-shown-in-python-into-an-opencv-image
            if make_video:
                out2.write(current_frame)
                # plt.savefig(os.path.join('temp', str(idx).zfill(4) + '_3d.jpg'))
                #
                # fp_2 = os.path.join('temp', str(idx).zfill(4) + '_3d.jpg')
                # im1 = cv2.resize(current_frame, (960, 540))
                # im2 = cv2.resize(cv2.imread(fp_2), (960, 540))
                # im3 = np.vstack([im1, im2])
                # print(im3.shape)
                # out.write(im3)

                # assert os.path.exists(fp_2)

            plt.ion()
            plt.draw()
            plt.pause(0.0000001)

            cv2.imshow('Visualization', current_frame)
            # cv2.waitKey()
            plt.cla()
            # ax.cla()
            results = {
                'image_fp': os.path.abspath(image_fp),
                'rvec': rvec,
                'tvec': tvec,
                'points_3D': points_3D,
                'points_2D': points_2D_emwa
            }
            # if make_video and idx == DEBUG_OFFSET +200:
            #     out.release()
            #     out2.release()
            #     break


    # print(results)
            # print(rvec)
            # print(tvec)
if make_video:
    out.release()
    out2.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)
plt.close()
