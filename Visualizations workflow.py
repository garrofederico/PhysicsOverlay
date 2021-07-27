import xml.etree.ElementTree as ET
import numpy as np
import cv2
import json
import os

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
    for idx, c in enumerate(root[2:]):
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

        #Visualization:
        current_frame = cv2.imread(image_fp)

        points_2D = np.array(points_2D, dtype=np.int32) #float to int convertion, for ops involving discrete pixels
        p_r = points_2D[0,0]
        p_g = points_2D[1][0]
        p_b = points_2D[2][0]
        p_black = points_2D[3][0]
        #Corner points:
        current_frame = cv2.circle(current_frame, (p_r[0],p_r[1]), radius=4, color=(0, 0, 255), thickness=-1) #Red point
        current_frame = cv2.circle(current_frame, (p_g[0],p_g[1]), radius=4, color=(0, 255, 0), thickness=-1) #Green point
        current_frame = cv2.circle(current_frame, (p_b[0],p_b[1]), radius=4, color=(255, 0, 0), thickness=-1) #Blue point
        current_frame = cv2.circle(current_frame, (p_black[0],p_black[1]), radius=4, color=(0, 0, 0), thickness=-1) #Black point

        cv2.imshow('frame: ' + str(idx) + '     image: ' + image_fp[-28:], current_frame)
        cv2.waitKey()
        results = {
            'image_fp': os.path.abspath(image_fp),
            'rvec': rvec,
            'tvec': tvec,
            'points_3D': points_3D,
            'points_2D': points_2D
        }
        print(results)
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)









