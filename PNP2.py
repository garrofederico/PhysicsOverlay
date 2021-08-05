import xml.etree.ElementTree as ET
import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
def project_points_intermediary(rvec, tvec, K, MESH):
    R = cv2.Rodrigues(rvec)[0]
    T = tvec
    RT = np.concatenate([R,T], axis=1)
    MESH_h = np.concatenate([MESH, np.ones((MESH.shape[0], 1))], axis=1)


    camera_coordinates = np.matmul(RT, MESH_h.T)
    uv_coordinates = np.matmul(K, camera_coordinates)
    image_coordiates = uv_coordinates / uv_coordinates[2,:]

    return camera_coordinates.T, uv_coordinates.T, image_coordiates.T[:,0:2]


K_new=np.array([
     [
        1027.5659973144532,
        0.0,
        967.1422899869882
     ],
     [
        0.0,
        1019.6757598876953,
        555.2828187622993
     ],
     [
        0.0,
        0.0,
        1.0
     ]
  ])
  
X = []
Y=[]
scale = 0.1925
points_3d = np.array([

[0, -1, 0],
[0, 1, 0],
[0, 0, -scale],
[0, 0, scale],
[-1, 0, 0],
[1, 0, 0]
], dtype=np.float32)

points_3d = np.reshape(points_3d, (6,3,1))

#tree = ET.parse('6_1-experiment1.xml')
tree = ET.parse('ANNOTATIONS/6_2-experiment2_points.xml')
root = tree.getroot()
dist=None
out = cv2.VideoWriter('test4.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (1920, 1080))

rep_1 = []
rep_2 = []
score = 0
for idx, c in enumerate(root[2:302]):
    # if os.path.exists(os.path.join('temp', str(idx).zfill(4) + '.jpg')):continue
    points = []
    for point in c[:6]:
        x, y = point.attrib['points'].split(',')
        x = float(x)
        y = float(y)
        points.append([x, y])
    #print(points)
        
    
    points = np.array(points, dtype=np.float32)
    points = np.reshape(points, (6,2,1))
    
    #print(points_3d.shape, points.shape)

    _, rvec, tvec = cv2.solvePnP(points_3d, points, K_new, dist)
    projected = cv2.projectPoints(points_3d, rvec, tvec, K_new, dist)[0]
    
    gt = np.squeeze(points, axis=2)
    pred = np.squeeze(projected, axis=1)
    
    a = (gt[:, 0] - pred[: , 0]) * (gt[:, 0] - pred[: , 0]) + (gt[:, 1] - pred[: , 1]) * (gt[:, 1] - pred[: , 1])
 
  
    score += np.sum(np.sqrt(a))

    projected = np.array(projected, dtype=np.int32)
    #print(projected.shape)
  
    #im = cv2.imread(r'D:\physic_overlay\6_1\GH010172\\' +str(3478+idx).zfill(5) + '.jpg')     
    im = cv2.imread(r'../DATA/6_2/GH010434/' +str(12240+309+idx).zfill(5) + '.jpg')     
    
    p1 = projected[0][0]
    p2 = projected[1][0]
    p3 = projected[2][0]
    p4 = projected[3][0]
    p5 = projected[4][0]
    p6 = projected[5][0]

    
    im = cv2.line(im, (p1[0], p1[1]),(p2[0], p2[1]),thickness=3, color=(0,0,255))
    im = cv2.line(im, (p5[0], p5[1]),(p6[0], p6[1]),thickness=3, color=(0,255,0))
    im = cv2.line(im, (p3[0], p3[1]),(p4[0], p4[1]),thickness=3, color=(255,0,0))
    #im = cv2.line(im, (b[0], b[1]),(x[0], x[1]),thickness=3, color=(255,0,0))
    #
    ##im = cv2.line(im, (rg[0], rg[1]),(r[0], r[1]),thickness=1, color=(0,0,0))
    ##im = cv2.line(im, (rg[0], rg[1]),(g[0], g[1]),thickness=1, color=(0,0,0))
    ##im = cv2.line(im, (rb[0], rb[1]),(r[0], r[1]),thickness=1, color=(0,0,0))
    ##im = cv2.line(im, (rb[0], rb[1]),(b[0], b[1]),thickness=1, color=(0,0,0))
    ##im = cv2.line(im, (gb[0], gb[1]),(b[0], b[1]),thickness=1, color=(0,0,0))
    ##im = cv2.line(im, (gb[0], gb[1]),(g[0], g[1]),thickness=1, color=(0,0,0))
    ##
    ##
    ##im = cv2.line(im, (rg[0], rg[1]),(w[0], w[1]),thickness=1, color=(0,0,0))
    ##im = cv2.line(im, (rb[0], rb[1]),(w[0], w[1]),thickness=1, color=(0,0,0))
    ##im = cv2.line(im, (gb[0], gb[1]),(w[0], w[1]),thickness=1, color=(0,0,0))
    ##
    ##
    ##
    ##im = cv2.circle(im, (r[0], r[1]), radius=3, thickness=3,color=(0,0,255))
    ##im = cv2.circle(im, (g[0], g[1]), radius=3, thickness=3,color=(0,255,0))
    ##im = cv2.circle(im, (b[0], b[1]), radius=3, thickness=3,color=(255,0,0))
    ##im = cv2.circle(im, (x[0], x[1]), radius=3, thickness=3,color=(0,0,0))
    ##im = cv2.circle(im, (rg[0], rg[1]), radius=3, thickness=3,color=(0,255,255))
    ##im = cv2.circle(im, (rb[0], rb[1]), radius=3, thickness=3,color=(255,0,255))
    ##im = cv2.circle(im, (gb[0], gb[1]), radius=3, thickness=3,color=(255,255,0))
    ##im = cv2.circle(im, (w[0], w[1]), radius=3, thickness=3,color=(255,255,255))
    #
    #points = []
    #for point in c[6:]:
    #    x, y = point.attrib['points'].split(',')
    #    x = float(x)
    #    y = float(y)
    #    points.append([x, y])
    #    
    #    
    #
    #points = np.array(points, dtype=np.float32)
    #
    #points = np.reshape(points, (6,2,1))
    #
    #_, rvec, tvec, _ = cv2.solvePnPRansac(points_3d, points, K_new, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    #P, _, _ = project_points_intermediary(rvec, tvec, K_new, MESH)
    #rep_2.append(P)
    #
    #
    #
    #projected = cv2.projectPoints(MESH, rvec, tvec, K_new, dist)[0]
    #projected = np.array(projected, dtype=np.int32)
    ##print(projected.shape)
    #
    ##im = cv2.imread(r'D:\physic_overlay\6_1\GH010172\\' +str(3478+idx).zfill(5) + '.jpg')     
    ##im = cv2.imread(r'D:\physic_overlay\6_2\GH010434\\' +str(3369+idx).zfill(5) + '.jpg')     
    #
    #r = projected[0][0]
    #g = projected[1][0]
    #b = projected[2][0]
    #x = projected[3][0]
    #rg = projected[4][0]
    #rb = projected[5][0]
    #gb = projected[6][0]
    #w = projected[7][0]
    #
    #im = cv2.line(im, (r[0], r[1]),(x[0], x[1]),thickness=3, color=(0,0,255))
    #im = cv2.line(im, (g[0], g[1]),(x[0], x[1]),thickness=3, color=(0,255,0))
    #im = cv2.line(im, (b[0], b[1]),(x[0], x[1]),thickness=3, color=(255,0,0))
    #
    #im = cv2.line(im, (rg[0], rg[1]),(r[0], r[1]),thickness=1, color=(0,0,0))
    #im = cv2.line(im, (rg[0], rg[1]),(g[0], g[1]),thickness=1, color=(0,0,0))
    #im = cv2.line(im, (rb[0], rb[1]),(r[0], r[1]),thickness=1, color=(0,0,0))
    #im = cv2.line(im, (rb[0], rb[1]),(b[0], b[1]),thickness=1, color=(0,0,0))
    #im = cv2.line(im, (gb[0], gb[1]),(b[0], b[1]),thickness=1, color=(0,0,0))
    #im = cv2.line(im, (gb[0], gb[1]),(g[0], g[1]),thickness=1, color=(0,0,0))
    #
    #
    #im = cv2.line(im, (rg[0], rg[1]),(w[0], w[1]),thickness=1, color=(0,0,0))
    #im = cv2.line(im, (rb[0], rb[1]),(w[0], w[1]),thickness=1, color=(0,0,0))
    #im = cv2.line(im, (gb[0], gb[1]),(w[0], w[1]),thickness=1, color=(0,0,0))
    #
    #
    #
    #im = cv2.circle(im, (r[0], r[1]), radius=3, thickness=3,color=(0,0,255))
    #im = cv2.circle(im, (g[0], g[1]), radius=3, thickness=3,color=(0,255,0))
    #im = cv2.circle(im, (b[0], b[1]), radius=3, thickness=3,color=(255,0,0))
    #im = cv2.circle(im, (x[0], x[1]), radius=3, thickness=3,color=(0,0,0))
    #im = cv2.circle(im, (rg[0], rg[1]), radius=3, thickness=3,color=(0,255,255))
    #im = cv2.circle(im, (rb[0], rb[1]), radius=3, thickness=3,color=(255,0,255))
    #im = cv2.circle(im, (gb[0], gb[1]), radius=3, thickness=3,color=(255,255,0))
    #im = cv2.circle(im, (w[0], w[1]), radius=3, thickness=3,color=(255,255,255))
    #
    out.write(im)
    #cv2.imwrite(os.path.join('temp', str(idx).zfill(4) + '.jpg'), im)
X.append(scale)
Y.append(score)
print(scale, score)
plt.plot(X,Y)
#plt.show()

# YY = np.concatenate([np.array(rep_1), np.array(rep_2)],axis=0)
# print(YY.shape)
# min_x = np.min(YY[:,:,0])
# min_y = np.min(YY[:,:,2])
# min_z = np.min(1-YY[:,:,1])
#
# max_x = np.max(YY[:,:,0])
# max_y = np.max(YY[:,:,2])
# max_z = np.max(1-YY[:,:,1])
#out = cv2.VideoWriter('test3.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 60, (640, 480))
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for idx, elm in enumerate(zip(rep_1, rep_2)):

    for points in elm:
        for pair in [(0, 3, 'red'), (1, 3, 'green'), [2, 3, 'blue']]:
            p1 = points[pair[0]]
            p2 = points[pair[1]]
            ax.plot([p1[0], p2[0]], [p1[2], p2[2]], [1-p1[1], 1-p2[1]], pair[2])
    #ax.set_xlim(min_x,max_x)
    #ax.set_ylim(min_y,max_y)
    #ax.set_zlim(min_z,max_z)
    #plt.savefig(os.path.join('temp', str(idx).zfill(4) + '_3d.jpg'))
    plt.draw()
    plt.pause(0.1)

    #out.write(cv2.imread('temp.jpg'))

out = cv2.VideoWriter('test3.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 60, (960, 1080))
for x in range(len(rep_1)):
    fp_1 = os.path.join('temp', str(x).zfill(4) + '.jpg')
    fp_2 = os.path.join('temp', str(x).zfill(4) + '_3d.jpg')
    im1 = cv2.resize(cv2.imread(fp_1), (960, 540))
    im2 = cv2.resize(cv2.imread(fp_2), (960, 540))
    im3 = np.vstack([im1, im2])
    print(im3.shape)
    out.write(im3)
    assert os.path.exists(fp_1) and os.path.exists(fp_2)
