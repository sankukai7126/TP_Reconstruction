# -*- coding: utf-8 -*-
"""
Created on 2 Dec 2020

@author: chatoux
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy import ndimage
import glob

def CameraCalibration():
    # Calcul de critere
    # cv.TERM_CRITERIA_EPS -->  Stop the algorithm iteration if specified accuracy, epsilon, is reached.
    # cv.TERM_CRITERIA_MAX_ITER --> Stop the algorithm after the specified number of iterations, max_iter.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    
    # Tableau correspondant aux points
    ############ to adapt ##########################
    objp = np.zeros((4*6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:4, 0:6].T.reshape(-1, 2)
    # Parametre de taille des cases du damier
    objp[:, :2] *= 40
    #################################################


    # Initialise les tableaux pour stocker les points objet et le spoints de l'image de toutes les images
    objpoints = []  # points 3D de l'espace réélle
    imgpoints = []  # points 2D du plan d'image
    ############ to adapt ##########################
    images = glob.glob('C:/Users/Mikrail/Downloads/TP_Reconstruction/TP/Images/chess/P30/*.jpg')
    #################################################
    for fname in images:
        # img = cv.imread(fname)
        img = cv.pyrDown(cv.imread(fname))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # # Cherche les coin du plateau
        ############ to adapt ##########################
        ret, corners = cv.findChessboardCorners(gray, (4, 6), None)
        #################################################
        print(ret)
        # On fais le traitement que si on repère un coin
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # Déssine et affiche les coins
            ############ to adapt ##########################
            cv.drawChessboardCorners(img, (4, 6), corners2, ret)
            #################################################
            cv.namedWindow('img', 0 )
            cv.imshow('img', img)
            cv.waitKey(500)
    cv.destroyAllWindows()

    # # On callibre la camera, on obtient ainsi matrice de la caméra, le coeff de distortion, le vecteur de rotation et le vecteur de translation
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print('camraMatrix\n',mtx)
    print('dist\n',dist)

    ############ to adapt ##########################
    img = cv.pyrDown(cv.imread('C:/Users/Mikrail/Downloads/TP_Reconstruction/TP/Images/chess/P30/IMG_20201206_093855.jpg'))
    #################################################
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    print('newcameramtx\n',newcameramtx)

    # On enlève la distortion pour ravoire l'image de début
    # Quels phénomènes peut engendrer la déformation de la figure 2d ?
    #
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    # On recadre l'image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv.namedWindow('img', 0)
    cv.imshow('img', dst)
    cv.waitKey(0)
    ############ to adapt ##########################
    cv.imwrite('C:/Users/Mikrail/Downloads/TP_Reconstruction/TP/Images/calibresultM.png', dst)
    #################################################

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error / len(objpoints)))

    return newcameramtx

def DepthMapfromStereoImages():
    ############ to adapt ##########################
    imgL = cv.pyrDown(cv.imread('C:/Users/Mikrail/Downloads/TP_Reconstruction/TP/Images/aloeL.jpg'))
    imgR = cv.pyrDown(cv.imread('C:/Users/Mikrail/Downloads/TP_Reconstruction/TP/Images/aloeR.jpg'))
    #################################################
    # On souhaite obtenir la profondeur de l'image
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = 16,
    P1 = 8 * 3 * window_size ** 2,
    P2 = 32 * 3 * window_size ** 2,
    disp12MaxDiff = 16,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32)
    # compute
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    # 
    plt.figure('3D')
    plt.imshow((disparity-min_disp)/num_disp,'gray')
    plt.colorbar()
    plt.show()


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color, 2)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def StereoCalibrate(Cameramtx):
    ############ to adapt ##########################
    img1 = cv.pyrDown(cv.imread('C:/Users/Mikrail/Downloads/TP_Reconstruction/TP/Images/leftT2.jpg', 0))
    img2 = cv.pyrDown(cv.imread('C:/Users/Mikrail/Downloads/TP_Reconstruction/TP/Images/rightT2.jpg', 0))
    #################################################
    # opencv 4.5
    sift = cv.SIFT_create()
    # opencv 3.4
    # sift = cv.xfeatures2d.SIFT_create()
    # On détecte les points cles des deux images avec l'algo sift, les deux images sont les mêmes juste retourner
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # On coupe l'images en plusieurs morceau pour faire un arbre (kdtree) pour travails en simultané plusieurs parties de l'images (pour du multithreading par exemple)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    pts1 = []
    pts2 = []
    # 
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    # On match les key points des deux images pour voire si on retrouve le même résultat
    img3 = cv.drawMatches(img1,kp1,img2,kp2,good, None, flags=2)
    plt.imshow(img3)
    plt.show()

    # Calcul une matrice essentiel entre les deux points de 2 images
    E, maskE = cv.findEssentialMat(pts1, pts2, Cameramtx, method=cv.FM_LMEDS)
    print('E\n',E)
    # On calcul la rotation de la camera et sa translation à partir de la matrice essentiel précedente
    retval, R, t, maskP = cv.recoverPose(E, pts1, pts2, Cameramtx, maskE)
    print('R\n', R)
    print('t\n', t)

    # Calcul de la matrice fondamentale extrait par openCV
    F, maskF = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
    
    #Calcul de la matrice essentielle théorique
    Ktest = Cameramtx.reshape(3,3) # on utulise la matric de la caméra calibrer pour trouver K

    Eth =  Ktest.T.dot(F).dot(Ktest)
    print('E théorique\n',Eth)
    
    #Calcul de R theorique
    U, S, Vt = np.linalg.svd(E)
    W = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
    Rth = U.dot(W).dot(Vt)
    print('R théorique\n',Rth) # On retrouve bien la bonne valeur

    # Matrice Fondamental
    print('F\n', F)

    # Calcul de la matrice fondamentale à partir de la matrice essentielle
    FT = np.cross(F*t,E)

    return pts1, pts2, F, maskF, FT, maskE

def EpipolarGeometry(pts1, pts2, F, maskF, FT, maskE):
    ############ to adapt ##########################
    img1 = cv.pyrDown(cv.imread('C:/Users/Mikrail/Downloads/TP_Reconstruction/TP/Images/leftT2.jpg', 0))
    img2 = cv.pyrDown(cv.imread('C:/Users/Mikrail/Downloads/TP_Reconstruction/TP/Images/rightT2.jpg', 0))
    #################################################
    r,c = img1.shape

    # On recupere deux tableaux qui ont des valeurs qui se croisent
    pts1F = pts1[maskF.ravel() == 1]
    pts2F = pts2[maskF.ravel() == 1]

    # On calcul des lignes epipolaires correspondantes dans les autres images a partir des tous les points de l'image et on les dessine
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1F,pts2F)
    # On recalcule de nouvelles lignes epipolaires et on les alignes sur toutes les images
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
    plt.figure('Fright')
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img6)
    plt.figure('Fleft')
    plt.subplot(121),plt.imshow(img4)
    plt.subplot(122),plt.imshow(img3)

    # On recupere deux tableaux qui ont des valeurs qui se croisent
    pts1 = pts1[maskE.ravel() == 1]
    pts2 = pts2[maskE.ravel() == 1]
    # On calcul de nouvelles lignes epipolaire du cotés droit de l'image FT que l'on affichent
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,FT)
    lines1 = lines1.reshape(-1,3)
    img5T,img6T = drawlines(img1,img2,lines1,pts1,pts2)
    plt.figure('FTright')
    plt.subplot(121),plt.imshow(img5T)
    plt.subplot(122),plt.imshow(img6T)
    # On calcul de nouvelles lignes epipolaire du cotés gauche de l'image FT que l'on affichent
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,FT)
    lines2 = lines2.reshape(-1,3)
    img3T,img4T = drawlines(img2,img1,lines2,pts2,pts1)
    plt.figure('FTleft')
    plt.subplot(121),plt.imshow(img4T)
    plt.subplot(122),plt.imshow(img3T)
    plt.show()

    # On calcul une rectification pour une camera stereo non calibre
    retval, H1, H2 = cv.stereoRectifyUncalibrated(pts1, pts2, F, (c,r))
    print(H1)
    print(H2)
    # On affiche les different coté de l'objet en changeant l'angle de la camera (Perspective)
    im_dst1 = cv.warpPerspective(img1, H1, (c,r))
    im_dst2 = cv.warpPerspective(img2, H2, (c,r))
    cv.namedWindow('left', 0)
    cv.imshow('left', im_dst1)
    cv.namedWindow('right', 0)
    cv.imshow('right', im_dst2)
    cv.waitKey(0)

if __name__ == "__main__":

    # 1. Recuperer la matrice de calibration de son pc    

    cameraMatrix = CameraCalibration()

    cameraMatrix = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])
    dist = [[0, 0, 0, 0, 0]]

    pts1, pts2, F, maskF, FT, maskE = StereoCalibrate(cameraMatrix)

    EpipolarGeometry(pts1, pts2, F, maskF, FT, maskE)

    DepthMapfromStereoImages()

