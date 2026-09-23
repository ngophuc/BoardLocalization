#!/usr/bin/env python3
# Code for the paper: An image processing method to recognize position of sawn boards within the log
# Authors: X. Li, G. Pot, P. Ngo, J. Viguier, H. Penvern

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import rescale, rotate
import pandas as pd
import time
import os

def preprocessSIFT(img):
    img=cv2.GaussianBlur(img,(5,5),0)
    clahe = cv2.createCLAHE()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img =clahe.apply(img)
    return (img)

def rotate_coordiate(img,angle_mean):#(angle,rect,center):
    imgHeight, imgWidth = img.shape[0], img.shape[1]
    centreY, centreX = imgHeight//2, imgWidth//2
    rotationMatrix = cv2.getRotationMatrix2D(center=(centreX, centreY),angle=-angle_mean,scale=1.0)
    # Now will take out sin and cos values from rotationMatrix
    # Also used numpy absolute function to make positive value
    cosofRotationMatrix = np.abs(rotationMatrix[0][0])
    sinofRotationMatrix = np.abs(rotationMatrix[0][1])
    # Now will compute new height & width of
    # an image so that we can use it in
    # warpAffine function to prevent cropping of image sides
    newImageWidth= int((imgHeight * sinofRotationMatrix) +
                     (imgWidth * cosofRotationMatrix))
    newImageHeight= int((imgHeight * cosofRotationMatrix) +
                    (imgWidth * sinofRotationMatrix))
    # After computing the new height & width of an image
    # we also need to update the values of rotation matrix
    rotationMatrix[0][2] += (newImageWidth/2) - centreX
    rotationMatrix[1][2] += (newImageHeight/2) - centreY
    rotated_image = cv2.warpAffine(src=img, M=rotationMatrix,dsize=(newImageWidth, newImageHeight),borderValue=(255,255,255))# dsize=(img.shape[1],img.shape[0])
    return (rotated_image)

def __SIFT_FLANN(template,img):
    # img=cv2.GaussianBlur(img,(5,5),0)
    # template=cv2.GaussianBlur(template,(5,5),0)
    template = preprocessSIFT(template)
    img = preprocessSIFT(img)

    sift = cv2.xfeatures2d.SIFT_create()
    #Initialize and use FLANN
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    MIN_MATCH_COUNT =10
    
    kp1, des1 = sift.detectAndCompute(template,None)
    kp2, des2 = sift.detectAndCompute(img,None)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    angle =float('nan')
    dst = []
    scale = []
    for m, n in matches:
        if m.distance < 0.8* n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        # Estimate homography between template and scene
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)# coordinate matrix of points in the source plane
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)# Coordinate matrix of points in the target plane
        M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]# M: transformation matrix between two planes
        if M is None:
            print("M is not found")
        else:
            # Draw detected template in scene image
            h, w = template.shape[0],template.shape[1]
            pts = np.float32([[0, 0],
                              [0, h - 1],
                              [w - 1, h - 1],
                              [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            dst_location = dst.reshape(4,2)
            y_coordinates = dst_location[0,0]-dst_location[3,0]
            x_coordinates = dst_location[0,1]-dst_location[3,1]
            angle = np.arctan2(y_coordinates,x_coordinates)* 180 / np.pi #np.arctan2(slope)
            angle = angle + 90
            if angle < 0 and angle !=np.nan:
                angle = 180+angle
            if angle >180:
                angle = angle -180
                
            '''determine rotation scale'''
            height1 = np.linalg.norm(dst_location[0,:] -dst_location[1,:])
            height2 = np.linalg.norm(dst_location[2,:] -dst_location[3,:])
            width1 = np.linalg.norm(dst_location[0,:] -dst_location[3,:])
            width2 = np.linalg.norm(dst_location[1,:] -dst_location[2,:])
            height = (height1+height2)/2
            width = (width1+width2)/2
            scale = min(height/template.shape[0],width/template.shape[1])#[height/template.shape[0],width/template.shape[1]]#
    return (angle,scale,dst)

def tu_four_sides(dst_location):
    #Coord of the four points of the board
    [x1,y1] =dst_location[0,:] # A
    [x2,y2] = dst_location[1,:] # B
    [x3,y3] = dst_location[2,:] # C
    [x4,y4] = dst_location[3,:] # D
    # Point BD on both sides of line segment AC
    z1 = (x2-x1)*(y3-y1) -(x3-x1)*(y2-y1) 
    z2 = (x4-x1)*(y3-y1)-(x3-x1)*(y4-y1)
    
    z3 = (x1-x2)*(y4-y2)-(x4-x2)*(y1-y2)
    z4 = (x3-x2)*(y4-y2) -(x4-x2)*(y3-y2) 

    if (z1*z2) <0 and z3*z4 <0:
        val = 1
    else:
        val = 0
    return (val)

def preprocessTM(img):
    img=cv2.GaussianBlur(img,(5,5),0)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE()#(clipLimit=0, tileGridSize=(8,8))
    img = clahe.apply(img)
    return (img)

def template_match(img,template):
    img = preprocessTM(img)
    template_flip = np.flip(template)
    template = preprocessTM(template)
    template_flip = preprocessTM(template_flip)
    methods = ["cv2.TM_CCOEFF_NORMED"]
    #["cv2.TM_CCOEFF","cv2.TM_CCOEFF_NORMED", "cv2.TM_CCORR", "cv2.TM_CCORR_NORMED","cv2.TM_SQDIFF","cv2.TM_SQDIFF_NORMED"]
    width,height = template_flip.shape[::-1]
    
    #create a for loop, using the eval() function
    for m in methods:
        method = eval(m)
        img_copy = img.copy()
        res1= cv2.matchTemplate(img_copy,template,method)
        res2= cv2.matchTemplate(img_copy,template_flip,method)
    # find the maximum and minimum values of the resulting map, as well as the max and min value locations
    # use the function cv2.minMaxLoc()
        min_val, max_val, min_loc,max_loc = cv2.minMaxLoc(res1)
        min_val2, max_val2, min_loc2,max_loc2 = cv2.minMaxLoc(res2)
        
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            if min_val2 < min_val:
                min_val, max_val, min_loc,max_loc = min_val2, max_val2, min_loc2,max_loc2
        else: 
            if max_val2 > max_val:
                min_val, max_val, min_loc,max_loc = min_val2, max_val2, min_loc2,max_loc2
    # SqDiff and SqDiffNormed: the minimum value will be considered the match
    # for other methods: the match will be located where the function finds the maximum value
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            best_value = min_val
            top_left = min_loc
        else:
            best_value=max_val
            top_left =max_loc
            #width,height = template.shape[::-1]
            
    bottom_right = (top_left[0]+width,top_left[1] + height)
    return(best_value,top_left,bottom_right,res1,res2)
def remove_bound(data):
    not_row = data[[not np.all(data[i,:,:]==255) for i in range(data.shape[0])],:,:]
    bot_col = not_row[:,[not np.all(not_row[:,i,:]==255) for i in range(not_row.shape[1])],:]
    return (bot_col)

if __name__ == '__main__':
    dirs = 'face2/'
    data = pd.read_excel('number_new.xlsx', usecols=[0,1], names=None) 
                  
    log_nameT = pd.unique(data["log_number"])
    angle_log = np.zeros((len(log_nameT),1))*np.nan
    time_begin = time.time()
    angleTotal = []
    Point_A = []
    Point_D= []
    #Undetection =[]
    scale_factor = 4
    for log in range(1):#(len(log_nameT)):
        log = 0
        time_begin1 = time.time()
        log_name = log_nameT[log]
        name_list =['log',str(log_name)]
        a = ''
        log_name2 = a.join(name_list)
        print("log_name=", log_name2)
        img_path = os.path.join(dirs,'{}.png'.format(log_name2))
        img=cv2.imread(img_path,-1)
        imgSIFT = img.copy()
        boards  = data.loc[data["log_number"]==log_name,["board_number"]]
        num_boards= len(boards)
        num_mark0 = [boards.iloc[i].values[0] for i in range (num_boards)]
        angleT = []
        scaleT = []
        xx = 0
        for j in range(num_boards):
            patch_name =boards.iloc[j].values[0]
            print("board_name=", patch_name)
            tem_path = os.path.join(dirs,'{}.png'.format(patch_name))
            template = cv2.imread(tem_path,-1)
            h,w = template.shape[0],template.shape[1]
            template = cv2.resize(template,dsize=(int(w/scale_factor),int(h/scale_factor)))
            angle,scale,dst = __SIFT_FLANN(template,imgSIFT)

            if np.isnan(angle)or dst.min() <= 0 or dst.max()>=imgSIFT.shape[1]:
                print ("Board is not detected")
                continue
            else:
                dst_location = dst.reshape(4,2)
                if dst_location.min() > 0:
                    lm=tu_four_sides(dst_location)
                    if lm == 1:
                        xx = 1
                        label_pointh=eval('(abs(dst_location[0,0]+dst_location[2,0]))*0.5-80')
                        label_pointw= eval('(abs(dst_location[0,1]+dst_location[2,1]))*0.5+10')
                        label0  = (int(label_pointh),int(label_pointw))
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        imgSIFT = cv2.polylines(imgSIFT, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
                        cv2.putText(imgSIFT,str(patch_name),label0, font,3,(255,255,255),3)
                        angleT.append(angle)
                        scaleT.append(scale)
        if xx == 1: # means there are boards that can be detected by SIFT
            print("found result for log:", log_name, angleT, scaleT)
            listBins = list(range(0,181,5))
            angle_cut = pd.cut(angleT,bins=listBins)
            pf = pd.value_counts(angle_cut)
            pf_cut =pf.index.values[0]
            angle_cut = [k1 for k1 in angleT if k1>pf_cut.left and k1<=pf_cut.right]
            scale_cut = [scaleT[s] for s in range(len(scaleT)) if angleT[s]>pf_cut.left and angleT[s]<=pf_cut.right]
            angle_mean  = np.mean(angle_cut)
            scale_mean  = np.mean(scale_cut)
            rotated_image = rotate_coordiate(img,angle_mean)
            rotated_image = remove_bound(rotated_image)
            
            Score_total = []
            A_total = []
            D_total = []
            imgPlot =rotated_image.copy()
            for k2 in range(len(num_mark0)):#
                patch_name =num_mark0[k2]#boards_dect[k2]
                tem_path = os.path.join(dirs,'{}.png'.format(patch_name))
                template2=cv2.imread(tem_path,-1)
                h,w = template2.shape[0],template2.shape[1]
                template2 = cv2.resize(template2,dsize=(int(w//scale_factor),int(h//scale_factor)))
                # template2=cv2.GaussianBlur(template2,(3,3),0)
                h0,w0 = template2.shape[0],template2.shape[1]
                template2 = cv2.resize(template2,dsize=(int(w0*scale_mean),int(h0*scale_mean)))
                template2_rotated = np.rot90(template2)
                best_val,A_top,D_Bott,res1,res2 = template_match(rotated_image,template2)
                best_val2,A_top2,D_Bott2,res13,res23 = template_match(rotated_image, template2_rotated)
                if best_val2 > best_val:
                    best_val = best_val2
                    A_top = A_top2
                    D_Bott = D_Bott2
                A_total.append(A_top)
                D_total.append(D_Bott)
                Score_total.append(best_val)

            for j in range(len(num_mark0)):
                patch_nameF =num_mark0[j]
                A_top = A_total[j]
                D_Bott = D_total[j]
                label_pointh=eval('(A_top[0]+D_Bott[0])*0.5-50')
                label_pointw= eval('(A_top[1]+D_Bott[1])*0.5+10')
                label0  = (int(label_pointh),int(label_pointw))
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.rectangle(imgPlot,A_top,D_Bott,(0,0,255),3)
                cv2.putText(imgPlot,str(patch_nameF),label0, font,3,(0,0,255),3)

            #Save output image
            patha = dirs3 + '/'  +'SIFT_face1_log'+str(log+1)+'.png'
            cv2.imwrite(patha,imgSIFT)
            path2 = dirs3 + '/'  +'rotation_face1_log'+str(log+1)+'.png'
            cv2.imwrite(path2,rotated_image)
            del rotated_image
            plt.imshow(cv2.cvtColor(imgPlot,cv2.COLOR_BGR2RGB))
            plt.xticks(())
            plt.yticks(())
            plt.xlabel(log_name2,fontsize=24)
            plt.title(log_name2,fontsize=24)
            angleTotal.append([log_name,angle_mean,scale_mean])
            #Save output image
            path3 = dirs3 + '/' +'log'+str(log+1)+'face1_'+str(angle_mean)+'_'+str(scale_mean)+'.png'
            cv2.imwrite(path3,imgPlot)
        else:
            print("No boards are detected by SIFT log-%d"%(log_name))
        Point_A =Point_A  + A_total
        Point_D =Point_D  + D_total
        angleTotal10 = angleTotal

        time_end1 = time.time()
        time1 = time_end1 - time_begin1
        result_SIFT = np.append(angleT,scaleT)
        result_SIFT = np.append(result_SIFT,time1)
        # pathb = dirs3 + '/'+'log'+str(log+1)+'face1' + '.txt'
        # np.savetxt(pathb,result_SIFT,delimiter=',')

    Point_A = np.array(Point_A)
    Point_D = np.array(Point_D)

    time_end = time.time()
    timeT = time_end - time_begin
    print("running time=", timeT)
    plt.figure()
    plt.imshow(cv2.cvtColor(imgSIFT,cv2.COLOR_BGR2RGB))
    path4 = dirs3 + '/'+ 'Total_face1'
    np.savetxt(path4,Point_A=Point_A,Point_D=Point_D,angleTotal = angleTotal,timeT=timeT)
