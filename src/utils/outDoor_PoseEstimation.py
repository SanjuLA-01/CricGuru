import os
import cv2
import time
import numpy as np


def outDoor_PoseEstimation_F(assets_dir):

    MODE = "COCO"

    if MODE == "COCO":
        protoFile = os.path.join(assets_dir, 'pose', 'pose_deploy_linevec.prototxt')
        weightsFile = os.path.join(assets_dir, 'pose', 'pose_iter_440000.caffemodel')
        nPoints = 18
        POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                      [11, 12], [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]

    elif MODE == "MPI":
        protoFile = ""
        weightsFile = ""
        nPoints = 15
        POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10],
                      [14, 11], [11, 12], [12, 13]]

    threshold = 0.2

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    # input image dimensions for the network
    inWidth = 368
    inHeight = 368

    # Directory containing the input images
    input_dir = os.path.join(assets_dir, 'videooutput')

    # Directory to save the output images
    output_dir = os.path.join(assets_dir, 'output')

    for filename in os.listdir(input_dir):
        frame = cv2.imread(os.path.join(input_dir, filename))
        frameCopy = np.copy(frame)
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        t = time.time()

        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                        (0, 0, 0), swapRB=False, crop=False)

        net.setInput(inpBlob)

        output = net.forward()
        #print("time taken by network : {:.3f}".format(time.time() - t))

        H = output.shape[2]
        W = output.shape[3]

        # Empty list to store the detected keypoints
        points = []

        for i in range(nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > threshold:
                cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else:
                points.append(None)

                # Draw Skeleton
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (255, 255, 255), 10)
                cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        # Resize the output image
        output_resized = cv2.resize(frame, (500, 500))

        cv2.imwrite(os.path.join(output_dir, filename), output_resized)