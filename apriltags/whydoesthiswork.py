import copy
import time
import argparse
import numpy as np

import cv2 as cv
from pupil_apriltags import Detector
from tag import Tag

TAG_SIZE = 0.15244
FAMILIES = "tag16h5"

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=1280)
    parser.add_argument("--height", help='cap height', type=int, default=720)

    parser.add_argument("--families", type=str, default=FAMILIES)
    parser.add_argument("--nthreads", type=int, default=4)
    parser.add_argument("--quad_decimate", type=float, default=2.0)
    parser.add_argument("--quad_sigma", type=float, default=0.0)
    parser.add_argument("--refine_edges", type=int, default=1)
    parser.add_argument("--decode_sharpening", type=float, default=0.25)
    parser.add_argument("--debug", type=int, default=0)

    args = parser.parse_args()

    return args

def metersToFeet(meters):
    return meters * 39.3701#* 3.2808399

def main():
    definedTags = Tag(TAG_SIZE, FAMILIES)

    # Add information about tag locations THIS ARE GLOBAL LOCATIONS IN INCHES
    # Function Arguments are id,x,y,z,theta_x,theta_y,theta_z
    definedTags.add_tag(1, 0., 0., 0., 0., 0., 0.)
    definedTags.add_tag(2, 12., 0., 0., 0., 0., 0.)
    definedTags.add_tag(3, 0., 0., 0., 0., 0., 0.)
    definedTags.add_tag(4, 0., 0., 0., 0., 0., 0.)
    definedTags.add_tag(5, 0., 0., 0., 0., 0., 0.)
    definedTags.add_tag(6, 0., 0., 0., 0., 0., 0.)


    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    families = args.families
    nthreads = args.nthreads
    quad_decimate = args.quad_decimate
    quad_sigma = args.quad_sigma
    refine_edges = args.refine_edges
    decode_sharpening = args.decode_sharpening
    debug = args.debug

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    at_detector = Detector(
        families=families,
        nthreads=nthreads,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=refine_edges,
        decode_sharpening=decode_sharpening,
        debug=debug,
    )

    elapsed_time = 1

    while True:
        start_time = time.time()
        ret, image = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        tags = at_detector.detect(
            image,
            estimate_tag_pose=True,
            camera_params=[307, 307, 640, 360],
            tag_size=TAG_SIZE,
        )
        
        detections = []
        pose_x_sum = 0
        pose_y_sum = 0
        pose_z_sum = 0
        for detection in tags:
            if detection.tag_id < 1 or detection.tag_id > 9 or detection.decision_margin < 20:
                continue
            detections.append(detection)
            curPose = definedTags.estimate_pose(detection.tag_id, detection.pose_R, detection.pose_t)
            pose_x_sum += curPose[0][0]
            pose_y_sum += curPose[1][0]
            pose_z_sum += curPose[2][0]

        size = len(detections)
        if size > 0:
            pose = np.array([pose_x_sum/size,pose_y_sum/size,pose_z_sum/size])
            debug_image = draw_tags(debug_image, detections, elapsed_time, pose)

        elapsed_time = time.time() - start_time

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        cv.imshow('AprilTags', debug_image)

    cap.release()
    cv.destroyAllWindows()


def draw_tags(
    image,
    tags,
    elapsed_time,
    pose
):
    for tag in tags:
        tag_id = tag.tag_id
        center = tag.center
        corners = tag.corners

        center = (int(center[0]), int(center[1]))
        corner_01 = (int(corners[0][0]), int(corners[0][1]))
        corner_02 = (int(corners[1][0]), int(corners[1][1]))
        corner_03 = (int(corners[2][0]), int(corners[2][1]))
        corner_04 = (int(corners[3][0]), int(corners[3][1]))

        cv.circle(image, (center[0], center[1]), 5, (0, 0, 255), 2)

        cv.line(image, (corner_01[0], corner_01[1]),
                (corner_02[0], corner_02[1]), (255, 0, 0), 2)
        cv.line(image, (corner_02[0], corner_02[1]),
                (corner_03[0], corner_03[1]), (255, 0, 0), 2)
        cv.line(image, (corner_03[0], corner_03[1]),
                (corner_04[0], corner_04[1]), (0, 255, 0), 2)
        cv.line(image, (corner_04[0], corner_04[1]),
                (corner_01[0], corner_01[1]), (0, 255, 0), 2)

        cv.putText(image, str(tag_id), (center[0] - 10, center[1] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv.LINE_AA)

    fps = round(1.0 / elapsed_time)
    cv.putText(image,
               "FPS:" + '{:.1f}'.format(fps),
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
               cv.LINE_AA)
    cv.putText(image,
               ("Pose: " + str(round(metersToFeet(pose[0]),3)) + " " + str(round(metersToFeet(pose[1]),3)) + " " + str(round(metersToFeet(pose[2]),3))),
               (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
               cv.LINE_AA)
    

    return image


if __name__ == '__main__':
    main()