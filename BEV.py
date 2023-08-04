import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


def imshow(img):
    plt.figure()
    if len(img.shape) == 3:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.show()


def crop_zeros(img):
    y, x, c = np.nonzero(img)
    xl, xr = x.min(), x.max()
    yl, yr = y.min(), y.max()
    return img[yl:yr + 1, xl:xr + 1]


def matching_points(frame, next_frame, threshold=10, num_matches=35, nfeatures=6000, show_matches=False, show_kp=False):
    """ finds matching points using sift """

    # detect features and compute descriptors
    sift_parameters = {
        "nfeatures": 500
        # "descriptorType": cv2.CV_8U
        #nOctaveLayers, contrastThreshold, edgeThreshold, sigma
    }
    sift = cv2.SIFT_create(**sift_parameters)
    kp1, des1 = sift.detectAndCompute(frame, None)
    kp2, des2 = sift.detectAndCompute(next_frame, None)

    if show_kp:
        img = frame.copy()
        img = cv2.drawKeypoints(img, kp1, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Detected keypoints", img)

    if len(kp1) == 0 or len(kp2) == 0:
        # print("no good key points")
        return np.zeros((0, 2)), np.zeros((0, 2))

    # create BFMatcher object and Match descriptors.
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:num_matches]
    if show_matches:
        img3 = cv2.drawMatches(frame, kp1, next_frame, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Selected keypoints", img3)
    frame_matches = np.array([kp1[match.queryIdx].pt for match in matches])  # left img matched points
    next_frame_matches = np.array([kp2[match.trainIdx].pt for match in matches])  # right img matched points

    return frame_matches, next_frame_matches


def erosion(img):
    """ erodes the images conotour"""
    mask = img != 0
    erosion_size = 3
    erosion_shape = cv2.MORPH_ELLIPSE

    element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))

    erosion_dst = cv2.erode(mask.astype('uint8'), element)
    return img * erosion_dst


def stitch2images(img_src, img_dst):
    """ stitching two images together using features matching and homography estimation"""
    points_src, points_dst = matching_points(img_src, img_dst, threshold=10, num_matches=35,
                                             nfeatures=1000, show_matches=False, show_kp=False)

    H, masked = cv2.findHomography(points_src, points_dst, cv2.RANSAC, 5.0)

    corner_00 = apply_H_on_point(np.matrix([[0], [0]]), H)
    corner_0n = apply_H_on_point(np.matrix([[img_src.shape[1]], [0]]), H)
    corner_m0 = apply_H_on_point(np.matrix([[0], [img_src.shape[0]]]), H)
    Tx, Ty = corner_00[0, 0], corner_00[1, 0]
    T = np.matrix([[1, 0, -Tx],
                   [0, 1, -Ty],
                   [0, 0, 1]])
    tformed_dims = (int(corner_0n[0, 0] - corner_00[0, 0]), int(corner_m0[1, 0] - corner_00[1, 0]))

    tformed_src = cv2.warpPerspective(img_src, T * H, tformed_dims, flags=cv2.INTER_LINEAR)  # warped image
    tformed_src = erosion(tformed_src)  # cleans the edges of the tformed image
    canvas = np.zeros((img_dst.shape[0] * 3, img_dst.shape[1] * 3, 3)).astype('uint8')
    mid_x, mid_y = img_dst.shape[1], img_dst.shape[0]
    canvas[img_dst.shape[0]:img_dst.shape[0] * 2, img_dst.shape[1]:img_dst.shape[1] * 2] = img_dst
    for i in range(tformed_dims[0]):
        for j in range(tformed_dims[1]):
            if tformed_src[j, i].sum() > 0:
                canvas[j + mid_y + int(np.round(Ty)), i + mid_x + int(np.round(Tx))] = tformed_src[j, i]

    return crop_zeros(canvas)


def find_intersection_point(line1, line2):
    """Implementation is based on code from https://stackoverflow.com/questions/46565975, Original author: StackOverflow contributor alkasm
    Find an intercept point of 2 lines model
    Args: line1,line2: 2 lines using rho and theta (polar coordinates) to represent
    Return: x0,y0: x and y for the intersection point
    """
    # rho and theta for each line
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    # Use formula from https://stackoverflow.com/a/383527/5087436 to solve for intersection between 2 lines
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    det_A = np.linalg.det(A)
    if det_A != 0:
        x0, y0 = np.linalg.solve(A, b)
        # Round up x and y because pixel cannot have float number
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return x0, y0
    else:
        return None


def find_dist_to_line(point, line):
    """ Find an intercept point of the line model with a normal from point to it, to calculate the distance betwee point and intercept
    Args: point: the point using x and y to represent
    line: the line using rho and theta (polar coordinates) to represent
    Return: dist: the distance from the point to the line
    """
    x0, y0 = point
    rho, theta = line[0]
    m = (-1 * (np.cos(theta))) / np.sin(theta)
    c = rho / np.sin(theta)
    # intersection point with the model
    x = (x0 + m * y0 - m * c) / (1 + m ** 2)
    y = (m * x0 + (m ** 2) * y0 - (m ** 2) * c) / (1 + m ** 2) + c
    dist = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    return dist


def RANSAC(lines, ransac_iterations, ransac_threshold, ransac_ratio):
    """ Uses RANSAC to identify the vanishing points for a given image's lines
    Args: lines: The lines for the image
    ransac_iterations,ransac_threshold,ransac_ratio: RANSAC hyperparameters
    Return: vanishing_point: Estimated vanishing point for the lines
    """
    inlier_count_ratio = 0.
    vanishing_point = (0, 0)
    # perform RANSAC iterations for each set of lines
    for iteration in range(ransac_iterations):
        # randomly sample 2 lines
        n = 2
        selected_lines = random.sample(lines, n)
        line1 = selected_lines[0]
        line2 = selected_lines[1]
        intersection_point = find_intersection_point(line1, line2)
        if intersection_point is not None:
            # count the number of inliers num
            inlier_count = 0
            # inliers are lines whose distance to the point is less than ransac_threshold
            for line in lines:
                # find the distance from the line to the point
                dist = find_dist_to_line(intersection_point, line)
                # check whether it's an inlier or not
                if dist < ransac_threshold:
                    inlier_count += 1

            # If the value of inlier_count is higher than previously saved value, save it, and save the current point
            if inlier_count / float(len(lines)) > inlier_count_ratio:
                inlier_count_ratio = inlier_count / float(len(lines))
                vanishing_point = intersection_point

            # We are done in case we have enough inliers
            if inlier_count > len(lines) * ransac_ratio:
                break
    return vanishing_point


def get_vpoint(img, idx, data_dir):
    """ Estimating vanishing point using the following pipelineL
    1. calculate the image edges map
    2. use hough lines to fit lines
    3. drop lines that are not in between -valid_angle to +valid_angle
    4. use RANSAC to fit the vanishing point from the line's intersections
    Args: img
    Return: vanishing_point: Estimated vanishing point for the image
    """
    canny_low = 100
    canny_high = 500
    canny_apertureSize = 3

    hough_rho = 1
    hough_theta = (np.pi / 180) * 1
    hough_thresh = 150  # 150

    valid_angle = 45 

    ransac_iterations = 200
    ransac_threshold = 10
    ransac_ratio = 0.9

    edges = cv2.Canny(img, canny_low, canny_high, apertureSize=canny_apertureSize)
    # plt.figure()
    # plt.imshow(edges, cmap='gray')

    hlines = cv2.HoughLines(image=edges, rho=hough_rho, theta=hough_theta, threshold=hough_thresh)

    valid_lines = []  # Useful lines for applying RANSAC
    for line in hlines:
        rho, theta = line[0]
        if (theta > np.deg2rad(180 - valid_angle) and theta < np.deg2rad(180)) or (theta > np.deg2rad(0) and theta < np.deg2rad(valid_angle)):
            valid_lines.append(line)
    if len(valid_lines) == 0:
        print("no valid lines")
        plt.show()
        return None
    img_with_lines = img.copy()
    for i in range(0, len(valid_lines)):
        rho = valid_lines[i][0][0]
        theta = valid_lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 10000 * (-b)), int(y0 + 10000 * (a)))
        pt2 = (int(x0 - 10000 * (-b)), int(y0 - 10000 * (a)))
        cv2.line(img_with_lines, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    # plt.figure()
    # plt.imshow(img_with_lines)
    vpoint = RANSAC(valid_lines, ransac_iterations=ransac_iterations, ransac_threshold=ransac_threshold, ransac_ratio=ransac_ratio)
    cv2.imwrite(f"{data_dir}/outputs/edges{idx+1}.png", edges)
    cv2.imwrite(f"{data_dir}/outputs/v_point{idx+1}.png", draw_vpoint(img_with_lines, vpoint))

    return vpoint


def draw_vpoint(img, vpoint):
    if vpoint[1] > 0:
        new_img = cv2.circle(img.copy(), (vpoint[0], vpoint[1] + 100), 5, (255, 0, 0), -1)
    else:
        new_img = np.zeros((img.shape[0] + abs(vpoint[1]) + 100, img.shape[1], 3)).astype('uint8')
        new_img[abs(vpoint[1]) + 100:, 0:, :] = img
        new_img = cv2.circle(new_img, (vpoint[0], 100), 10, (255, 0, 0), -1)
    return new_img


def apply_H_on_point(point, H):
    """ writes the point as homogenous coordinates, applies H then divides by w"""
    if len(point) == 2:
        point = np.concatenate([point, np.matrix([1])])
    point = H * point
    return point[:2] / point[2]


if __name__ == '__main__':
    downsample_factor = 0.3
    data_dir = "data1"
    img45deg = cv2.resize(cv2.imread(f"45deg.jpg"), (0, 0), fx=downsample_factor, fy=downsample_factor)
    images = [cv2.resize(cv2.imread(f"data1/{i+1}.jpg"), (0, 0), fx=downsample_factor, fy=downsample_factor) for i in range(7)]

    # calibration:
    vpoint_pixels = get_vpoint(img45deg, 0, data_dir)
    vp_x_pix, vp_y_pix = vpoint_pixels
    alpha = abs((vp_y_pix - img45deg.shape[0] / 2) / np.tan(np.deg2rad(45)))
    K = np.matrix([[alpha, 0, img45deg.shape[1] / 2],
                   [0, alpha, img45deg.shape[0] / 2],
                   [0, 0, 1]])

    BEVs = []
    for i, img in enumerate(images):
        vpoint_pixels = get_vpoint(img, i, data_dir)
        vp_x_pix, vp_y_pix = vpoint_pixels
        theta = np.arctan(alpha / (vp_y_pix - img.shape[0]/2))
        theta_deg = np.rad2deg(theta)

        R = np.matrix([[1, 0, 0],
                       [0, np.cos(theta), np.sin(theta)],
                       [0, -np.sin(theta), np.cos(theta)]])

        H_rot = K*R*np.linalg.inv(K)

        corner_00 = apply_H_on_point(np.matrix([[0], [0]]), H_rot)
        corner_0n = apply_H_on_point(np.matrix([[img.shape[1] + 1], [0]]), H_rot)
        corner_m0 = apply_H_on_point(np.matrix([[0], [img.shape[0] + 1]]), H_rot)
        corner_mn = apply_H_on_point(np.matrix([[img.shape[1] + 1], [img.shape[0] + 1]]), H_rot)

        # translate the image to fit good in the frame
        Tx, Ty = corner_00[0, 0], corner_00[1, 0]
        T = np.matrix([[1, 0, -Tx],
                     [0, 1, -Ty],
                     [0, 0, 1]])

        H = T * H_rot

        tformed_dims = (int(corner_0n[0, 0] - corner_00[0, 0]), int(corner_m0[1, 0] - corner_00[1, 0]))
        warped = cv2.warpPerspective(img, H, tformed_dims)
        cv2.imwrite(f"{data_dir}/outputs/full_bev_{i + 1}.png", warped)
        warped = warped[int(warped.shape[0]/1.8):, warped.shape[1]//2 - img.shape[1]:warped.shape[1]//2 + img.shape[1]]
        warped = cv2.resize(warped, (0, 0), fx=0.3, fy=0.3)
        cv2.imwrite(f"{data_dir}/outputs/cropped_bev_{i + 1}.png", warped)
        BEVs.append(warped)

    for i, img in enumerate(BEVs[2:]):
        cv2.imshow(f"{i}", img)

    stitched = stitch2images(img_src=BEVs[1], img_dst=BEVs[0])
    for i, img in enumerate(BEVs[2:]):
        cv2.imwrite(f"{data_dir}/outputs/stitched{i}_BEV.png", stitched)
        stitched = stitch2images(img_src=img, img_dst=stitched)

    cv2.imshow('stitched', stitched)
    cv2.imwrite(f"{data_dir}/outputs/full_stitched_BEV.png", stitched)

    cv2.waitKey(0)







