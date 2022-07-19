
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import plot.plotting_params as params
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import glob
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from skimage import img_as_ubyte
import imageio

# from lib.util import get_correlation, combine
# from lib.math_3d import project_3d_points_in_4D_format, project_3d_corners
# from lib.core import iou
# from lib.core import iou3d
from lib.helpers.file_io import read_csv, pickle_read, pickle_write

sub_index = 1

def savefig(plt, path, show_message= True, tight_flag= True, pad_inches= 0, newline= True):
    if show_message:
        print("Saving to {}".format(path))
    if tight_flag:
        plt.savefig(path, bbox_inches='tight', pad_inches= pad_inches)
    else:
        plt.savefig(path)
    if newline:
        print("")

def diverge_map(top_color_frac=(1.0, 0.0, 0.25), bottom_color_frac=(1.0, 0.906, 0.04)):
    '''
    top_color_frac and bottom_color_frac are colors that will be used for the two
    ends of the spectrum.
    Reference
    https://towardsdatascience.com/creating-colormaps-in-matplotlib-4d4de78a04b8
    '''
    # import matplotlib.colors as mcolors
    from matplotlib.colors import ListedColormap

    N = 128
    top_color = np.ones((N, 4))
    top_color[:, 0] = np.linspace(top_color_frac[0], 1, N) # R
    top_color[:, 1] = np.linspace(top_color_frac[1], 1, N) # G
    top_color[:, 2] = np.linspace(top_color_frac[2], 1, N) # B
    top_color_cmp = ListedColormap(top_color)

    if bottom_color_frac is not None:
        bottom_color = np.ones((N, 4))
        bottom_color[:, 0] = np.linspace(bottom_color_frac[0], 1, N)
        bottom_color[:, 1] = np.linspace(bottom_color_frac[1], 1, N)
        bottom_color[:, 2] = np.linspace(bottom_color_frac[2], 1, N)
        bottom_color_cmp   = ListedColormap(bottom_color)

        newcolors2 = np.vstack((bottom_color_cmp(np.linspace(0, 1, 128)), top_color_cmp(np.linspace(1, 0, 128))))
    else:
        newcolors2 = top_color_cmp(np.linspace(1, 0, 128))
    double = ListedColormap(newcolors2, name='double')

    return double


def parse_predictions(folder, before= True, return_score= True):
    if before:
        file = "predictions_bef_nms.npy"
    else:
        file = "predictions.npy"
    pred_path       = os.path.join(folder, file)
    print("Loading {}...".format(pred_path))
    predictions     = np.load(pred_path)

    # Drop the label column and convert the numpy array to float
    predictions     = predictions[:, sub_index:].astype(float)
    score           = predictions[:, 15 - sub_index]
    h2d_pred_all    = predictions[:, 17 - sub_index]
    h2d_general_all = predictions[:, 18 - sub_index]
    h2d_special_all = predictions[:, 19 - sub_index]

    return h2d_pred_all, h2d_general_all, score

def get_output_file_path(prefix, postfix="", relative= False, threshold= 0, before_nms_flag= True, folder= None):
    output_image_file = prefix
    if relative:
        output_image_file += '_rel'
    output_image_file += postfix

    if before_nms_flag:
        output_image_file += '_on_all_predictions'

    if threshold > 0:
        output_image_file += '_class_conf_gt_' + str(threshold)

    output_image_file += '.png'

    if folder is not None:
        path = os.path.join(folder, output_image_file)

    return path

def throw_samples(x, y, frac_to_keep= 0.9, show_message= True, throw_around_center= True):
    """
        Throws the outliers based on sorting
        frac = 0.75 suggests keep the first three quarters of the data and throw
        away the remaining data. Keep the elements belonging to the valid data
    """
    if show_message:
        print("Using {:.2f}% of the data".format(frac_to_keep * 100.0))
    samples_to_keep = int(x.shape[0] * frac_to_keep)

    # Sort the x array and get the indices
    sorted_indices = np.abs(x).argsort()
    # Keep the indices which are required
    if throw_around_center:
        center_index = sorted_indices.shape[0]//2
        keep_index   = sorted_indices[(center_index-samples_to_keep//2) : (center_index + samples_to_keep//2)+1]
    else:
        keep_index   = sorted_indices[0:samples_to_keep]

    # Use the same index for x and y
    x = x[keep_index]
    y = y[keep_index]

    return x, y

def get_bins(x, num_bins):
    # sort the data
    x_min = np.min(x)
    x     = np.sort(x)

    pts_per_bin = int(np.ceil(x.shape[0]/(num_bins)))
    # print("Num bins= {}, pts per bin  = {}".format(num_bins, pts_per_bin))
    # bins contain a lower bound. so 1 extra element
    bins  = np.zeros((num_bins+1, ))

    bins[0] = x_min
    for i in range(1,bins.shape[0]):
        if i*pts_per_bin < x.shape[0]:
            end_ind = i*pts_per_bin
        else:
            end_ind = x.shape[0]-1
        bins[i] = x[end_ind]

    return bins

def draw_rectangle(ax, rect_x_left, rect_y_left, rect_width, rect_height, img_width= 100, img_height= 100, edgecolor= 'r', linewidth= params.lw):
    """
    Draw a rectangle on the image
    :param ax:
    :param img_width:
    :param img_height:
    :param rect_x_left:
    :param rect_y_left:
    :param rect_width:
    :param rect_height:
    :param angle:
    :return:
    """

    x = rect_x_left
    y = rect_y_left
    # Rectangle patch takes the following coordinates (xy, width, height)
    # Reference https://matplotlib.org/api/_as_gen/matplotlib.patches.Rectangle.html#matplotlib.patches.Rectangle
    # :                xy------------------+
    # :                |                  |
    # :              height               |
    # :                |                  |
    # :                ------- width -----+
    # Create a Rectangle patch
    rect = patches.Rectangle((x, y), width= rect_width, height= rect_height, linewidth= linewidth, edgecolor= edgecolor, facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    return ax

def draw_circle(ax, x, y, radius= 6, color= 'r', edgecolor= None):
    circle1 = plt.Circle((x, y), radius, color= color)
    ax.add_artist(circle1)

def get_left_point_width_height(gts):
    rect_x_left  = gts[0]
    rect_y_left  = gts[1]
    rect_x_right = gts[2]
    rect_y_right = gts[3]
    rect_width   = rect_x_right - rect_x_left
    rect_height  = rect_y_right - rect_y_left

    return rect_x_left, rect_y_left, rect_width, rect_height

ground_truth_folder   = "data/kitti_split1/validation/label_2/"
p2_folder             = "/home/abhinav/project/mono_object_detection_July_09/output/run_4_on_refact_1/results/results_test/p2"
error_list_pkl        = "error_list_2.pkl"
min_iou2d_overlap     = 0.4
display_frequency     = 250
num_bins              = 25
frac_to_keep          = 0.7

def plot_one_error_with_box_error(y, error_in_box, name="y", plt= None, show_message= False, throw_samples_flag= True, throw_samples_in_bin= False, iou_on_x= True, do_decoration= True, label= "", color= None, bins= None):
    error_in_box = np.abs(error_in_box)

    if throw_samples_flag:
        # Throw the outliers
        error_in_box, y = throw_samples(error_in_box, y, show_message= show_message)

    if bins is None:
        # Get bins on the data
        bins     = get_bins(error_in_box, num_bins)
    xplot      = []
    yplot_mean = []
    yplot_std  = []

    for j in range(0, bins.shape[0] - 1):
        index         = np.logical_and((error_in_box >= bins[j]), (error_in_box < bins[j + 1]))
        if y[index].shape[0] > 0 :
            x_final = error_in_box[index]
            y_final = y[index]
            if throw_samples_in_bin:
                x_final, y_final = throw_samples(x_final, y_final, frac_to_keep= frac_to_keep, show_message= show_message, throw_around_center= True)
            xplot.append(np.mean(x_final))
            yplot_mean.append(np.mean(y_final))
            yplot_std .append(np.std (y_final))

    xplot = np.array(xplot)
    yplot_mean = np.array(yplot_mean)
    yplot_std  = np.array(yplot_std )

    # Remember to compute correlation on the original data and not the binned ones
    correlation = get_correlation(error_in_box, y)
    print(correlation)
    if color is None:
        color = "dodgerblue"
    #plt.scatter(xplot, yplot_mean, color= params.color2, s= 4*params.ms, label= r"Scale", lw= params.lw)

    # diff = bins.shape[0] - yplot_mean.shape[0]  - 1
    # xtemp = (bins[:-1] + bins[1:])/2
    # xtemp = xtemp[:-diff]
    plt.plot(xplot, yplot_mean, color= color, lw= params.lw, label= label + ", Corr={:.3f}".format(correlation))

    if do_decoration:
        plt.grid(True)
        if iou_on_x:
            plt.xlim((0, 1.005))
        else:
            plt.xlim(left= 0.0)
        plt.ylim(bottom= 0.7, top= 1.05)
        plt.ylabel(name)
        plt.xlabel(r"$IoU_{3D}$")
        plt.title(r"Corr = {:.2f}".format())

def plot_one_set_of_error_plots_with_box_error(error_variable_x_axis, z3d_gt, score, error_in_x1, error_in_y1, error_in_w2d, error_in_h2d, error_in_x3d, error_in_y3d, error_in_l3d, error_in_w3d, error_in_h3d, error_in_rot, error_in_alpha, error_general, error_general_with_gt, error_in_x3d_2d= None, error_in_y3d_2d= None, error_in_z3d_2d= None, before_nms_flag= True, relative= False, postfix="_z_error", save_path= None):
    if relative:
        print("Doing relative error in z")
        error_variable_x_axis /= z3d_gt

    iou_on_x= True
    throw_samples_flag= False

    if error_in_x3d_2d is None or error_in_y3d_2d is None or error_in_z3d_2d is None:
        plot_more = False
    else:
        plot_more = True

    width = 30
    height = 16
    rows = 4
    cols = 5

    plt.figure(figsize= params.size, dpi= params.DPI)
    plt.subplot(rows, cols, 1)
    plot_one_error_with_box_error(score, error_variable_x_axis, name=r"class conf ", plt= plt, show_message= True, throw_samples_flag= throw_samples_flag, iou_on_x= iou_on_x)
    plt.subplot(rows, cols, 2)
    plot_one_error_with_box_error(error_in_x1, error_variable_x_axis, name=r"err_x1", plt= plt, throw_samples_flag= throw_samples_flag, iou_on_x= iou_on_x)
    plt.subplot(rows, cols, 3)
    plot_one_error_with_box_error(error_in_y1, error_variable_x_axis, name=r"err_y1", plt= plt, throw_samples_flag= throw_samples_flag, iou_on_x= iou_on_x)
    plt.subplot(rows, cols, 4)
    plot_one_error_with_box_error(error_in_w2d, error_variable_x_axis, name=r"err_w2d", plt= plt, throw_samples_flag= throw_samples_flag, iou_on_x= iou_on_x)
    plt.subplot(rows, cols, 5)
    plot_one_error_with_box_error(error_in_h2d, error_variable_x_axis, name=r"err_h2d", plt= plt, throw_samples_flag= throw_samples_flag, iou_on_x= iou_on_x)

    plt.subplot(rows, cols, 6)
    plot_one_error_with_box_error(error_in_x3d, error_variable_x_axis, name=r"err_x3d", plt= plt, throw_samples_flag= throw_samples_flag, iou_on_x= iou_on_x)
    plt.subplot(rows, cols, 7)
    plot_one_error_with_box_error(error_in_y3d, error_variable_x_axis, name=r"err_y3d", plt= plt, throw_samples_flag= throw_samples_flag, iou_on_x= iou_on_x)

    if plot_more:
        plt.subplot(rows, cols, 8)
        plot_one_error_with_box_error(error_in_x3d_2d, error_variable_x_axis, name=r"err_x3d_2d", plt=plt, throw_samples_flag= throw_samples_flag, iou_on_x= iou_on_x)
        plt.subplot(rows, cols, 9)
        plot_one_error_with_box_error(error_in_y3d_2d, error_variable_x_axis, name=r"err_y3d_2d", plt=plt, throw_samples_flag= throw_samples_flag, iou_on_x= iou_on_x)
        plt.subplot(rows, cols, 10)
        plot_one_error_with_box_error(error_in_z3d_2d, error_variable_x_axis, name=r"err_z3d_2d", plt=plt, throw_samples_flag= throw_samples_flag, iou_on_x= iou_on_x)

    shift = 3

    plt.subplot(rows, cols, 8+ shift)
    plot_one_error_with_box_error(error_in_l3d, error_variable_x_axis, name=r"err_l3d", plt= plt, throw_samples_flag= throw_samples_flag, iou_on_x= iou_on_x)
    plt.subplot(rows, cols, 9 + shift)
    plot_one_error_with_box_error(error_in_w3d, error_variable_x_axis, name=r"err_w3d", plt= plt, throw_samples_flag= throw_samples_flag, iou_on_x= iou_on_x)
    plt.subplot(rows, cols, 10 + shift)
    plot_one_error_with_box_error(error_in_h3d, error_variable_x_axis, name=r"err_h3d", plt= plt, throw_samples_flag= throw_samples_flag, iou_on_x= iou_on_x)
    plt.subplot(rows, cols, 11 + shift)
    plot_one_error_with_box_error(error_in_rot, error_variable_x_axis, name=r"err_rot", plt= plt, throw_samples_flag= throw_samples_flag, iou_on_x= iou_on_x)
    plt.subplot(rows, cols, 12 + shift)
    plot_one_error_with_box_error(error_in_alpha, error_variable_x_axis, name=r"err_alp", plt= plt, throw_samples_flag= throw_samples_flag, iou_on_x= iou_on_x)

    plt.subplot(rows, cols, 13 + shift)
    plot_one_error_with_box_error(error_general, error_variable_x_axis, name=r"err_gen", plt= plt, throw_samples_flag= throw_samples_flag, iou_on_x= iou_on_x)
    plt.subplot(rows, cols, 14 + shift)
    plot_one_error_with_box_error(error_general_with_gt, error_variable_x_axis, name=r"err_gen_gt", throw_samples_flag= throw_samples_flag, plt= plt, iou_on_x= iou_on_x)

    if save_path is None:
        save_path = get_output_file_path(prefix= "pred_error_vs", postfix= "", relative= relative, threshold= threshold_score,
                                                before_nms_flag= before_nms_flag, folder= params.IMAGE_DIR)
    savefig(plt, save_path)
    plt.close()

def read_folder_and_get_all_errors(input_folder, prediction_folder_relative, num_images= -1, num_predictions_boxes= -1, threshold_score= 0.0, threshold_depth= 100):
    full_folder_path = os.path.join(input_folder, prediction_folder_relative)
    error_list_file_path = os.path.join(input_folder, error_list_pkl)

    print("\n=========================================================================")
    print("Processing {}...".format(full_folder_path))
    print("=========================================================================")
    if os.path.exists(error_list_file_path):
        # Read from cache
        big_list = pickle_read(error_list_file_path)
        return big_list
    else:
        # Process and create the cache
        print("{} not found".format(error_list_file_path))

    predictions_all = None
    gts_all = None
    iou_3d_all = None

    prediction_files = sorted(glob.glob(os.path.join(input_folder, prediction_folder_relative + "/*.txt")))
    num_prediction_files = len(prediction_files)

    if num_images < 0:
        num_images = num_prediction_files
    print("Choosing {} files out of {} prediction files for plotting".format(num_images, num_prediction_files))
    if num_predictions_boxes > 0:
        print("Taking top {} boxes per image...".format(num_predictions_boxes))
    file_index = np.sort(np.random.choice(range(num_prediction_files), num_images, replace=False))

    for i in range(num_images):
        filename = prediction_files[file_index[i]]
        basename = os.path.basename(filename)
        ground_truth_file_path = os.path.join(ground_truth_folder, basename)

        p2_npy_file = os.path.join(p2_folder, basename.replace(".txt", ".npy"))
        p2 = np.load(p2_npy_file)

        predictions_img = read_csv(filename, ignore_warnings= True)
        gt_img = read_csv(ground_truth_file_path,  ignore_warnings= True)

        if predictions_img.size > 0 and gt_img is not None:

            # Add dimension if there is a single point
            if gt_img.ndim == 1:
                gt_img = gt_img[np.newaxis, :]

            if predictions_img.ndim == 1:
                predictions_img = predictions_img[np.newaxis, :]

            if num_predictions_boxes > 0:
                predictions_img = predictions_img[:num_predictions_boxes]

            # Remove labels
            predictions_img = predictions_img[:, 1:]
            gt_img = gt_img[:, 1:]

            #        0   1    2     3   4   5  6    7    8    9    10   11   12   13    14      15     16     17
            # (cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score, width, height, h2d_general  )
            # Add projected 3d center information to predictions_img
            predictions_centers_3d = predictions_img[:, 10:13].T
            predictions_centers_3d_2d = project_3d_points_in_4D_format(p2, predictions_centers_3d, pad_ones=True)
            predictions_centers_3d_2d = predictions_centers_3d_2d[:3].T
            predictions_img = np.hstack((predictions_img, predictions_centers_3d_2d))

            # Add projected 3d center information to gt_img
            gt_centers_3d = gt_img[:, 10:13].T
            gt_centers_3d_2d = project_3d_points_in_4D_format(p2, gt_centers_3d, pad_ones=True)
            gt_centers_3d_2d = gt_centers_3d_2d[:3].T
            gt_img = np.hstack((gt_img, gt_centers_3d_2d))

            row_index = np.where(gt_img[:, 12].astype(int) == -1000)[0]
            gt_img = np.delete(gt_img, row_index, axis=0)

            if gt_img.shape[0] > 0:
                # Compute the best overlaps between images
                overlaps = iou(predictions_img[:, 3:7], gt_img[:, 3:7], mode='combinations')
                max_overlaps = np.max(overlaps, axis=1)

                gt_max_overlaps_index = np.argmax(overlaps, axis=1).astype(int)
                gt_matched_img = gt_img[gt_max_overlaps_index].copy()

                suff_overlap_ind = np.where(max_overlaps > min_iou2d_overlap)[0]
                predictions_img = predictions_img[suff_overlap_ind]
                gt_matched_img = gt_matched_img[suff_overlap_ind]
                predictions_all = combine(predictions_all, predictions_img)
                gts_all = combine(gts_all, gt_matched_img)

                N = suff_overlap_ind.shape[0]
                iou_3d_img = np.zeros((N,))
                for j in range(N):
                    _, corners_3d_b1 = project_3d_corners(p2, predictions_img[j, 10], predictions_img[j, 11],
                                                          predictions_img[j, 12], w3d=predictions_img[j, 8],
                                                          h3d=predictions_img[j, 7], l3d=predictions_img[j, 9],
                                                          ry3d=predictions_img[j, 13], iou_3d_convention=True)#,
                                                          #return_in_4D=False)
                    corners_3d_b1 = corners_3d_b1[:3]
                    _, corners_3d_b2 = project_3d_corners(p2, gt_matched_img[j, 10], gt_matched_img[j, 11],
                                                          gt_matched_img[j, 12], w3d=gt_matched_img[j, 8],
                                                          h3d=gt_matched_img[j, 7], l3d=gt_matched_img[j, 9],
                                                          ry3d=gt_matched_img[j, 13], iou_3d_convention=True)#,
                                                          #return_in_4D=False)
                    corners_3d_b2 = corners_3d_b2[:3]
                    _, iou_3d_img[j] = iou3d(corners_3d_b1, corners_3d_b2)
                    # print(iou_3d_img)
                iou_3d_all = combine(iou_3d_all, iou_3d_img)

        if (i + 1) % display_frequency == 0 or i == num_images-1:
            print("{} images done".format(i + 1))

    # ============================== All kinds of thresholding ==========================
    #          0     1     2     3   4   5  6    7    8    9    10   11   12   13    14      15     16     17             18          19      20     21
    # (cls, trunc, occl, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score, width, height, h2d_general, h2d_special, x3d_2d, y3d_2d, z3d_2d  )
    score = predictions_all[:, 14]

    index = np.where(score >= threshold_score)[0]
    gts_all = gts_all[index]
    score = score[index]
    predictions_all = predictions_all[index]
    iou_3d_all = iou_3d_all[index]

    z3d_pred = predictions_all[:, 12]
    z3d_gt = gts_all[:, 12]
    error_in_z = np.abs(z3d_pred - z3d_gt)
    index = np.where(np.abs(error_in_z) <= threshold_depth)
    gts_all = gts_all[index]
    score = score[index]
    predictions_all = predictions_all[index]
    iou_3d_all = iou_3d_all[index]
    # truncation and occlusion
    truncation = gts_all[:, 0]
    occlusion = gts_all[:, 1]

    # =========== 2D stuff ==========
    x1_pred = predictions_all[:, 3]
    x1_gt = gts_all[:, 3]
    y1_pred = predictions_all[:, 4]
    y1_gt = gts_all[:, 4]
    h2d_gt = gts_all[:, 6] - gts_all[:, 4]
    h2d_pred = predictions_all[:, 16]
    h2d_general = predictions_all[:, 17]
    w2d_gt = gts_all[:, 5] - gts_all[:, 3]
    w2d_pred = predictions_all[:, 15]
    iou_2d_all = iou(predictions_all[:, 3:7], gts_all[:, 3:7], mode='list')

    error_in_x1 = np.abs(predictions_all[:, 3] - gts_all[:, 3])
    error_in_y1 = np.abs(predictions_all[:, 4] - gts_all[:, 4])
    error_in_h2d = np.abs(h2d_pred - h2d_gt)
    error_in_w2d = np.abs(w2d_pred - w2d_gt)

    # =========== 3D stuff ==========
    x3d_pred = predictions_all[:, 10]
    x3d_gt = gts_all[:, 10]
    y3d_pred = predictions_all[:, 11]
    y3d_gt = gts_all[:, 11]
    z3d_pred = predictions_all[:, 12]
    z3d_gt = gts_all[:, 12]

    if predictions_all.shape[1] > 18:
        x3d_2d_pred = predictions_all[:, 19]
        y3d_2d_pred = predictions_all[:, 20]
        z3d_2d_pred = predictions_all[:, 21]
    if gts_all.shape[1] > 14:
        x3d_2d_gt = gts_all[:, 14]
        y3d_2d_gt = gts_all[:, 15]
        z3d_2d_gt = gts_all[:, 16]

    h3d_pred = predictions_all[:, 7]
    h3d_gt = gts_all[:, 7]
    w3d_pred = predictions_all[:, 8]
    w3d_gt = gts_all[:, 8]
    l3d_pred = predictions_all[:, 9]
    l3d_gt = gts_all[:, 9]

    rotY_pred = predictions_all[:, 13]
    rotY_gt = gts_all[:, 13]
    alpha_pred = predictions_all[:, 2]
    alpha_gt = gts_all[:, 2]

    scale_h3d = h3d_gt / h3d_pred
    scale_w3d = w3d_gt / w3d_pred
    scale_l3d = l3d_gt / l3d_pred

    error_in_h3d = np.abs(h3d_pred - h3d_gt)
    error_in_l3d = np.abs(l3d_pred - l3d_gt)
    error_in_w3d = np.abs(w3d_pred - w3d_gt)
    error_in_x3d = np.abs(x3d_pred - x3d_gt)
    error_in_y3d = np.abs(y3d_pred - y3d_gt)
    error_in_z = (z3d_pred - z3d_gt)
    error_in_rot = np.abs(rotY_pred - rotY_gt)
    error_in_alpha = np.abs(alpha_pred - alpha_gt)

    if predictions_all.shape[1] > 18:
        error_in_x3d_2d = np.abs(x3d_2d_pred - x3d_2d_gt)
        error_in_y3d_2d = np.abs(y3d_2d_pred - y3d_2d_gt)
        error_in_z3d_2d = np.abs(z3d_2d_pred - z3d_2d_gt)
    else:
        error_in_x3d_2d = None
        error_in_y3d_2d = None
        error_in_z3d_2d = None

    error_general = np.abs(h2d_general - h2d_pred)
    error_general_with_gt = np.abs(h2d_general - h2d_gt)

    error_list = [iou_3d_all, z3d_gt, score, iou_2d_all, truncation, occlusion, error_in_x1, error_in_y1,  error_in_w2d, error_in_h2d, error_in_x3d, error_in_y3d, error_in_l3d, error_in_w3d, error_in_h3d, error_in_rot, error_in_alpha, error_general, error_general_with_gt, error_in_x3d_2d]
    # Create the cache
    pickle_write(file_path= error_list_file_path, obj= error_list)

    return error_list

def open_gif_writer(file_path, duration= 0.5):
    print("=> Saving to {}".format(file_path))
    gif_writer = imageio.get_writer(file_path, mode='I', duration= duration)

    return gif_writer

def convert_fig_to_ubyte_image(fig):
    canvas = FigureCanvas(fig)
    # draw canvas as image
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    image = np.fromstring(s, np.uint8).reshape((height, width, 4))
    image = img_as_ubyte(image)

    return image

def add_ubyte_image_to_gif_writer(gif_writer, ubyte_image):

    gif_writer.append_data(ubyte_image)

def close_gif_writer(gif_writer):

    gif_writer.close()

def draw_red_border(sub1):
    autoAxis = sub1.axis()
    rec = patches.Rectangle((autoAxis[0],autoAxis[2]),(autoAxis[1]-autoAxis[0]),(autoAxis[3]-autoAxis[2]),fill=False, color= 'red', lw= 7)
    rec = sub1.add_patch(rec)
    rec.set_clip_on(False)