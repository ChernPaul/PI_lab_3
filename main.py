import numpy as np
from skimage.io import imsave, imshow, show, imread
from matplotlib import pyplot as plt
from scipy import signal
from scipy.ndimage import median_filter
from skimage.util import random_noise

MAX_BRIGHTNESS_VALUE = 255
MIN_BRIGHTNESS_VALUE = 0

LEFT_BRIGHTNESS_VALUE = 96
RIGHT_BRIGHTNESS_VALUE = 160
CELL_HEIGHT = 16
IMAGE_HEIGHT = 128
IMAGE_LENGTH = 128
BORDER_PROCESSING_PARAMETER = 1
VALUE_OF_ONE = 1

MEDIAN_FILTER_MASK_1 = np.array([[0, 1, 0],
                                 [1, 3, 1],
                                 [0, 1, 0]])

MEDIAN_FILTER_MASK_2 = np.array([[1, 1, 1],
                                 [1, 3, 1],
                                 [1, 1, 1]])

MEDIAN_FILTER_MASK_3 = np.array([[0, 1, 0],
                                 [1, 1, 1],
                                 [0, 1, 0]])


MEDIAN_FILTER_MASK_4 = np.array([[1, 1, 1],
                                 [1, 1, 1],
                                 [1, 1, 1]])

LINEAR_FILTER_MASK_A = np.array([[1, 1, 1],
                                 [1, 1, 1],
                                 [1, 1, 1]])*(1/9)

LINEAR_FILTER_MASK_B = np.array([[1, 1, 1],
                                 [1, 2, 1],
                                 [1, 1, 1]])*(1/10)

LINEAR_FILTER_MASK_C = np.array([[1, 2, 1],
                                 [2, 4, 2],
                                 [1, 2, 1]])*(1/16)


def border_processing_function(element_value):
    if element_value >= BORDER_PROCESSING_PARAMETER:
        return RIGHT_BRIGHTNESS_VALUE
    else:
        return LEFT_BRIGHTNESS_VALUE


def border_processing(img_as_arrays):
    shape = np.shape(img_as_arrays)
    new_img_list = list(map(border_processing_function, np.reshape(img_as_arrays, img_as_arrays.size)))
    single_dimension_array = np.array(new_img_list)
    new_img = np.reshape(single_dimension_array, (shape[0], shape[1]))
    return new_img


def create_wb_histogram_plot(img_as_arrays):
    hist, bins = np.histogram(img_as_arrays.flatten(), 256, [0, 256])
    plt.plot(bins[:-1], hist, color='blue', linestyle='-', linewidth=1)


def open_image_as_arrays(filepath):
    return imread(filepath)


def save_image(img, directory):
    imsave(directory, img)


def create_chess_field_image():
    img = np.ones((IMAGE_LENGTH, IMAGE_HEIGHT)).astype(int)
    line_index = 0
    indexes_of_start_black_odd = start_of_black_sells_odd()
    indexes_of_start_black_even = start_of_black_sells_even()
    odd_row = False

    while line_index < IMAGE_HEIGHT:

        column_index = 0

        if odd_row:
            current_indexes = indexes_of_start_black_odd
        else:
            current_indexes = indexes_of_start_black_even

        while column_index < IMAGE_LENGTH/(2*CELL_HEIGHT):
            j = 0
            while j < CELL_HEIGHT:
                img[line_index][current_indexes[column_index] + j] = 0
                j = j + 1
            column_index = column_index + 1
        line_index = line_index + 1
        if line_index % CELL_HEIGHT == 0:
            odd_row = not odd_row
    return img


def start_of_black_sells_odd():
    result = []
    i = 0
    j = 0
    while 2*i < IMAGE_HEIGHT:
        result.insert(j, 2*i)
        i = i + CELL_HEIGHT
        j = j + 1
    return result


def start_of_black_sells_even():
    result = []
    i = CELL_HEIGHT
    j = 0
    while i < IMAGE_HEIGHT:
        result.insert(j, i)
        i = i + 2*CELL_HEIGHT
        j = j + 1
    return result


def window_processing(matrix, window):
    return signal.convolve2d(matrix, window, boundary='symm', mode='same').astype(int)


def create_figure_of_linear_filter_processing(img_noise_1, img_noise_10,  noise_img_1, noise_img_10, linear_filter_1, linear_filter_10):
    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(2, 3, 1)
    plt.title("Image with noise 1")
    imshow(img_noise_1, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 2)
    plt.title("Noise image relation 1")
    imshow(noise_img_1, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 3)
    plt.title("Noise image linear_filter_1")
    imshow(linear_filter_1, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 4)
    plt.title("Image with noise 10 ")
    imshow(img_noise_10, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 5)
    plt.title("Noise image relation 10")
    imshow(noise_img_10, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 6)
    plt.title("Noise image linear_filter_10")
    imshow(linear_filter_10, cmap='gray', vmin=0, vmax=255)
    return fig


def create_figure_of_median_filter_processing(img_noise_1, img_noise_10, noise_img_1, noise_img_10, median_filter_1, median_filter_10):
    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(2, 3, 1)
    plt.title("Image with noise 1")
    imshow(img_noise_1, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 2)
    plt.title("Noise image relation 1")
    imshow(noise_img_1, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 3)
    plt.title("Noise image median_filter_1")
    imshow(median_filter_1, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 4)
    plt.title("Image with noise 10 ")
    imshow(img_noise_10, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 5)
    plt.title("Noise image relation 10")
    imshow(noise_img_10, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 6)
    plt.title("Noise image median_filter_10")
    imshow(median_filter_10, cmap='gray', vmin=0, vmax=255)
    return fig


def create_figure_of_median_filter_processing_imp(imp_noise, img_imp_noise, linear_filtered_img, median_filtered_img):
    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(2, 2, 1)
    plt.title("Image of noise")
    imshow(imp_noise, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 2, 2)
    plt.title("Source image with noise")
    imshow(img_imp_noise, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 2, 3)
    plt.title("Noised image median_filter")
    imshow(median_filtered_img, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 2, 4)
    plt.title("Noised image linear_filter")
    imshow(linear_filtered_img, cmap='gray', vmin=0, vmax=255)
    return fig

def check_and_correct_limits(img_as_arrays):
    shape = np.shape(img_as_arrays)
    new_img_list = list(map(correct_limits_function, np.reshape(img_as_arrays, img_as_arrays.size)))
    single_dimension_array = np.array(new_img_list)
    new_img = np.reshape(single_dimension_array, (shape[0], shape[1]))
    return new_img


def correct_limits_function(element_value):
    if element_value < MIN_BRIGHTNESS_VALUE:
        return MIN_BRIGHTNESS_VALUE
    if element_value > MAX_BRIGHTNESS_VALUE:
        return MAX_BRIGHTNESS_VALUE
    return element_value


def coefficient_of_decreasing_noise(src_img, img_and_noise, filtered_img):
    divided = middle_square_error_pow_2(src_img, filtered_img)
    delimiter = np.mean(np.square(img_and_noise - src_img))
    return (divided/delimiter).astype(float)


def middle_square_error_pow_2(src_img, filtered_img):
    return np.mean(np.square(filtered_img - src_img))


def fill_salt_and_pepper_help(element_value):
    if element_value == 0:
        return MAX_BRIGHTNESS_VALUE
    if element_value < 0:
        return MIN_BRIGHTNESS_VALUE
    return element_value


def replace_zeros_and_ones(element_value):
    if element_value == 0:
        return 1
    if element_value == 1:
        return 0
    return element_value


def fill_salt_and_pepper(noise_matrix, src_img):
    noise_matrix_copy = noise_matrix
    shape = np.shape(noise_matrix_copy)
    new_noise_list = list(map(replace_zeros_and_ones, np.reshape(noise_matrix_copy, noise_matrix_copy.size)))
    single_dimension_array = np.array(new_noise_list)
    noise_matrix_copy = np.multiply(np.reshape(single_dimension_array, (shape[0], shape[1])), src_img)
    new_img_list = list(map(fill_salt_and_pepper_help, np.reshape(noise_matrix_copy, noise_matrix_copy.size)))
    single_dimension_array = np.array(new_img_list)
    new_img = np.reshape(single_dimension_array, (shape[0], shape[1]))
    return new_img


def to_show_impulse_noise(element_value):
    if element_value == -1:
        return MIN_BRIGHTNESS_VALUE
    if element_value == 1:
        return MAX_BRIGHTNESS_VALUE
    return 128


def show_impulse_noise(noise_matrix):
    shape = np.shape(noise_matrix)
    new_img_list = list(map(to_show_impulse_noise, np.reshape(noise_matrix, noise_matrix.size)))
    single_dimension_array = np.array(new_img_list)
    new_img = np.reshape(single_dimension_array, (shape[0], shape[1]))
    return new_img


chess_img = create_chess_field_image()
print(chess_img)
chess_board_img = border_processing(chess_img)


img_dispersion = np.var(chess_board_img)
white_noise_10 = np.random.normal(loc=0, scale=float(np.sqrt(float(img_dispersion / 10))), size=(IMAGE_HEIGHT, IMAGE_LENGTH))
print(white_noise_10)
print("min of noise 10")
print(np.min(white_noise_10))
print("max of noise 10")
print(np.max(white_noise_10))
dispersion_noise_10 = np.var(white_noise_10)


white_noise_matrix_10 = white_noise_10.astype(int)
img_with_white_noise_10 = check_and_correct_limits(chess_board_img + white_noise_matrix_10)

median_filter_img_10 = median_filter(img_with_white_noise_10, footprint=MEDIAN_FILTER_MASK_1)
linear_filter_img_10 = window_processing(img_with_white_noise_10, LINEAR_FILTER_MASK_A)


print("MSE linear filter 10")
print(middle_square_error_pow_2(chess_board_img, linear_filter_img_10))

print("MSE median filter 10")
print(middle_square_error_pow_2(chess_board_img, median_filter_img_10))

print("Another test")


white_noise_1 = np.random.normal(loc=0, scale=float(np.sqrt(float(img_dispersion))), size=(IMAGE_HEIGHT, IMAGE_LENGTH))
print("min of noise 1")
print(np.min(white_noise_1))
print("max of noise 1")
print(np.max(white_noise_1))


white_noise_matrix_1 = white_noise_1.astype(int)
img_with_white_noise_1 = check_and_correct_limits(chess_board_img + white_noise_matrix_1)

median_filter_img_1 = median_filter(img_with_white_noise_1, footprint=MEDIAN_FILTER_MASK_1)
linear_filter_img_1 = window_processing(img_with_white_noise_1, LINEAR_FILTER_MASK_A)


print("MSE linear filter 1")
print(middle_square_error_pow_2(chess_board_img, linear_filter_img_1))

print("MSE median filter 1")
print(middle_square_error_pow_2(chess_board_img, median_filter_img_1))

fig = plt.figure(figsize=(20, 10))
fig.add_subplot(1, 1, 1)
plt.title("Source image")
imshow(border_processing(chess_img), cmap='gray', vmin=0, vmax=255)
show()

create_figure_of_linear_filter_processing(img_with_white_noise_1, img_with_white_noise_10, np.abs(white_noise_matrix_1), np.abs(white_noise_matrix_10), linear_filter_img_1, linear_filter_img_10)
show()

create_figure_of_median_filter_processing(img_with_white_noise_1, img_with_white_noise_10, np.abs(white_noise_matrix_1), np.abs(white_noise_matrix_10), median_filter_img_1, median_filter_img_10)
show()

array_to_generate = np.zeros((128, 128), dtype=int)
array_to_generate[1][1] = -1

impulse_noise_01 = random_noise(array_to_generate, mode="s&p", amount=0.1)
img_with_impulse_noise_intense_01 = fill_salt_and_pepper(impulse_noise_01, chess_board_img)


median_filter_img_imp_noise_01 = median_filter(img_with_impulse_noise_intense_01, footprint=MEDIAN_FILTER_MASK_1)
linear_filter_img_imp_noise_01 = window_processing(img_with_impulse_noise_intense_01, LINEAR_FILTER_MASK_A)

impulse_noise_03 = random_noise(array_to_generate, mode="s&p", amount=0.3)
img_with_impulse_noise_intense_03 = fill_salt_and_pepper(impulse_noise_03, chess_board_img)


median_filter_img_imp_noise_03 = median_filter(img_with_impulse_noise_intense_03, footprint=MEDIAN_FILTER_MASK_1)
linear_filter_img_imp_noise_03 = window_processing(img_with_impulse_noise_intense_03, LINEAR_FILTER_MASK_A)

create_figure_of_median_filter_processing_imp(show_impulse_noise(impulse_noise_01), img_with_impulse_noise_intense_01, linear_filter_img_imp_noise_01, median_filter_img_imp_noise_01)
show()
create_figure_of_median_filter_processing_imp(show_impulse_noise(impulse_noise_03), img_with_impulse_noise_intense_03, linear_filter_img_imp_noise_03, median_filter_img_imp_noise_03)
show()


print("=============================================================================================================")
print("=============================================================================================================")
print("=============================================================================================================")

print("Dispersion of images and noises")
print("=============================================================================================================")

print("dispersion of image")
print(img_dispersion)

print("dispersion of noise 10")
print(dispersion_noise_10)

print("dispersion of image with noise 10")
print(np.var(img_with_white_noise_10))


print("dispersion of noise 1")
print(np.var(white_noise_1))

print("dispersion of image with noise 1")
print(np.var(img_with_white_noise_1))

print("dispersion of noise 01")
print(np.var(impulse_noise_01))

print("dispersion of image with noise 01")
print(np.var(img_with_impulse_noise_intense_01))

print("dispersion of noise 03")
print(np.var(impulse_noise_03))

print("dispersion of image with noise 03")
print(np.var(img_with_impulse_noise_intense_03))

print("=============================================================================================================")


print("Dispersion filtration errors")
print("=============================================================================================================")


print("dispersion of linear filtered image with noise 10")
print(middle_square_error_pow_2(chess_board_img, linear_filter_img_10))

print("dispersion of median filtered image with noise 10")
print(middle_square_error_pow_2(chess_board_img, median_filter_img_10))

print("dispersion of linear filtered image with noise 1")
print(middle_square_error_pow_2(chess_board_img, linear_filter_img_1))

print("dispersion of median filtered image with noise 1")
print(middle_square_error_pow_2(chess_board_img, median_filter_img_1))

print("dispersion of linear filtered image with noise 01")
print(middle_square_error_pow_2(chess_board_img, linear_filter_img_imp_noise_01))

print("dispersion of median filtered image with noise 01")
print(middle_square_error_pow_2(chess_board_img, median_filter_img_imp_noise_01))

print("dispersion of linear filtered image with noise 03")
print(middle_square_error_pow_2(chess_board_img, linear_filter_img_imp_noise_03))

print("dispersion of median filtered image with noise 03")
print(middle_square_error_pow_2(chess_board_img, median_filter_img_imp_noise_03))


print("=============================================================================================================")
print("Suppress noise coefficients")
print("=============================================================================================================")


print("Coefficient of decreasing noise linear filter 10")
print(coefficient_of_decreasing_noise(chess_board_img, img_with_white_noise_10, linear_filter_img_10))

print("Coefficient of decreasing noise median filter 10")
print(coefficient_of_decreasing_noise(chess_board_img, img_with_white_noise_10, median_filter_img_10))

print("Coefficient of decreasing noise linear filter 1")
print(coefficient_of_decreasing_noise(chess_board_img, img_with_white_noise_1, linear_filter_img_1))

print("Coefficient of decreasing noise median filter 1")
print(coefficient_of_decreasing_noise(chess_board_img, img_with_white_noise_1, median_filter_img_1))

print("Coefficient of decreasing noise linear filter 01")
print(coefficient_of_decreasing_noise(chess_board_img, img_with_impulse_noise_intense_01, linear_filter_img_imp_noise_01))

print("Coefficient of decreasing noise median filter 01")
print(coefficient_of_decreasing_noise(chess_board_img, img_with_impulse_noise_intense_01, median_filter_img_imp_noise_01))

print("Coefficient of decreasing noise linear filter 03")
print(coefficient_of_decreasing_noise(chess_board_img, img_with_impulse_noise_intense_03, linear_filter_img_imp_noise_03))

print("Coefficient of decreasing noise median filter 03")
print(coefficient_of_decreasing_noise(chess_board_img, img_with_impulse_noise_intense_03, median_filter_img_imp_noise_03))


print("=============================================================================================================")
print("Chernyshev")

