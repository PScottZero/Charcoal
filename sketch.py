import matplotlib.image as mpimg
import numpy as np
import os
import time

# ===============================
# Convert rgb image to grayscale
# ===============================
def imgToGrayscale(img):
    print('Converting Image To Grayscale...')
    grayscale = np.zeros((len(img), len(img[0])))
    for i in range(len(img)):
        for j in range(len(img[i])):
            grayscale[i][j] = (int(img[i][j][0]) + int(img[i][j][1]) + int(img[i][j][2])) / 3
    return grayscale

# ===============================
# Pad image using extend method
# =============================== 
def padImg(img):
    print('Padding Image...')
    img_pad = np.zeros((len(img) + 2, len(img[0]) + 2))
    for i in range(len(img_pad)):
        for j in range(len(img_pad[i])):

            # corner padding
            if i == 0 and j == 0:
                img_pad[0][0] = img[0][0]
            elif i == 0 and j == len(img[0]) + 1:
                img_pad[0][len(img[0]) + 1] = img[0][len(img[0]) - 1]
            elif i == len(img) + 1 and j == 0:
                img_pad[len(img) + 1][0] = img[len(img) - 1][0]
            elif i == len(img) + 1 and j == len(img[0]) + 1:
                img_pad[len(img) + 1][len(img[0]) + 1] = img[len(img) - 1][len(img[0]) - 1]

            # side padding
            elif i == 0:
                img_pad[i][j] = img[i][j - 1]
            elif i == len(img) + 1:
                img_pad[i][j] = img[i - 2][j - 1]
            elif j == 0:
                img_pad[i][j] = img[i - 1][j]
            elif j == len(img[0]) + 1:
                img_pad[i][j] = img[i - 1][j - 2]

            # copy image to padded image
            else:
                img_pad[i][j] = img[i - 1, j - 1]
    return img_pad

# ===============================
# apply given 3x3 filter
# to padded image
# ===============================
def applyFilter(img_pad, fil, fil_name):
    print('Applying {} Filter...'.format(fil_name))
    img_fil = np.zeros((len(img_pad) - 2, len(img_pad[0]) - 2))
    for i in range(len(img_pad) - 2):
        for j in range(len(img_pad[i]) - 2):
            new_val = 0
            for m in range(3):
                for n in range(3):
                    new_val += img_pad[i + m - 1][j + n - 1] * fil[2 - m][2 - n]
            img_fil[i][j] = new_val
    return img_fil

# ===============================
# Edge detection convolution for
# padded image
# ===============================
def edgeConv(img_pad):
    deriv_x = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
    deriv_y = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]
    img_deriv_x = applyFilter(img_pad, deriv_x, 'X Derivative')
    img_deriv_y = applyFilter(img_pad, deriv_y, 'Y Derivative')
    img_edge = np.zeros((len(img_pad) - 2, len(img_pad[0]) - 2))
    print('Merging X And Y Derivatives...')
    for i in range(len(img_edge)):
        for j in range(len(img_edge[i])):
            img_edge[i][j] = np.sqrt(img_deriv_x[i][j] ** 2 + img_deriv_y[i][j] ** 2)
    return img_edge

# ===============================
# Invert image
# ===============================
def invertImage(img):
    print("Inverting Image...")
    for i in range(len(img)):
        for j in range(len(img[0])):
            img[i][j] = (255 - img[i][j])
    return img

# ===============================
# Reduce intensity of image
# ===============================
def reduceIntensity(img):
    print("Reducing Image Intensity...")
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] < 60:
                img[i][j] = 60
    return img

# ===============================
# Colorize image
# ===============================
def colorize(img, img_org):
    print("Colorizing Image...")
    colorized = np.zeros((len(img), len(img[0]), 3))
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] <= 200:
                colorized[i][j][0] = img_org[i][j][0] / 255
                colorized[i][j][1] = img_org[i][j][1] / 255
                colorized[i][j][2] = img_org[i][j][2] / 255
            else:
                colorized[i][j][0] = 1
                colorized[i][j][1] = 1
                colorized[i][j][2] = 1
    return colorized

def getElapsedTime(start_time):
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    return '{}m {}s'.format(round(minutes), round(seconds))

# ===============================
# Apply sketch filter to
# given image
# ===============================
def applySketchFilter(img_name):
    print('>>>> Applying Sketch Filter To {} <<<<<'.format(img_name))
    start_time = time.time()
    img = mpimg.imread('images/' + img_name)
    img_gray = imgToGrayscale(img)
    img_pad = padImg(img_gray)
    img_edge = edgeConv(img_pad)
    img_inv = invertImage(img_edge)
    img_sketch = reduceIntensity(img_inv)
    print('Saving B/W Sketch...')
    mpimg.imsave('results/' + img_name + '_sketch.png', img_sketch, cmap='gray', vmin=0, vmax=255)
    img_color = colorize(img_inv, img)
    print('Saving Colorized Sketch...')
    mpimg.imsave('results/' + img_name + '_sketch_color.png', img_color)
    print('Finished {} In {}\n'.format(img_name, getElapsedTime(start_time)))

# ===============================
# Get all jpg images in 
# image directory and apply
# sketch filter to them
# ===============================
def run():
    print('======================================')
    print('| Python Sketch Filter               |')
    print('| Programmer By Paul Scott           |')
    print('| Computer Science Major, Penn State |')
    print('| Version 2020.4.29                  |')
    print('======================================\n')
    img_count = 0
    start_time = time.time()
    for file in os.listdir('images/'):
        if file.endswith('.jpg'):
            img_count += 1
            applySketchFilter(file)
    if img_count > 1:
        print('Finished In {}\n'.format(getElapsedTime(start_time)))
    elif img_count == 0:
        print('No .jpg Files Found In Images Directory!!!\n')

run()
