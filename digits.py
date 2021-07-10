import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

model = keras.models.load_model('my_model')


def load_img(img_name):
    img = ~mpimg.imread(img_name)
    img = np.delete(img, 2, 2)
    img = np.delete(img, 1, 2)
    img = img.squeeze()
    img = clear_noise(img)
    return img


def split_img(img):
    n = 0
    between_nums = False
    height, width = img.shape
    # [White start, Number start, Number end, White end]
    first_num = {'white start': 0, 'num start': -1, 'num end': -1, 'white end': -1}
    second_num = {'white start': -1, 'num start': -1, 'num end': -1, 'white end': -1}
    third_num = {'white start': -1, 'num start': -1, 'num end': -1, 'white end': width - 1}

    img = img.swapaxes(0, 1)

    for i in range(len(img)):
        column = img[i]
        if n == 0:
            if not is_full_white(column):
                n = 1
                first_num['num start'] = i
        elif n == 1:
            if is_full_white(column):
                first_num['num end'] = i
                second_num['white start'] = i
                n = 2
                between_nums = True
        elif n == 2 and between_nums:
            if not is_full_white(column):
                first_num['white end'] = i
                second_num['num start'] = i
                between_nums = False
        elif n == 2:
            if is_full_white(column):
                second_num['num end'] = i
                third_num['white start'] = i
                n = 3
                between_nums = True
        elif n == 3 and between_nums:
            if not is_full_white(column):
                second_num['white end'] = i
                third_num['num start'] = i
                between_nums = False
        elif n == 3:
            if is_full_white(column):
                third_num['num end'] = i
                break

    img = img.swapaxes(0, 1)
    if first_num['num start'] == -1:  # Empty canvas
        return None, None, None
    elif second_num['num start'] == -1:  # One digit
        first_num['white end'] = width - 1
        return center_digit(img, first_num, height), None, None
    elif third_num['num start'] == -1:  # Two digit
        second_num['white end'] = width - 1
        return center_digit(img, first_num, height), center_digit(img, second_num, height), None
    else:
        return center_digit(img, first_num, height), center_digit(img, second_num, height), \
               center_digit(img, third_num, height)


def get_y_start_end(img):
    start = -1
    end = -1
    for i in range(len(img)):
        row = img[i]
        if start == -1 and not is_full_white(row):
            start = i
        elif start != -1 and is_full_white(row):
            end = i
            break
    return start, end


def center_digit(img, img_dict, height):
    white_start, num_start, num_end, white_end = img_dict.values()
    img = img.copy()
    recommended_diff_x = int((1 / 3) * (num_end - num_start))
    diff_x = min(num_start - white_start, white_end - num_end, recommended_diff_x)
    img = img[:, num_start - diff_x:num_end + diff_x]
    y_start, y_end = get_y_start_end(img)
    recommended_diff_y = int((1 / 3) * (y_end - y_start))
    diff_y = min(y_start - 0, height - y_end, recommended_diff_y)
    img = img[y_start - diff_y:y_end + diff_y, :]
    resizer = Image.fromarray(img)
    resizer = resizer.resize((28, 28))
    img = np.array(resizer)
    img = clear_noise(img)
    return img


def is_full_white(line):
    for j in range(len(line)):
        if line[j] != 0:
            return False
    return True


def show(img):
    if img is None:
        return
    try:
        plt.imshow(img, cmap=plt.cm.binary)
    except:
        plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()


def predict(img):
    try:
        lst = model.predict(np.expand_dims(img, 0))
    except:
        lst = model.predict(img)

    index = 0
    val = lst[0][0]
    for i in range(1, len(lst[0])):
        if lst[0][i] > val:
            index = i
            val = lst[0][i]
    return index, val


def clear_noise(img):
    img = img.copy()
    for i in img:
        for j in range(len(i)):
            if i[j] < 130:
                i[j] = 0
    return img


def train_model():
    mnist = keras.datasets.mnist
    (train_img, train_label), (test_img, test_label) = mnist.load_data()
    all_img = np.concatenate((train_img, test_img), axis=0)
    all_label = np.concatenate((train_label, test_label), axis=0)

    lst_img = []
    lst_label = []

    for number in range(10):
        for pic_index in range(256):
            current = load_img(f"new_trained/{number}/{pic_index}.jpg")
            lst_img.append(current)
            lst_label.append(np.uint8(number))

    for number in range(10):
        for pic_index in range(100):
            current = load_img(f"Trained/{number}/{pic_index}.jpg")
            lst_img.append(current)
            lst_label.append(np.uint8(number))

    lst_img = np.array(lst_img)
    lst_label = np.array(lst_label)

    all_img = np.concatenate((lst_img, all_img), axis=0)
    all_label = np.concatenate((lst_label, all_label), axis=0)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(all_img, all_label, epochs=512)
    model.save("my_model")

    test_loss, test_accuracy = model.evaluate(test_img, test_label)
    predictions = model.predict(test_img)


def get_results(img_name="tmp_img.jpg"):
    full_img = load_img(img_name)
    digits = list(split_img(full_img))

    if digits[0] is None:
        return None, None
    elif digits[1] is None:
        plt.imsave('img1.jpg', digits[0], cmap=plt.cm.binary)
        dig_1, accuracy_1 = predict(digits[0])
        return dig_1, round(accuracy_1 * 100, 2)

    elif digits[2] is None:
        plt.imsave('img1.jpg', digits[0], cmap=plt.cm.binary)
        plt.imsave('img2.jpg', digits[1], cmap=plt.cm.binary)
        dig_1, accuracy_1 = predict(digits[0])
        dig_2, accuracy_2 = predict(digits[1])
        return dig_1 * 10 + dig_2, round(accuracy_1 * accuracy_2 * 100, 2)
    else:
        plt.imsave('img1.jpg', digits[0], cmap=plt.cm.binary)
        plt.imsave('img2.jpg', digits[1], cmap=plt.cm.binary)
        plt.imsave('img3.jpg', digits[2], cmap=plt.cm.binary)
        dig_1, accuracy_1 = predict(digits[0])
        dig_2, accuracy_2 = predict(digits[1])
        dig_3, accuracy_3 = predict(digits[2])
        return dig_1 * 100 + dig_2 * 10 + dig_3, round(accuracy_1 * accuracy_2 * accuracy_3 * 100, 2)
