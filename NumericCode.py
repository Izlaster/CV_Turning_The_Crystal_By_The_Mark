import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import ImageTk, Image

import cv2
import numpy as np
import imagehash

import os

root = tk.Tk()

# Генерирование хеша изображения
def generate_matrix_id(img):
    rgb_image  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_image)
    matrix_hash = int(str(imagehash.average_hash(image)), 16) // 10**6
    return str(matrix_hash)

def circle_mask(image, center, radius):
    # Получение высоты и ширины изображения
    height, width = image.shape[:2]

    # Создание маски нулевой матрицей с размерами изображения
    mask = np.zeros((height, width), dtype=np.uint8)

    # Нанесение круга на маску с заданным центром и радиусом
    cv2.circle(mask, center, radius, 255, -1)

    # Применение маски к исходному изображению
    masked = cv2.bitwise_and(image, image, mask=mask)

    # Возвращение маскированного изображения
    return masked

def check_black_pixels(image):
    # Преобразование изображения в оттенки серого
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Подсчет количества черных пикселей
    total_pixels = gray_image.size
    black_pixels = np.count_nonzero(gray_image == 0)

    # Вычисление процента черных пикселей
    percentage = (black_pixels / total_pixels) * 100

    # Проверка условия
    if percentage > 60:
        return True
    else:
        return False
    
def variance_of_laplacian(image):
    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Рассчитывает лапласиан изображения, а также вариацию
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm

def rotate(image, center_circle, center_box):
    # Вычисление угла между красной линией и вертикальной осью
    angle = np.arctan2(center_box[1] - center_circle[1], center_box[0] - center_circle[0]) * 180 / np.pi + 90

    # Поворот изображения вокруг его центра
    image_height, image_width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((image_width/2, image_height/2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image_width, image_height))

    # Возвращение повернутого изображения
    return rotated_image

def show_image(image_cv):
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    
    global image
    image = Image.fromarray(image_rgb)

    # Изменяем размер изображения, чтобы поместиться в окно
    update_image_label(image)

def update_image_label(image):
    # Получаем размеры окна
    window_width = root.winfo_width()
    window_height = root.winfo_height()

    # Изменяем размер изображения, чтобы сохранить пропорции
    image_width, image_height = image.size
    aspect_ratio = image_width / image_height

    if window_width / window_height > aspect_ratio:
        new_width = int(window_height * aspect_ratio)
        new_height = window_height
    else:
        new_width = window_width
        new_height = int(window_width / aspect_ratio)

    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    photo = ImageTk.PhotoImage(resized_image)

    if hasattr(update_image_label, 'image_label'):  # Проверяем, существует ли уже image_label
        update_image_label.image_label.configure(image=photo)
        update_image_label.image_label.image = photo
    else:
        # Создаем виджет Label для отображения изображения
        update_image_label.image_label = tk.Label(frame)
        update_image_label.image_label.image = photo
        update_image_label.image_label.pack(expand=True, fill=tk.BOTH)

def on_window_resize(event):
    # При изменении размера окна вызываем функцию обновления размеров image_label
    update_image_label(image)

def get_glossy(path):
    # Загрузка изображения
    image = cv2.imread(path)

    # Определение нижнего и верхнего порогов значений HSV для фильтрации
    hsv_min = np.array((68, 0, 0), np.uint8)
    hsv_max = np.array((255, 255, 255), np.uint8)

    # Преобразование цветовой модели из BGR в HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Применение цветового фильтра к изображению
    thresh = cv2.inRange(hsv, hsv_min, hsv_max)

    # Поиск контуров на изображении
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Вычисление площади каждого контура
    areas = [cv2.contourArea(c) for c in contours]
    sorted_areas = np.sort(areas)

    # Поиск основной окружности
    i = -1
    while True:
        cnt = contours[areas.index(sorted_areas[i])] # наибольший контур
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius < min(image.shape[0], image.shape[1])/2:
            center = (int(x), int(y))
            radius = int(radius)
            break
        else:
            i -= 1

    # Создание маски, ограничивающей окружность
    masked_image = circle_mask(image, center, radius)

    # Проверка яркости изображения с помощью вариации лапласиана
    if variance_of_laplacian(image) < 58:
        return masked_image, 0
    
    # Возвращение маскированного изображения и координат центра окружности
    return masked_image, center

def remove_background(path):
    # Преобразование изображения
    image = cv2.imread(path)

    # Желаемый размер
    if image.shape[0] < 300:
        width = image.shape[0] * 5
        height = image.shape[0] * 5

        # Изменение размера изображения
        image = cv2.resize(image, (width, height))

    # Выполнить сегментацию со средним сдвигом
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    shifted = cv2.pyrMeanShiftFiltering(image_lab, 20, 45)

    # Преобразование в оттенки серого
    shifted_gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)

    # Порог изображения
    _, thresh = cv2.threshold(shifted_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Найти наибольший контур
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    # Выполнение слежения за объектом
    (x, y, w, h) = cv2.boundingRect(largest_contour)

    # Вычислить центр прямоугольника
    center_x = x + w // 2
    center_y = y + h // 2

    # Вычислить главную и малую оси эллипса
    a = w // 2
    b = h // 2

    canvas = np.zeros_like(image) 
    # Рисуем окружность на изображении
    cv2.ellipse(canvas, (center_x, center_y), (a, b), 0, 0, 360, (255, 255, 255), -1)

    # Побитовое И между исходным изображением и маской
    result = cv2.bitwise_and(image, canvas)

    return result, [center_x, center_y]

def try_filter(image, center_circle, check=False):
    # Проверка наличия черных пикселей на изображении или изображение сильно размыто
    if check_black_pixels(image):
        # messagebox.showerror(title='Ошибка', message='Невозможно найти кристалл!')
        return False

    metka = False

    # Размытие изображения
    blurred = cv2.medianBlur(image, 15)

    # Определение минимальных и максимальных значений HSV для фильтрации
    hsv_min = np.array((0, 0, 0), np.uint8)
    hsv_max_range = np.arange(130, 256)[::-1]

    # Перебор фильтраций изображения
    for v_max in hsv_max_range:
        hsv_max = np.array((255, 255, v_max), np.uint8)

        # Преобразование цветовой модели из BGR в HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Применение цветового фильтра к изображению
        thresh = cv2.inRange(hsv, hsv_min, hsv_max)

        # Поиск контуров на изображении
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Обход контуров
        for cnt in contours:
            # Вычисление площади контура
            area = cv2.contourArea(cnt)

            # Проверка условия площади для фильтрации
            if 3479 < area < 9000:
                # Поиск ограничивающего прямоугольника минимальной площади
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.intp(box)

                # Вычисление длины и ширины прямоугольника
                length = np.linalg.norm(box[0] - box[1])
                width = np.linalg.norm(box[1] - box[2])

                # Проверка условия на форму прямоугольника
                if abs(width - length) < width / 4:
                    # Проверка, что все точки прямоугольника находятся в пределах изображения
                    if (box[:, 0] >= 0).all() and (box[:, 0] < image.shape[1]).all() and \
                            (box[:, 1] >= 0).all() and (box[:, 1] < image.shape[0]).all():
                        # Проверка, что хотя бы один пиксель внутри прямоугольника имеет значение 0
                        if np.any(image[box[:, 1], box[:, 0]] == 0):
                            # Проверка, является ли это проверочной операцией
                            if check:
                                # messagebox.showinfo(title='Информация', message='Метка была найдена!')
                                return True
                            else:
                                # Вычисление центра прямоугольника
                                center_box = (int((box[0][0] + box[2][0]) / 2), int((box[0][1] + box[2][1]) / 2))

                                # Копирование изображения и рисование ограничивающего прямоугольника
                                imBox = image.copy()
                                cv2.drawContours(imBox, [box], 0, (255, 255, 0), 2)

                                # Поворот изображения относительно центральной окружности
                                imRotate = rotate(image, center_circle, center_box)

                                # Объединение изображений для отображения
                                vis = np.concatenate((imBox, imRotate), axis=1)
                                show_image(vis)

                                # Генерация числового кода
                                result = generate_matrix_id(imRotate)
                                info['text'] = f'Числовой код: \n{result}'
                                return result
                        else:
                            metka = True

    # Обработка случаев, когда метка не была найдена или была найдена неверная метка
    if metka:
        messagebox.showerror(title='Предупреждение', message='Была найдена неверная метка!')
    # else:
    #     messagebox.showerror(title='Ошибка', message='Метки не было найдено!')
    return False

def btn_click_crystal():
    file = fileInput.get()
    if file == '':
        messagebox.showerror(title='Ошибка', message='Поле не было заполнено!')
        return
    file_path = filedialog.askopenfilename()
    image, center_circle = remove_background(file_path)
    with open(f"{str(file)}.txt", "a") as f:
        img = try_filter(image, center_circle)
        if img: f.write(os.path.basename(file_path) + " = " + str(img) + "\n")

def btn_click_glossy():
    file = fileInput.get()
    if file == '':
        messagebox.showerror(title='Ошибка', message='Поле не было заполнено!')
        return
    file_path = filedialog.askopenfilename()
    image, center_circle = get_glossy(file_path)
    with open(f"{str(file)}.txt", "a") as f:
        img = try_filter(image, center_circle)
        if img: f.write(os.path.basename(file_path) + " = " + str(img) + "\n")

def btn_check():
    file_path = filedialog.askopenfilename()
    image1, center_circle1 = get_glossy(file_path)
    image2, center_circle2 = remove_background(file_path)
    if try_filter(image1, center_circle1, check=True) and try_filter(image2, center_circle2, check=True):
        messagebox.showinfo(title='Информация', message='Метка была найдена на двух методам!')
    elif try_filter(image1, center_circle1, check=True):
        messagebox.showinfo(title='Информация', message='Метка была найдена по глянцу!')
    elif try_filter(image2, center_circle2, check=True):
        messagebox.showinfo(title='Информация', message='Метка была найдена по кристаллу!')
    else:
        messagebox.showinfo(title='Информация', message='Метка не была найдена!')


root.title('Числовой код')
root.wm_attributes('-alpha', 1)
root.geometry('500x450')

canvas = tk.Canvas(root, height=500, width=450)
canvas.pack()

frame = tk.Frame(root)
frame.place(rely=0.1, relwidth=1, relheight=1)

check_btn = tk.Button(frame, text='Проверить', bg='yellow', command=btn_check)
check_btn.pack()

title = tk.Label(frame, text='Введите название файла, куда всё запишется', font=40)
title.pack()
fileInput = tk.Entry(frame, bg='white')
fileInput.pack()

btn_crystal = tk.Button(frame, text='Преобразовать по кристаллу', bg='yellow', command=btn_click_crystal)
btn_crystal.pack()

btn_glossy = tk.Button(frame, text='Преобразовать по глянцу', bg='yellow', command=btn_click_glossy)
btn_glossy.pack()

info = tk.Label(frame, text='Числовой код:', font=40)
info.pack()

# Связываем событие изменения размера окна с функцией on_window_resize
root.bind("<Configure>", on_window_resize)

root.mainloop()