'''
Модуль выравнивания изображений. Незапускаемый файл.
'''
import numpy as np
from tqdm import tqdm
from enum import Enum
import cv2
import os
from pathlib import Path

class CropType(Enum):
    NONE = 0
    INNER = 1
    OUTER = 2

def create_image_pyramid(img: cv2.Mat, depth=5):
    pyramid = []
    for _ in range(depth):
        pyramid.append(img)
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
    return pyramid[::-1]

def get_warp(
    img1: cv2.Mat, # изображение-шаблон
    img2: cv2.Mat  # изображение, для которого вычисляется матрица 
                   # преобразований
) -> np.array:
    '''
    Вычисляет матрицу афинных преобразований между `img1` и `img2`

    Вид warp_mat:

    [ cos(theta), sin(theta), X]

    [-sin(theta), cos(theta), Y]

    [0,           0,          1]
    '''

    if len(img1.shape) == 3:
        imga = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    if len(img2.shape) == 3:
        imgb = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    warpMatrix=np.eye(3, 3, dtype=np.float32)

    _, warp_matrix = cv2.findTransformECC(
        templateImage=imga, inputImage=imgb,
        warpMatrix=warpMatrix, motionType=cv2.MOTION_HOMOGRAPHY
    )
    return warp_matrix 

def create_warp_stack(
    images: [] # список исзодных изображений
) -> np.array:
    '''
    Формирует стек матриц перспективных преобразований 
    для каждого кадра (исключая первый)
    '''
    warp_stack = []
    '''
    Будем вычислять разницу между двумя соседними кадрами -- они более
    похожи друг на друга, чем если бы мы сравнивали первый кадр 
    со всеми остальными
    '''
    prev = images[0]
    
    for i in range(1, len(images)):
        curr = images[i]
        warp_stack += [get_warp(curr, prev)]
        prev = curr

    return np.array(warp_stack)

def homography_gen(
    warp_stack: np.array # массив матриц преобразования между 
                         # соседними кадрами
):
    '''
    Умножает с накоплением и возвращает матрицу перобразований.
    Таким образом для каждого кадра учитываются сдвиги предыдущих 
    кадров относительно первого.
    '''
    ws_homography = warp_stack

    H = [np.eye(3, 3)]
    warp = np.eye(3, 3)

    for w in ws_homography:
        warp = np.matmul(w, warp)
        H.append(warp)

    return H


def get_shift(
    h: float, # исходная высота исображения
    w: float  # ширина
) -> tuple:
    '''
    Вычисляет 3% сдвиг границ кадра.
    Фунцкия используется для определения границ обрезки.
    '''
    a = max(h, w)
    b = min(h, w)

    percent = 0.01

    top    = int(percent * a)
    bottom = int((1-percent) * a)
    left   = int(percent * b)
    right  = int((1-percent) * b)

    return top, bottom, left, right



def align_ecc(
    images: [cv2.Mat],  # список исходных изображеник
    filenames,          # список названий выровненных изображений 
    out: str            # каталог, в который будут сохранены 
                        # изображения
) -> [cv2.Mat]: 
    '''
    Выравнивает исходные изображения `images` с помощью ECC.
    Кадрирует результат.
    Сохраняет результат в путь `out`.
    Возвращает список выровненных изображений.
    '''    
    warp_stack = create_warp_stack(images)
    
    h = int(images[0].shape[0])
    w = int(images[0].shape[1])

    H = homography_gen(warp_stack)
    
    '''
    Кадрирование необходимо для того, чтобы не было пустых участков
    в результате выравнивания
    '''
    bottom, top, right, left = get_shift(h, w)
    bottom, top, right, left = -bottom, -top, -right, -left

    new_w = w+left+right
    new_h = h+top+bottom

    aligned = []

    for i in range(0, len(images)):
        img = images[i]
        
        shift = np.array([
            [0, 0, left],
            [0, 0, top],
            [0, 0, 0.0]
        ])
        warp = H[i]
        warp = warp+shift

        img_warp = cv2.warpPerspective(img, warp, (new_w, new_h))
        aligned.append(img_warp)
        cv2.imwrite(out+filenames[i], img_warp)

    return aligned

def align_pyramids(
    images: [cv2.Mat],  # список исходных изображеник
    filenames,          # список названий выровненных изображений 
    out: Path,           # каталог, в который будут сохранены 
                        # изображения
    depth=5             # глубина пирамиды
) -> cv2.Mat:
    pyramids = [create_image_pyramid(img, depth) for img in images]
    levels = [list(level) for level in zip(*pyramids)]
    os.makedirs(out, exist_ok=True)
    
    warp_stack = []
    for i, level in tqdm(enumerate(levels)):
        if i > 0:
            h, w = level[0].shape[:2]
            for j, warp in enumerate(warp_stack):
                level[j] = cv2.warpPerspective(level[j], warp, (w, h))
                
        cur_warp_stack = create_warp_stack(level)
        cur_warp_stack = homography_gen(cur_warp_stack)
        
        for j in range(len(warp_stack)):
            warp_stack[j] = cur_warp_stack[j] @ warp_stack[j]
            
        if len(warp_stack) == 0:
            warp_stack = cur_warp_stack
    
    h, w = images[0].shape[:2]
    top, bottom, left, right = get_shift(h, w)
    aligned = []
    for img, warp, name in zip(images, warp_stack, filenames):
        warped = cv2.warpPerspective(img, warp, (w, h))
        aligned.append(warped[top:bottom, left:right])
        cv2.imwrite(out / name, aligned[-1])
    
    return aligned
    

def align_mtb(
    images: [cv2.Mat],  # список исходных изображеник
    filenames,          # список названий выровненных изображений 
    out: str            # каталог, в который будут сохранены 
                        # изображения
) -> [cv2.Mat]: 
    '''
    Выравнивает исходные изображения `images` с помощью MTB.
    Кадрирует результат.
    Сохраняет результат в путь '`out`'
    Возвращает список выровненных изображений
    '''
    h = int(images[0].shape[0])
    w = int(images[0].shape[1])
    
    """ 
    Кадрирование -- необходимо для того, чтобы не было пустых 
    участков в результате выравнивания
    """
    bottom, top, right, left = get_shift(h, w)
    bottom, top, right, left = -bottom, -top, -right, -left

    new_w = w+left+right
    new_h = h+top+bottom

    alignMTB = cv2.createAlignMTB()
    alignMTB.process(images, images)

    aligned = []

    for i in range(0, len(images)):
        img = images[i]
        
        shift = np.array([
            [0, 0, left],
            [0, 0, top],
            [0, 0, 0.0]
        ])
        warp = np.eye(3, 3)
        warp = warp+shift

        img_warp = cv2.warpPerspective(img,warp,(new_w, new_h))
        aligned.append(img_warp)
        cv2.imwrite(out+filenames[i], img_warp)

    return aligned


if __name__ == '__main__':
    # img = cv2.imread(r"D:\win\Documents\University\BACH\Diploma_linux\HDR\project\src\158\[1.0].(0.003632064).17-06-2023-16-46-06-848.jpg")
    dir = Path(r'D:\win\Documents\University\BACH\Diploma_linux\HDR\project\src\158')
    aligned_dir_root = Path(r'D:\win\Documents\University\MAGA\Diploma\dataset\aligned')
    
    images = [cv2.imread(dir / name) for name in os.listdir(dir)]
    images = sorted(images, key=lambda img: img.mean())
    align_pyramids(
        images, 
        [f'{i}.jpg' for i in range(len(images))], 
        str(aligned_dir_root / dir.stem) + '/'
    )
    pass