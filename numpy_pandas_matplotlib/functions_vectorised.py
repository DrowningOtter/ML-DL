import numpy as np
from typing import Tuple


def sum_non_neg_diag(X: np.ndarray) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    d = np.diagonal(X).copy()
    mask = d >= 0
    if not mask.any():
        return -1
    d[~mask] = 0
    return d.sum()


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    x.sort(), y.sort()
    return (x == y).all()


def max_prod_mod_3(x: np.ndarray) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    work = np.vstack((x[:-1], x[1:]))
    prd = np.prod(work, axis=0)
    mask = prd % 3 == 0
    if mask.any():
        prd = np.delete(prd, ~mask)
        return prd.max()
    return -1


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Сложить каналы изображения с указанными весами.
    """
    return np.einsum('ijk,k->ij', image, weights)


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    x = np.repeat(x[:, 0], x[:, 1])
    y = np.repeat(y[:, 0], y[:, 1])
    try:
        return np.dot(x, y)
    except ValueError:
        return -1


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y.
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    prod = X.dot(Y.T)
    left = np.array(np.linalg.norm(X, axis=1))[:, np.newaxis]
    right = np.array(np.linalg.norm(Y, axis=1))[np.newaxis, :]
    norms = left.dot(right)
    mask = norms == 0
    prod[mask] = 1
    norms[mask] = 1
    return prod / norms