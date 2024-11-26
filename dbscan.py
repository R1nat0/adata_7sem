import pygame
import numpy as np

# Инициализация pygame
pygame.init()

# Настройки окна
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('DBSCAN Visualization')

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# Список точек
points = []

# Функция для рисования точек
def draw_points(points, color=BLACK):
    for point in points:
        pygame.draw.circle(screen, color, point, 5)

# Функция для расчета евклидова расстояния
def euclidean_distance(a, b):
    return np.sqrt(np.sum((np.array(a) - np.array(b)) ** 2))

# Функция для поиска соседей
def region_query(points, point, eps):
    neighbors = []
    for p in points:
        if euclidean_distance(point, p) <= eps:
            neighbors.append(p)
    return neighbors

# Функция для расширения кластера
def expand_cluster(points, labels, point, neighbors, cluster_id, eps, min_pts):
    labels[points.index(point)] = cluster_id
    i = 0
    while i < len(neighbors):
        neighbor_point = neighbors[i]
        if labels[points.index(neighbor_point)] == -1:
            labels[points.index(neighbor_point)] = cluster_id
        elif labels[points.index(neighbor_point)] == 0:
            labels[points.index(neighbor_point)] = cluster_id
            neighbor_neighbors = region_query(points, neighbor_point, eps)
            if len(neighbor_neighbors) >= min_pts:
                neighbors.extend(neighbor_neighbors)
        i += 1

# Функция для выполнения алгоритма DBSCAN
def dbscan(points, eps, min_pts):
    labels = [0] * len(points)
    cluster_id = 0

    for idx, point in enumerate(points):
        if labels[idx] != 0:
            continue

        neighbors = region_query(points, point, eps)
        if len(neighbors) < min_pts:
            labels[idx] = -1  # Шум
        else:
            cluster_id += 1
            expand_cluster(points, labels, point, neighbors, cluster_id, eps, min_pts)

    return labels

# Основной цикл для рисования точек
running = True
drawing = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Левая кнопка мыши
                points.append(event.pos)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:  # Нажатие Enter
                drawing = False
            elif event.key == pygame.K_SPACE:  # Нажатие Space для завершения
                running = False

    screen.fill(WHITE)
    draw_points(points)
    pygame.display.flip()

# Параметры DBSCAN
eps = 20
min_pts = 3

# Применение DBSCAN
labels = dbscan(points, eps, min_pts)

# Цвета для кластеров
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

# Функция для рисования точек с метками
def draw_labeled_points(points, labels):
    for idx, point in enumerate(points):
        if labels[idx] == -1:
            color = BLACK  # Шум
        else:
            color = colors[labels[idx] % len(colors)]
        pygame.draw.circle(screen, color, point, 5)

# Основной цикл для визуализации результатов
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)
    draw_labeled_points(points, labels)
    pygame.display.flip()

pygame.quit()
