import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import imageio

# Загрузка данных
iris = load_iris()
X = iris.data

# Метод локтя для определения оптимального количества кластеров
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Построение графика метода локтя
plt.figure(figsize=(8, 6))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Количество кластеров')
plt.ylabel('Инерция')
plt.title('Метод локтя для определения оптимального количества кластеров')
plt.show()

# Оптимальное количество кластеров (например, 3)
k = 3

# Инициализация центроидов случайным образом
np.random.seed(42)
centroids = X[np.random.choice(X.shape[0], k, replace=False)]

# Функция для расчета расстояния между точками
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Функция для назначения кластеров
def assign_clusters(X, centroids):
    clusters = []
    for x in X:
        distances = [euclidean_distance(x, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)

# Функция для обновления центроидов
def update_centroids(X, clusters, k):
    new_centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        points = X[clusters == i]
        new_centroids[i] = points.mean(axis=0)
    return new_centroids

# Функция для визуализации
def plot_clusters(X, clusters, centroids, step):
    plt.figure(figsize=(8, 6))
    for i in range(k):
        points = X[clusters == i]
        plt.scatter(points[:, 0], points[:, 1], label=f'Cluster {i}')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')
    plt.title(f'Step {step}')
    plt.legend()
    plt.savefig(f'step_{step}.png')
    plt.close()

# Основной цикл k-means
max_iterations = 100
step = 0
images = []

for _ in range(max_iterations):
    clusters = assign_clusters(X, centroids)
    new_centroids = update_centroids(X, clusters, k)

    # Проверка на сходимость
    if np.all(centroids == new_centroids):
        break

    centroids = new_centroids
    step += 1

    # Визуализация
    plot_clusters(X, clusters, centroids, step)
    images.append(imageio.imread(f'step_{step}.png'))

# Создание GIF
imageio.mimsave('kmeans_process.gif', images, fps=1)
