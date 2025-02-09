import sys
from pathlib import Path

import cv2
import numpy as np
import pygame
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

from numbs.digits import model


class Point:
    def __init__(self, x: float, y: float, label: int = 0):
        self.x: float = x
        self.y: float = y
        self.label = label

    def draw(self, scr: pygame.Surface):
        pygame.draw.circle(scr, self.label, (self.x, self.y), 1)


def update_display():
    """Update pygame display."""
    pygame.display.update()


def generate_colors(n):
    """Generate colors for clusters."""
    return [tuple(int(c * 255) for c in plt.cm.tab10(i % 10)[:3]) for i in range(n)]


def recognize_digit(directory: str = "."):
    """Recognize all .png images in passed directory."""
    number: list[str] = []
    images: list[Path] = sorted(
        Path(directory).glob("*.png"), key=lambda x: x.stat().st_ctime
    )

    for image in images:
        image = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(image, (28, 28))
        normalized_image = resized_image / 255

        input_image = normalized_image.reshape(1, 28, 28)

        reshaped_for_display = input_image[0]
        plt.imshow(reshaped_for_display, cmap="gray")
        plt.title("Изображение, которое подается на вход модели")
        # plt.axis('off')
        plt.show()

        prediction = model.predict(input_image)
        predicted_label = np.argmax(prediction, axis=-1)
        number.append(str(predicted_label[0]))

    print(f"Написанное число: {''.join(number)}")


def redraw():
    """Redraw all points with new colors."""
    screen.fill("#FFFFFF")
    for p, label in zip(points, labels):
        if label != -1:  # Ignore noise points
            p.label = colors[label]
            p.draw(screen)
    update_display()


def save_cluster_images():
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            # noise in DBSCAN
            continue

        # Get points belonging to the current cluster
        cluster_points = [p for p, l in zip(points, labels) if l == label]

        # Determine bounding box for the cluster
        min_x = min(p.x for p in cluster_points) - 10
        max_x = max(p.x for p in cluster_points) + 10
        min_y = min(p.y for p in cluster_points) - 10
        max_y = max(p.y for p in cluster_points) + 10

        # Create a surface for the cluster
        cluster_surface = pygame.Surface(
            (max_x - min_x + 1, max_y - min_y + 1), pygame.SRCALPHA
        )

        # Draw the cluster points onto the surface
        for p in cluster_points:
            cluster_surface.set_at(
                (p.x - min_x, p.y - min_y), (255, 255, 255)  # noqa
            )

        # Saving image in current directory
        pygame.image.save(cluster_surface, f"clusters_{label}.png")


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((28 * 10, 30), pygame.RESIZABLE)
    screen.fill("#FFFFFF")
    update_display()

    points: list[Point] = []
    labels: list[int] = []
    colors: list[tuple] = [(0, 0, 0)]

    is_drawing: bool = False
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == pygame.BUTTON_LEFT:
                    is_drawing = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == pygame.BUTTON_LEFT:
                    is_drawing = False

            elif event.type == pygame.MOUSEMOTION and is_drawing is True:
                point = Point(*event.pos)
                points.append(point)
                point.draw(screen)

                update_display()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    dbscan = DBSCAN(eps=5, min_samples=5)
                    dbscan.fit(np.array([(p.x, p.y) for p in points]))
                    labels = dbscan.labels_

                    colors = generate_colors(np.max(labels) + 2)
                    redraw()

                elif event.key == pygame.K_ESCAPE:
                    save_cluster_images()
                    recognize_digit()
