import random
from pprint import pprint

import matplotlib.pyplot as plt
import math

import requests

new_cities: dict[str, tuple[float, float]] = {
    "A": (0, 0), "B": (0, 1), "C": (1, 0), "D": (1, 1),
    "E": (1, 2), "F": (2, 0), "G": (2, 1), "H": (2, 2),
    "I": (3, 1), "J": (4, 0), "K": (4, 2), "L": (5, 1),
    "M": (3, 3), "N": (4, 3), "O": (5, 3), "P": (6, 1),
    "Q": (7, 0), "R": (7, 2), "S": (8, 1), "T": (9, 1)
}

API_KEY: str = "5e2dab25-1fc0-4de0-8ae6-7d5ee1db4f1b"


def get_city_coordinates(city: str) -> tuple[float, float]:
    response = requests.get(f"https://geocode-maps.yandex.ru/1.x/?apikey={API_KEY}&geocode={city}&format=json")
    data: dict = response.json()

    height, width = (data.get("response").get("GeoObjectCollection").get("featureMember")[0]
                     .get("GeoObject").get("Point").get("pos").split())
    height, width = float(height), float(width)

    return height, width


def get_cities_matrix(cities: list[str]) -> dict[str, tuple[float, float]]:
    matrix: dict[str, tuple[float, float]] = {}
    for city in cities:
        matrix[city] = get_city_coordinates(city)

    return matrix


class GeneticSalesmanSolver:
    def __init__(self, cities: dict[str, tuple[float, float]]):
        self.cities: dict[str, tuple[float, float]] = cities

    @staticmethod
    def get_distance(city1: tuple[float, float], city2: tuple[float, float]) -> float:
        return math.sqrt(
            (city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2
        )

    def get_route_length(self, route: list[str]) -> float:
        total_distance: float = 0
        for i in range(len(route)):
            total_distance += self.get_distance(
                city1=self.cities[route[i]],
                city2=self.cities[route[(i + 1) % len(route)]]
            )

        return total_distance

    def create_initial_population(self, size: int) -> list[list[str]]:
        population = []
        city_names: list[str] = list(self.cities.keys())
        for _ in range(size):
            route: list[str] = city_names[:]
            random.shuffle(route)
            population.append(route)

        return population

    @staticmethod
    def crossover(
            parent_route1: list[str], parent_route2: list[str]
    ) -> list[str]:
        start, end = sorted(
            random.sample(
                population=range(len(parent_route1)),
                k=2
            )  # отбираем k случайных элементов
        )
        child = parent_route1[start:end + 1]
        child += [gene for gene in parent_route2 if gene not in child]

        return child

    @staticmethod
    def mutate(route: list[str]) -> None:
        """Случайные перестановки двух генов."""

        a, b = random.sample(
            population=range(len(route)),
            k=2
        )
        route[a], route[b] = route[b], route[a]

    def get_best_route(
            self,
            population_size: int = 100,
            generations: int = 500,
            mutation_rate: float = 0.1
    ) -> tuple[list[str], float]:
        population: list[list[str]] = self.create_initial_population(size=population_size)
        best_route: list[str] = min(population, key=self.get_route_length)
        best_distance = self.get_route_length(best_route)

        for generation in range(generations):
            new_population: list[list[str]] = []
            sorted_population: list[list[str]] = sorted(population, key=self.get_route_length)

            best_route_in_generation: list[str] = sorted_population[0]
            best_distance_in_generation: float = self.get_route_length(
                route=best_route_in_generation
            )
            if best_distance_in_generation < best_distance:
                best_route, best_distance = best_route_in_generation, best_distance_in_generation

            # сохраняем лучшие маршруты (первые 20% от всех)
            new_population.extend(sorted_population[:population_size // 5])

            while len(new_population) < population_size:
                parent1, parent2 = random.sample(sorted_population[:50], 2)
                child: list[str] = self.crossover(parent1, parent2)
                if random.random() < mutation_rate:
                    self.mutate(child)

                new_population.append(child)

            population = new_population

        return best_route, best_distance

    def plot_route(self, route: list[str]):
        x: list[float] = [self.cities[city][0] for city in route + [route[0]]]
        y: list[float] = [self.cities[city][1] for city in route + [route[0]]]

        plt.figure(figsize=(10, 10))
        plt.plot(x, y, marker='o', linestyle='-', color='b')
        for city, coord in self.cities.items():
            plt.text(coord[0] + 0.05, coord[1] + 0.05, city, fontsize=12, ha='center', va='center', color='red')

        plt.title("Route")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")

        plt.show()


if __name__ == "__main__":
    cities_matrix = get_cities_matrix([
        "Казань", "Самара", "Ижевск", "Альметьевск", "Нижнекамск", "Димитровград", "Нурлат"
    ])

    solver: GeneticSalesmanSolver = GeneticSalesmanSolver(cities=cities_matrix)
    best_r, best_l = solver.get_best_route()
    solver.plot_route(route=best_r)

    print(f"Best route: {' -> '.join(best_r)}")