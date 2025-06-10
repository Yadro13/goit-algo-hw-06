import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
import heapq

# 🔹 Лінії метро
blue_line = [
    "Героїв Дніпра", "Мінська", "Оболонь", "Почайна", "Тараса Шевченка",
    "Контрактова площа", "Поштова площа", "Майдан Незалежності", "Площа Льва Толстого",
    "Олімпійська", "Палац Україна", "Либідська", "Деміївська", "Голосіївська",
    "Васильківська", "Виставковий центр", "Іподром", "Теремки"
]

red_line = [
    "Академмістечко", "Житомирська", "Святошин", "Нивки", "Берестейська",
    "Шулявська", "Політехнічний інститут", "Вокзальна", "Університет",
    "Театральна", "Хрещатик", "Арсенальна", "Дніпро", "Гідропарк", "Лівобережна",
    "Дарниця", "Чернігівська", "Лісова"
]

green_line = [
    "Сирець", "Дорогожичі", "Лук'янівська", "Золоті ворота", "Палац спорту",
    "Кловська", "Печерська", "Дружби народів", "Видубичі", "Славутич",
    "Осокорки", "Позняки", "Харківська", "Вирлиця", "Бориспільська", "Червоний хутір"
]

# Пересадки між лініями
transfers = [
    ("Золоті ворота", "Театральна"),
    ("Майдан Незалежності", "Хрещатик"),
    ("Палац спорту", "Площа Льва Толстого")
]

# Координати розміщення станцій
station_positions = {
    # Red line
    "Академмістечко": (-1, 16.5), "Житомирська": (0, 15.5), "Святошин": (1, 14.5),
    "Нивки": (2, 13.5), "Берестейська": (3, 12.5), "Шулявська": (4, 11.5),
    "Політехнічний інститут": (5, 11), "Вокзальна": (6, 10.5),
    "Університет": (7, 10), "Театральна": (8, 9.6), "Хрещатик": (9.4, 9.8),
    "Арсенальна": (11, 9.5), "Дніпро": (12, 10.5), "Гідропарк": (13, 11),
    "Лівобережна": (14, 11.5), "Дарниця": (15, 12), "Чернігівська": (16, 13),
    "Лісова": (17, 14),

    # Blue line
    "Героїв Дніпра": (10, 17), "Мінська": (10, 16), "Оболонь": (10, 15),
    "Почайна": (10, 14), "Тараса Шевченка": (10, 13), "Контрактова площа": (10, 12),
    "Поштова площа": (10, 11), "Майдан Незалежності": (9.8, 9.3),
    "Площа Льва Толстого": (9.0, 8.4), "Олімпійська": (8.5, 7.3),
    "Палац Україна": (8, 6), "Либідська": (8, 5), "Деміївська": (8, 4),
    "Голосіївська": (8, 3), "Васильківська": (8, 2),
    "Виставковий центр": (8, 1), "Іподром": (8, 0), "Теремки": (8, -1),

    # Green line
    "Сирець": (6, 14), "Дорогожичі": (7, 13), "Лук'янівська": (8, 12),
    "Золоті ворота": (8.6, 10.3), "Палац спорту": (9.6, 8.4), "Кловська": (10.1, 7.6),
    "Печерська": (11, 6.8), "Дружби народів": (11.8, 6), "Видубичі": (12.5, 5),
    "Славутич": (13, 4.2), "Осокорки": (13.8, 3.4), "Позняки": (14.6, 2.6),
    "Харківська": (15.4, 1.8), "Вирлиця": (16.2, 1), "Бориспільська": (17, 0.5),
    "Червоний хутір": (17.7, 0)
}

# Підписи станцій (з індивідуальними зсувами для пересадок)
label_offsets = {
    "Театральна": (0.2, -0.5), "Золоті ворота": (0.5, 0.4),
    "Хрещатик": (0.5, 0.2), "Майдан Незалежності": (1, -0.3),
    "Палац спорту": (0.7, 0), "Площа Льва Толстого": (-1.0, -0.1)
}

# Функція для пошуку найкоротших шляхів за допомогою алгоритму Дейкстри
# heapq	для швидкісти - Складність O(log n) VS O(n) для звичайного списку
def dijkstra_all_paths(graph, start):

    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0
    paths = {node: [] for node in graph.nodes}
    paths[start] = [start]

    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        for neighbor in graph.neighbors(current_node):
            weight = graph[current_node][neighbor].get('weight', 1)
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                paths[neighbor] = paths[current_node] + [neighbor]
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances, paths

# Функції для пошуку шляхів за допомогою DFS
def dfs_path(graph, start, goal, path=None, visited=None):

    if visited is None:
        visited = set()
    if path is None:
        path = []

    visited.add(start)
    path = path + [start]

    if start == goal:
        return path

    for neighbor in graph.neighbors(start):
        if neighbor not in visited:
            new_path = dfs_path(graph, neighbor, goal, path, visited)
            if new_path:
                return new_path
    return None

# Функція для пошуку шляху за допомогою BFS
def bfs_path(graph, start, goal):

    visited = set()
    queue = deque([[start]])

    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path
        if node not in visited:
            visited.add(node)
            for neighbor in graph.neighbors(node):
                new_path = path + [neighbor]
                queue.append(new_path)
    return None

# Функція для додавання ребер з вагами
def add_weighted_edges_realistic(line, weight=2):

    for i in range(len(line) - 1):
        GKM.add_edge(line[i], line[i + 1], weight=weight)

# Функція для додавання ребер між станціями лінії
def add_line_edges(line):
    for i in range(len(line) - 1):
        GKM.add_edge(line[i], line[i + 1], line='main')


# Створення графа
GKM = nx.Graph()

add_line_edges(blue_line)
add_line_edges(red_line)
add_line_edges(green_line)
GKM.add_edges_from(transfers, line='transfer')

# 🔹 Кольори вершин
color_map = []
for node in GKM.nodes():
    if node in blue_line:
        color_map.append("dodgerblue")
    elif node in red_line:
        color_map.append("crimson")
    elif node in green_line:
        color_map.append("green")
    else:
        color_map.append("gray")

default_offset_x = 0.2
default_offset_y = 0.3
red_label_offset_y = 0.6

# 🔹 Візуалізація
plt.figure(figsize=(20, 12))
nx.draw_networkx_nodes(GKM, station_positions, node_color=color_map, node_size=200)

main_edges = [(u, v) for u, v, d in GKM.edges(data=True) if d["line"] == "main"]
transfer_edges = [(u, v) for u, v, d in GKM.edges(data=True) if d["line"] == "transfer"]

nx.draw_networkx_edges(GKM, station_positions, edgelist=main_edges, width=2)
nx.draw_networkx_edges(GKM, station_positions, edgelist=transfer_edges, style="dashed", width=2, edge_color="black")

# Підписи
for station, (x, y) in station_positions.items():
    if station in label_offsets:
        dx, dy = label_offsets[station]
        plt.text(x + dx, y + dy, station, fontsize=8, ha='center', va='center')
    elif station in red_line:
        plt.text(x, y + red_label_offset_y, station, fontsize=8, ha='center', va='bottom')
    else:
        plt.text(x + default_offset_x, y + default_offset_y, station, fontsize=8, ha='left', va='center')

# Додаємо основні лінії з вагою 2 хв
add_weighted_edges_realistic(blue_line, weight=2)
add_weighted_edges_realistic(red_line, weight=2)
add_weighted_edges_realistic(green_line, weight=2)

# Додавання ваг ребер до візуалізації
edge_labels = nx.get_edge_attributes(GKM, 'weight')

nx.draw_networkx_edge_labels(
    GKM,
    pos=station_positions,
    edge_labels=edge_labels,
    font_size=8,
    label_pos=0.5  # розташування мітки по середині ребра
)

# Візуалізація
plt.title("Київське метро — ну як би :)", fontsize=16)
plt.axis("off")
plt.tight_layout()
plt.show()

# Аналіз характеристик
print("Загальна кількість станцій:", GKM.number_of_nodes())
print("Загальна кількість з'єднань (ребер):", GKM.number_of_edges())
print("Середній ступінь вершин:", sum(dict(GKM.degree()).values()) / GKM.number_of_nodes())
print("Топ-5 станцій з найбільшим ступенем (пересадки):")
for node, deg in sorted(GKM.degree, key=lambda x: x[1], reverse=True)[:5]:
    print(f"  - {node}: {deg}")

dfs_result = dfs_path(GKM, "Теремки", "Лісова")
bfs_result = bfs_path(GKM, "Теремки", "Лісова")

print("\nШлях за допомогою DFS:", " -> ".join(dfs_result) if dfs_result else "Шлях не знайдено")
print("...\n")
print("Шлях за допомогою BFS:", " -> ".join(bfs_result) if bfs_result else "Шлях не знайдено")

# Пересадки — 3 хв
for u, v in transfers:
    GKM.add_edge(u, v, weight=3)


# Приклад: отримати найкоротший час між двома станціями
start = "Теремки"
end = "Лісова"
shortest_time_start, route = dijkstra_all_paths(GKM, start)
shortest_time = shortest_time_start[end]

print("...\n")
print(f"Час поїздки з {start} до {end} за Дейкстрою: {shortest_time} хв")
print("Маршрут за Дейкстрою:", " -> ".join(route[end]) if end in route else "Маршрут не знайдено")