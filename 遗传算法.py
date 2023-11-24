"""
作者：zhaozhijie
日期：2022年03月16日10时21分
"""
import math
import os
import random
import copy
import csv
import sys
import numpy as np
import collections
import xlsxwriter
import matplotlib.pyplot as plt
import time


class Sol:
    def __init__(self):
        self.obj = None
        self.node_id_list = []  # TSP的解
        self.cost_of_distance = None
        self.cost_of_time = None
        self.fitness = None
        self.route_list = []  # SBRP的解
        self.timetable = []  # 时间表
        self.vehicle_schedule = []  # 存储车辆调度信息
        self.num_vehicles = 0


class Node:
    def __init__(self):
        self.id = None
        self.x_coord = 0
        self.y_coord = 0
        self.demand = None
        self.depot_capacity = None
        self.start_time = 0
        self.end_time = 0
        self.service_time = None


class Vehicle:
    def __init__(self):
        self.id = None
        self.capacity = 0


class Model:
    def __init__(self):
        self.best_sol = None
        self.sol_list = []
        self.demand_dict = {}
        self.demand_id_list = []
        self.depot_dict = {}
        self.depot_id_list = []
        self.school_dict = {}
        self.school_id_list = []
        self.school_sequence = []
        self.school_id_sequence = []
        self.distance_matrix = {}
        self.time_matrix = {}
        self.number_of_demand = 0
        self.opt_type = 0
        self.distance_opt = 0
        self.vehicle_assignment_mode = 0
        self.vehicle_id_list = []
        self.vehicles_capacity = []
        self.vehicle_dict = {}
        self.vehicle_num = 0
        self.vehicle_speed = 1
        self.pc = 0.5
        self.pm = 0.1
        self.popsize = 100
        self.n_select = 80
        self.time = None


def read_csv_file(demand_file, depot_file, school_file, vehicle_file, model):
    with open(depot_file, "r") as f:
        depot_reader = csv.DictReader(f)
        for row in depot_reader:
            node = Node()
            node.id = row["id"]
            node.x_coord = float(row["x_coord"])
            node.y_coord = float(row["y_coord"])
            node.depot_capacity = int(row["capacity"])
            node.start_time = int(row["start_time"])
            node.end_time = int(row["end_time"])
            model.depot_id_list.append(node.id)
            model.depot_dict[node.id] = node

    with open(school_file, "r") as f:
        school_reader = csv.DictReader(f)
        for row in school_reader:
            node = Node()
            node.id = row["id"]
            node.x_coord = float(row["x_coord"])
            node.y_coord = float(row["y_coord"])
            node.start_time = int(row["start_time"])
            node.end_time = int(row["end_time"])
            node.service_time = int(row["service_time"])
            model.school_id_list.append(node.id)
            model.school_dict[node.id] = node

    with open(vehicle_file, "r") as f:
        vehicle_reader = csv.DictReader(f)
        for row in vehicle_reader:
            vehicle = Vehicle()
            vehicle.id = row["id"]
            vehicle.capacity = int(row["capacity"])
            model.vehicle_id_list.append(vehicle.id)
            model.vehicles_capacity.append(vehicle.capacity)
            model.vehicle_dict[vehicle.id] = vehicle
        model.vehicle_num = len(model.vehicle_dict)

    with open(demand_file, "r") as f:
        demand_reader = csv.DictReader(f)
        for row in demand_reader:
            demand = int(row["demand"])
            if demand <= max(model.vehicles_capacity):
                node = Node()
                node.id = int(row["id"])
                node.x_coord = float(row["x_coord"])
                node.y_coord = float(row["y_coord"])
                node.demand = int(row["demand"])
                node.start_time = int(row["start_time"])
                node.end_time = int(row["end_time"])
                node.service_time = int(row["service_time"])
                model.demand_id_list.append(node.id)
                model.demand_dict[node.id] = node
            else:
                if model.vehicle_assignment_mode == 0:          # 模式0不会违反车辆限制
                    if not demand % max(model.vehicles_capacity):           # 当demand=最大车辆容量的倍数时
                        n = demand // max(model.vehicles_capacity)          # 虚拟节点数量
                    else:
                        n = demand // max(model.vehicles_capacity)+1
                else:                                           # 模式1要尽可能少的使用容量最大的校车
                    n = demand // max(model.vehicles_capacity)+1
                common_demand = demand // n                             # 尽可能的平均每个虚拟节点的demand
                special_demand = demand // n + demand % n               # common和special的demand差不会超过n
                for i in range(n):
                    node = Node()
                    node.x_coord = float(row["x_coord"])
                    node.y_coord = float(row["y_coord"])
                    if i == 0:
                        node.id = int(row["id"])
                        node.demand = special_demand
                    else:
                        node.id = int('-'+str(i)+row["id"])
                        node.demand = common_demand
                    node.start_time = int(row["start_time"])
                    node.end_time = int(row["end_time"])
                    node.service_time = int(row["service_time"])
                    model.demand_id_list.append(node.id)
                    model.demand_dict[node.id] = node
        model.number_of_demand = len(model.demand_id_list)


def cal_school_sequence(model):
    school_id_sequence = []
    school_sequence = []
    _ = {}
    for school in model.school_dict.values():
        _[school] = school.end_time
    while _:
        for key, value in _.items():
            if value == min(_.values()):
                school_sequence.append(key)
                school_id_sequence.append(key.id)
                del _[key]
                break

    # 判断学校序列的时间窗可行性
    for i in range(len(school_sequence) - 1):
        if model.distance_opt == 0:
            distance = math.sqrt(
                (school_sequence[i].x_coord - school_sequence[i + 1].x_coord) ** 2
                + (school_sequence[i].y_coord - school_sequence[i + 1].y_coord) ** 2
            )
        elif model.distance_opt == 1:
            distance = abs(school_sequence[i].x_coord - school_sequence[i + 1].x_coord) \
                       + abs(school_sequence[i].y_coord - school_sequence[i + 1].y_coord)

        if (
            school_sequence[i].end_time
            + distance / model.vehicle_speed
            + school_sequence[i + 1].service_time
            > school_sequence[i + 1].end_time
        ):
            print(
                school_sequence[i].end_time,
                distance / model.vehicle_speed,
                school_sequence[i + 1].service_time,
                school_sequence[i + 1].end_time,
            )
            print(f"{school_sequence[i].id}到{school_sequence[i+1].id}的学校序列不可行")
            sys.exit(0)

    # 判断学校序列到车库的时间窗可行性
    from_node = school_sequence[-1]
    for depot in model.depot_dict.values():
        if model.distance_opt == 0:
            distance = math.sqrt(
                (from_node.x_coord - depot.x_coord) ** 2
                + (from_node.y_coord - depot.y_coord) ** 2
            )
        elif model.distance_opt == 1:
            distance = abs(from_node.x_coord - depot.x_coord) + abs(from_node.y_coord - depot.y_coord)
        if from_node.end_time + distance / model.vehicle_speed > depot.end_time:
            print(f"无法分配到{depot.id}车库")
            sys.exit(0)
        # print(from_node.end_time + distance/model.vehicle_speed)

    model.school_id_sequence = school_id_sequence
    model.school_sequence = school_sequence


def cal_time_distance_matrix(model):
    for i in range(len(model.demand_id_list)):
        from_node_id = model.demand_id_list[i]

        school_id = model.school_id_sequence[0]
        if model.distance_opt == 0:
            distance = math.sqrt(
                (
                    model.demand_dict[from_node_id].x_coord
                    - model.school_dict[school_id].x_coord
                )
                ** 2
                + (
                    model.demand_dict[from_node_id].y_coord
                    - model.school_dict[school_id].y_coord
                )
                ** 2
            )
        elif model.distance_opt == 1:
            distance = abs(model.demand_dict[from_node_id].x_coord - model.school_dict[school_id].x_coord) + abs(model.demand_dict[from_node_id].y_coord
                    - model.school_dict[school_id].y_coord)
        model.distance_matrix[school_id, from_node_id] = model.distance_matrix[
            from_node_id, school_id
        ] = distance
        model.time_matrix[school_id, from_node_id] = model.time_matrix[
            from_node_id, school_id
        ] = math.ceil(distance / model.vehicle_speed)

        for j in range(i + 1, len(model.demand_id_list)):
            to_node_id = model.demand_id_list[j]
            if model.distance_opt == 0:
                distance = math.sqrt(
                    (
                        model.demand_dict[from_node_id].x_coord
                        - model.demand_dict[to_node_id].x_coord
                    )
                    ** 2
                    + (
                        model.demand_dict[from_node_id].y_coord
                        - model.demand_dict[to_node_id].y_coord
                    )
                    ** 2
                )
            elif model.distance_opt == 1:
                distance = abs( model.demand_dict[from_node_id].x_coord - model.demand_dict[to_node_id].x_coord) + abs(model.demand_dict[from_node_id].y_coord
                        - model.demand_dict[to_node_id].y_coord)
            model.distance_matrix[from_node_id, to_node_id] = model.distance_matrix[
                to_node_id, from_node_id
            ] = distance
            model.time_matrix[from_node_id, to_node_id] = model.time_matrix[
                to_node_id, from_node_id
            ] = math.ceil(distance / model.vehicle_speed)

        for depot_id, depot in model.depot_dict.items():
            if model.distance_opt == 0:
                distance = math.sqrt(
                    (model.demand_dict[from_node_id].x_coord - depot.x_coord) ** 2
                    + (model.demand_dict[from_node_id].y_coord - depot.y_coord) ** 2
                )
            elif model.distance_opt == 1:
                distance = abs(model.demand_dict[from_node_id].x_coord - depot.x_coord) + abs(model.demand_dict[from_node_id].y_coord - depot.y_coord)
            model.distance_matrix[from_node_id, depot_id] = model.distance_matrix[
                depot_id, from_node_id
            ] = distance
            model.time_matrix[from_node_id, depot_id] = model.time_matrix[
                depot_id, from_node_id
            ] = math.ceil(distance / model.vehicle_speed)

    # 学校和学校之间的时间和距离矩阵
    for i in range(len(model.school_id_list)):
        from_node_id = model.school_id_list[i]
        for j in range(i + 1, len(model.school_id_list)):
            to_node_id = model.school_id_list[j]
            if model.distance_opt == 0:
                distance = math.sqrt(
                    (
                        model.school_dict[from_node_id].x_coord
                        - model.school_dict[to_node_id].x_coord
                    )
                    ** 2
                    + (
                        model.school_dict[from_node_id].y_coord
                        - model.school_dict[to_node_id].y_coord
                    )
                    ** 2
                )
            elif model.distance_opt == 1:
                distance = abs(model.school_dict[from_node_id].x_coord
                        - model.school_dict[to_node_id].x_coord) + abs(model.school_dict[from_node_id].y_coord
                        - model.school_dict[to_node_id].y_coord)
            model.distance_matrix[from_node_id, to_node_id] = model.distance_matrix[
                to_node_id, from_node_id
            ] = distance
            model.time_matrix[from_node_id, to_node_id] = model.time_matrix[
                to_node_id, from_node_id
            ] = math.ceil(distance / model.vehicle_speed)

        # 学校和车场之间的时间和距离矩阵
        for j in range(len(model.depot_id_list)):
            to_node_id = model.depot_id_list[j]
            if model.distance_opt == 0:
                distance = math.sqrt(
                    (
                        model.school_dict[from_node_id].x_coord
                        - model.depot_dict[to_node_id].x_coord
                    )
                    ** 2
                    + (
                        model.school_dict[from_node_id].y_coord
                        - model.depot_dict[to_node_id].y_coord
                    )
                    ** 2
                )
            elif model.distance_opt == 1:
                distance = abs(model.school_dict[from_node_id].x_coord
                        - model.depot_dict[to_node_id].x_coord) + abs(model.school_dict[from_node_id].y_coord
                        - model.depot_dict[to_node_id].y_coord)
            model.distance_matrix[from_node_id, to_node_id] = model.distance_matrix[
                to_node_id, from_node_id
            ] = distance
            model.time_matrix[from_node_id, to_node_id] = model.time_matrix[
                to_node_id, from_node_id
            ] = math.ceil(distance / model.vehicle_speed)


def generate_initial_sol(model):
    demand_id_list = copy.deepcopy(model.demand_id_list)
    for i in range(model.popsize):
        seed = int(random.randint(0, 1001))
        random.seed(seed)
        random.shuffle(demand_id_list)
        sol = Sol()
        sol.node_id_list = copy.deepcopy(demand_id_list)
        model.sol_list.append(sol)


def cal_fitness(model):
    max_obj = -float('inf')
    best_sol = Sol()
    best_sol.obj = float('inf')

    for sol in model.sol_list:
        node_id_list = copy.deepcopy(sol.node_id_list)
        sol.num_vehicles, route_list, vehicle_schedule = split_routes(node_id_list, model)
        if not sol.num_vehicles:
            return 0
        timetable, cost_of_time, cost_of_distance = cal_travel_cost(route_list, model)
        if model.opt_type == 0:
            sol.obj = cost_of_distance + 100000 * sol.num_vehicles
        else:
            sol.obj = cost_of_time
        sol.route_list = route_list
        sol.timetable = timetable
        sol.cost_of_distance = cost_of_distance
        sol.cost_of_time = cost_of_time
        sol.vehicle_schedule = vehicle_schedule
        if sol.obj > max_obj:
            max_obj = sol.obj
        if sol.obj < best_sol.obj:
            best_sol = copy.deepcopy(sol)

    for sol in model.sol_list:
        sol.fitness = max_obj - sol.obj
    if best_sol.obj < model.best_sol.obj:
        model.best_sol = copy.deepcopy(best_sol)
    return 1


def split_routes(node_id_list, model):
    depot = model.depot_id_list[0]
    school = model.school_id_list[0]
    vehicles_capacity = copy.deepcopy(model.vehicles_capacity)
    cumulative_min_cost = {id: float("inf") for id in model.demand_id_list}
    cumulative_min_cost[depot] = 0
    labels = {id: depot for id in model.demand_id_list}
    for i in range(len(node_id_list)):
        n_1 = node_id_list[i]
        demand = 0
        departure = 0
        j = i
        cost = 0
        while True:
            n_2 = node_id_list[j]
            demand = demand + model.demand_dict[n_2].demand
            if n_1 == n_2:
                arrival = max(
                    model.demand_dict[n_2].start_time,
                    model.depot_dict[depot].start_time + model.time_matrix[depot, n_2],
                )
                departure = (
                    arrival
                    + model.demand_dict[n_2].service_time
                    + model.time_matrix[n_2, school]
                )
                if model.opt_type == 0:
                    cost = (
                        model.distance_matrix[depot, n_2]
                        + model.distance_matrix[n_2, school]
                    )
                else:
                    cost = (
                        model.time_matrix[depot, n_2] + model.time_matrix[n_2, school]
                    )

            else:
                n_3 = node_id_list[j - 1]
                arrival = max(
                    departure
                    - model.time_matrix[n_3, school]
                    + model.time_matrix[n_3, n_2],
                    model.demand_dict[n_2].start_time,
                )
                departure = (
                    arrival
                    + model.demand_dict[n_2].service_time
                    + model.time_matrix[n_2, school]
                )
                if model.opt_type == 0:
                    cost = (
                        cost
                        - model.distance_matrix[n_3, school]
                        + model.distance_matrix[n_3, n_2]
                        + model.distance_matrix[n_2, school]
                    )
                else:
                    wait_time = max(
                        0,
                        model.demand_dict[n_2].start_time
                        - (
                            (departure - model.time_matrix[n_3, school])
                            + model.time_matrix[n_3, n_2]
                        ),
                    )
                    cost = (
                        cost
                        - model.time_matrix[n_3, school]
                        + model.time_matrix[n_3, n_2]
                        + model.time_matrix[n_2, school]
                        + wait_time
                    )

            if (
                demand <= vehicles_capacity[-1]
                and departure - model.time_matrix[n_2, school]
                <= model.demand_dict[n_2].end_time
            ):
                if departure <= model.school_dict[school].end_time:
                    n_4 = node_id_list[i - 1] if i - 1 >= 0 else depot
                    if cumulative_min_cost[n_4] + cost <= cumulative_min_cost[n_2]:
                        cumulative_min_cost[n_2] = cumulative_min_cost[n_4] + cost
                        labels[n_2] = i - 1
                    else:
                        break
                    j += 1
                else:
                    break
            else:
                break
            if j == len(node_id_list):
                break
    route_list, vehicles_schedule = extract_routes(node_id_list, labels, model)
    if not route_list:
        return 0, 0, 0
    return len(route_list), route_list, vehicles_schedule


def extract_routes(node_id_list, labels, model):
    routes = []
    route_pure = []
    demand_dict = []
    vehicles_schedule = []
    # -----------------------------
    depot_dict = copy.deepcopy(model.depot_dict)
    route_list = []
    route = []
    label = labels[node_id_list[0]]
    for node_id in node_id_list:
        if labels[node_id] == label:
            route.append(node_id)
            route_pure.append(node_id)
        else:
            route, depot_dict = select_depot(route, depot_dict, model)
            route_list.append(route)
            routes.append(route_pure)
            route = [node_id]
            route_pure = [node_id]
            label = labels[node_id]                               # 被字典坑惨了 以后用列表+元组代替字典
    routes.append(route_pure)
    for i in range(len(routes)):
        demand = 0
        for j in range(len(routes[i])):
            demand += model.demand_dict[routes[i][j]].demand
        demand_dict.append((demand, routes[i]))
    vehicle_schedule = select_vehicle(demand_dict, model)
    if not vehicle_schedule:
        return 0, 0
    route, depot_dict = select_depot(route, depot_dict, model)
    route_list.append(route)
    for i in range(len(vehicle_schedule)):
        vehicles_schedule.append((vehicle_schedule[i], route_list[i]))
    return route_list, vehicles_schedule


def select_vehicle(demand_dict, model):
    vehicle_id_list = copy.deepcopy(model.vehicle_id_list)
    vehicle_schedule = []
    capacity_list = copy.deepcopy(model.vehicles_capacity)
    # print(len(vehicle_id_list))
    # print(demand_dict)
    # print(len(demand_dict))
    # print(capacity_list)
    for i in range(len(demand_dict)):
        if model.vehicle_assignment_mode == 0:
            try:
                index = list(map(lambda x: x >= demand_dict[i][0], capacity_list)).index(True)
                # print('try')
            except ValueError:
                index = -1
                # print('except')
            try:
                vehicle_schedule.append(vehicle_id_list[index])
            except IndexError:
                return 0
            vehicle_id_list.pop(index)
            capacity_list.pop(index)

        elif model.vehicle_assignment_mode == 1:
            index = list(map(lambda x: x >= demand_dict[i][0], capacity_list)).index(True)
            vehicle_schedule.append(vehicle_id_list[index])
    # print(vehicle_id_list)
    # print(vehicle_schedule)
    # print(len(vehicle_schedule))
    # sys.exit()
    return vehicle_schedule


def select_depot(route, depot_dict, model):
    first_node_id = route[0]
    last_school_id = model.school_id_sequence[-1]
    min_distance = float("inf")
    index = None
    for depot_id, depot in depot_dict.items():
        if depot.depot_capacity > 0:
            distance = (
                model.distance_matrix[depot_id, first_node_id]
                + model.distance_matrix[last_school_id, depot_id]
            )
            if distance < min_distance:
                index = depot_id
                min_distance = distance

    if index is None:
        print("没有车辆可分配")
        sys.exit(0)

    route.insert(0, index)
    route.extend(model.school_id_sequence)
    route.append(index)
    depot_dict[index].depot_capacity -= 1
    return route, depot_dict


def cal_travel_cost(route_list, model):
    timetable_list = []
    # cumsum_routes_distance = []
    # routes_distance = []
    cost_of_distance = 0
    cost_of_time = 0
    for route in route_list:
        timetable = []
        for i in range((len(route))):
            if i == 0:
                depot_id = route[i]
                next_node_id = route[i + 1]
                travel_time = model.time_matrix[depot_id, next_node_id]
                departure = max(
                    0, model.demand_dict[next_node_id].start_time - travel_time
                )
                timetable.append((departure, departure))
            elif 1 <= i <= len(route) - 2:
                last_node_id = route[i - 1]
                current_node_id = route[i]
                if type(current_node_id) == int:
                    current_node = model.demand_dict[current_node_id]
                else:
                    current_node = model.school_dict[current_node_id]
                travel_time = model.time_matrix[last_node_id, current_node_id]
                arrival = max(timetable[-1][1] + travel_time, current_node.start_time)
                departure = arrival + current_node.service_time
                timetable.append((arrival, departure))
                cost_of_distance += model.distance_matrix[last_node_id, current_node_id]
                cost_of_time += (
                    model.time_matrix[last_node_id, current_node_id]
                    + current_node.service_time
                    + max(current_node.start_time - timetable[-1][1] + travel_time, 0)
                )
            else:
                last_node_id = route[i - 1]
                depot_id = route[i]
                travel_time = model.time_matrix[last_node_id, depot_id]
                departure = timetable[-1][1] + travel_time
                timetable.append((departure, departure))
                cost_of_distance += model.distance_matrix[last_node_id, depot_id]
                cost_of_time += model.time_matrix[last_node_id, depot_id]
        # cumsum_routes_distance.append(cost_of_distance)
        timetable_list.append(timetable)
    # for i in range(len(cumsum_routes_distance)):
    #     if not i:
    #         routes_distance.append(cumsum_routes_distance[i])
    #     else:
    #         routes_distance.append(cumsum_routes_distance[i]-cumsum_routes_distance[i-1])
    return timetable_list, cost_of_time, cost_of_distance


def select_sol(model):
    sol_list = copy.deepcopy(model.sol_list)
    model.sol_list = []
    for i in range(model.n_select):
        f1_index = random.randint(0, len(sol_list) - 1)
        f2_index = random.randint(0, len(sol_list) - 1)
        f1_fit = sol_list[f1_index].fitness
        f2_fit = sol_list[f2_index].fitness
        if f1_fit < f2_fit:
            model.sol_list.append(sol_list[f2_index])
        else:
            model.sol_list.append(sol_list[f1_index])


def cross_sol(model):
    sol_list = copy.deepcopy(model.sol_list)
    model.sol_list = []
    while True:
        f1_index = random.randint(0, len(sol_list) - 1)
        f2_index = random.randint(0, len(sol_list) - 1)
        if f1_index != f2_index:
            f1 = copy.deepcopy(sol_list[f1_index])
            f2 = copy.deepcopy(sol_list[f2_index])
            if random.random() <= model.pc:
                cro1_index = int(random.randint(0, len(model.demand_id_list) - 1))
                cro2_index = int(random.randint(cro1_index, len(model.demand_id_list) - 1))
                new_c1_f = []
                new_c1_m = f1.node_id_list[cro1_index:cro2_index + 1]
                new_c1_b = []
                new_c2_f = []
                new_c2_m = f2.node_id_list[cro1_index:cro2_index + 1]
                new_c2_b = []
                for index in range(len(model.demand_id_list)):
                    if len(new_c1_f) < cro1_index:
                        if f2.node_id_list[index] not in new_c1_m:
                            new_c1_f.append(f2.node_id_list[index])
                    else:
                        if f2.node_id_list[index] not in new_c1_m:
                            new_c1_b.append(f2.node_id_list[index])
                for index in range(len(model.demand_id_list)):
                    if len(new_c2_f) < cro1_index:
                        if f1.node_id_list[index] not in new_c2_m:
                            new_c2_f.append(f1.node_id_list[index])
                    else:
                        if f1.node_id_list[index] not in new_c2_m:
                            new_c2_b.append(f1.node_id_list[index])
                new_c1 = copy.deepcopy(new_c1_f)
                new_c1.extend(new_c1_m)
                new_c1.extend(new_c1_b)
                f1.nodes_seq = new_c1
                new_c2 = copy.deepcopy(new_c2_f)
                new_c2.extend(new_c2_m)
                new_c2.extend(new_c2_b)
                f2.nodes_seq = new_c2
                model.sol_list.append(copy.deepcopy(f1))
                model.sol_list.append(copy.deepcopy(f2))
            else:
                model.sol_list.append(copy.deepcopy(f1))
                model.sol_list.append(copy.deepcopy(f2))
            if len(model.sol_list) >= model.popsize:
                break


def mutate_sol(model):
    sol_list = copy.deepcopy(model.sol_list)
    model.sol_list = []
    while True:
        f1_index = int(random.randint(0, len(sol_list) - 1))
        f1 = copy.deepcopy(sol_list[f1_index])
        m1_index = random.randint(0, len(model.demand_id_list)-1)
        m2_index = random.randint(0, len(model.demand_id_list)-1)
        if m1_index != m2_index:
            if random.random() <= model.pm:
                node1 = f1.node_id_list[m1_index]
                f1.node_id_list[m1_index] = f1.node_id_list[m2_index]
                f1.node_id_list[m2_index] = node1
                model.sol_list.append(copy.deepcopy(f1))
            else:
                model.sol_list.append(copy.deepcopy(f1))
            if len(model.sol_list) >= model.popsize:
                break


def plot_obj(obj_list):
    plt.rcParams["axes.unicode_minus"] = False
    plt.plot(np.arange(1, len(obj_list) + 1), obj_list)
    plt.xlabel("Iterations")
    plt.ylabel("obj value")
    plt.grid()
    plt.xlim(1, len(obj_list) + 1)
    plt.savefig("result遗传算法.png")
    plt.show()


def plot_routes(model):
    for route in model.best_sol.route_list:
        x_coord = [model.depot_dict[route[0]].x_coord]
        y_coord = [model.depot_dict[route[0]].y_coord]
        for node_id in route[1 : -(1 + len(model.school_id_list))]:
            x_coord.append(model.demand_dict[node_id].x_coord)
            y_coord.append(model.demand_dict[node_id].y_coord)
        for node_id in route[-(1 + len(model.school_id_list)) : -1]:
            x_coord.append(model.school_dict[node_id].x_coord)
            y_coord.append(model.school_dict[node_id].y_coord)
        x_coord.append(model.depot_dict[route[-1]].x_coord)
        y_coord.append(model.depot_dict[route[-1]].y_coord)
        plt.grid()
        if route[0] == "d1":
            plt.plot(
                x_coord[: -(len(model.school_id_list))],
                y_coord[: -(len(model.school_id_list))],
                marker="o",
                color="red",
                linewidth=0.5,
                markersize=5,
            )
            plt.plot(
                x_coord[-2:],
                y_coord[-2:],
                marker="o",
                color="red",
                linewidth=2,
                markersize=5,
            )
            plt.plot(
                x_coord[0],
                y_coord[0],
                marker="s",
                color="black",
                linewidth=0.5,
                markersize=10,
            )
        elif route[0] == "d2":
            plt.plot(
                x_coord[: -(len(model.school_id_list))],
                y_coord[: -(len(model.school_id_list))],
                marker="o",
                color="orange",
                linewidth=0.5,
                markersize=5,
            )
            plt.plot(
                x_coord[-2:],
                y_coord[-2:],
                marker="o",
                color="orange",
                linewidth=2,
                markersize=5,
            )
            plt.plot(
                x_coord[0],
                y_coord[0],
                marker="s",
                color="black",
                linewidth=0.5,
                markersize=10,
            )
        elif route[0] == "d3":
            plt.plot(
                x_coord[: -(len(model.school_id_list))],
                y_coord[: -(len(model.school_id_list))],
                marker="o",
                color="green",
                linewidth=0.5,
                markersize=5,
            )
            plt.plot(
                x_coord[-2:],
                y_coord[-2:],
                marker="o",
                color="green",
                linewidth=2,
                markersize=5,
            )
            plt.plot(
                x_coord[0],
                y_coord[0],
                marker="s",
                color="black",
                linewidth=0.5,
                markersize=10,
            )
        elif route[0] == "d4":
            plt.plot(
                x_coord[: -(len(model.school_id_list))],
                y_coord[: -(len(model.school_id_list))],
                marker="o",
                color="gray",
                linewidth=0.5,
                markersize=5,
            )
            plt.plot(
                x_coord[-2:],
                y_coord[-2:],
                marker="o",
                color="gray",
                linewidth=2,
                markersize=5,
            )
            plt.plot(
                x_coord[0],
                y_coord[0],
                marker="s",
                color="black",
                linewidth=0.5,
                markersize=10,
            )
        elif route[0] == "d5":
            plt.plot(
                x_coord[: -(len(model.school_id_list))],
                y_coord[: -(len(model.school_id_list))],
                marker="o",
                color="cyan",
                linewidth=0.5,
                markersize=5,
            )
            plt.plot(
                x_coord[-2:],
                y_coord[-2:],
                marker="o",
                color="cyan",
                linewidth=2,
                markersize=5,
            )
            plt.plot(
                x_coord[0],
                y_coord[0],
                marker="s",
                color="black",
                linewidth=0.5,
                markersize=10,
            )
        elif route[0] == "d6":
            plt.plot(
                x_coord[: -(len(model.school_id_list))],
                y_coord[: -(len(model.school_id_list))],
                marker="o",
                color="magenta",
                linewidth=0.5,
                markersize=5,
            )
            plt.plot(
                x_coord[-2:],
                y_coord[-2:],
                marker="o",
                color="magenta",
                linewidth=2,
                markersize=5,
            )
            plt.plot(
                x_coord[0],
                y_coord[0],
                marker="s",
                color="black",
                linewidth=0.5,
                markersize=10,
            )
        else:
            plt.plot(
                x_coord[: -(len(model.school_id_list))],
                y_coord[: -(len(model.school_id_list))],
                marker="o",
                color="blue",
                linewidth=0.5,
                markersize=5,
            )
            plt.plot(
                x_coord[-2:],
                y_coord[-2:],
                marker="o",
                color="blue",
                linewidth=2,
                markersize=5,
            )
            plt.plot(
                x_coord[0],
                y_coord[0],
                marker="s",
                color="black",
                linewidth=0.5,
                markersize=10,
            )
        plt.plot(
            x_coord[-(1 + len(model.school_id_list)) : -1],
            y_coord[-(1 + len(model.school_id_list)) : -1],
            marker="^",
            color="c",
            linewidth=2,
            markersize=10,
        )
    plt.xlabel("x_coord")
    plt.ylabel("y_coord")
    plt.savefig("routes遗传算法.png")
    plt.show()


def output(model, epochs):
    depot_count = []
    for route in model.best_sol.route_list:
        depot_count.append(route[0])
    c = collections.Counter(depot_count)

    work = xlsxwriter.Workbook(os.getcwd()+'/遗传算法.xlsx')
    worksheet = work.add_worksheet()
    worksheet.write(0, 0, "时间cost")
    worksheet.write(0, 1, "距离cost")
    worksheet.write(0, 2, "优化目标(0:最短距离；1:最短时间)")
    worksheet.write(0, 3, "目标值")
    worksheet.write(0, 4, "迭代次数")
    worksheet.write(0, 5, "校车使用数量")
    worksheet.write(0, 6, "cpu时间")
    for i in range(len(model.depot_id_list)):
        worksheet.write(0, 7 + i, f"车场{model.depot_id_list[i]}的车辆数")
    worksheet.write(1, 0, model.best_sol.cost_of_time)
    worksheet.write(1, 1, model.best_sol.cost_of_distance)
    worksheet.write(1, 2, model.opt_type)
    worksheet.write(1, 3, model.best_sol.obj)
    worksheet.write(1, 4, epochs)
    worksheet.write(1, 5, model.best_sol.num_vehicles)
    worksheet.write(1, 6, model.time)
    for i in range(len(model.depot_id_list)):
        worksheet.write(1, 7 + i, c[model.depot_id_list[i]])
    worksheet.write(2, 0, "vehicleID")
    worksheet.write(2, 1, "route")
    worksheet.write(2, 2, "timetable")
    for row, route in enumerate(model.best_sol.vehicle_schedule):
        worksheet.write(row + 3, 0, route[0])
        r = [str(i) for i in route[1]]
        worksheet.write(row + 3, 1, "-".join(r))
        r = [str(i) for i in model.best_sol.timetable[row]]
        worksheet.write(row + 3, 2, "-".join(r))
    work.close()
    with open('/Users/zhaozhijie/PycharmProjects/VRP/heterogenerous/data_遗传算法.txt', 'a', encoding='utf-8') as f:
        txt = []
        txt.append(str(model.best_sol.num_vehicles) + ',')
        txt.append(str(model.best_sol.cost_of_distance) + ',')
        txt.append(str(model.best_sol.cost_of_time) + ',')
        txt.append(str(model.time) + ',')
        f.write('\n')
        f.writelines(txt)


def run(demand_file, depot_file, school_file, vehicle_file, epochs, pc, pm, popsize, n_select,v_speed, opt_type, vehicle_assignment_mode, distance_opt):
    begin = time.time()
    model = Model()
    model.opt_type = opt_type
    model.vehicle_speed = v_speed
    model.vehicle_assignment_mode = vehicle_assignment_mode
    model.pc = pc
    model.pm = pm
    model.popsize = popsize
    model.n_select = n_select
    model.distance_opt = distance_opt
    sol = Sol()
    sol.obj = float("inf")
    model.best_sol = sol
    history_best_obj = []
    read_csv_file(demand_file, depot_file, school_file, vehicle_file, model)
    cal_school_sequence(model)
    cal_time_distance_matrix(model)
    generate_initial_sol(model)
    for ep in range(epochs):
        indicator = cal_fitness(model)
        if indicator == 0:
            print('没有足够的车辆可够分配。尽管最终优化结果可能不需要更多的车辆，但再算法优化过程中可能需要更多的车辆来计算'
                  '请尝试增加车辆数量或选择车辆分配模式1')
            sys.exit()
        select_sol(model)
        cross_sol(model)
        mutate_sol(model)
        history_best_obj.append(model.best_sol.obj)
        print(f'{ep}/{epochs}    距离cost:{model.best_sol.cost_of_distance}   车辆数量:{model.best_sol.num_vehicles}   时间cost:{model.best_sol.cost_of_time}   目标值:{model.best_sol.obj}')
    print(model.best_sol.vehicle_schedule)
    list01 = []
    for i in range(len(model.best_sol.vehicle_schedule)):
        list01.append(model.best_sol.vehicle_schedule[i][0])
    set01 = set(list01)
    dict01 = {}
    for item in set01:
        dict01.update({item: list01.count(item)})
    print(f'各车型使用次数{dict01}')
    end = time.time()
    model.time = end - begin
    plot_obj(history_best_obj)
    plot_routes(model)
    output(model, epochs)


if __name__ == "__main__":
    demand_file = "/Users/zhaozhijie/PycharmProjects/VRP/heterogenerous/时间修正/3学校3车库30需求点/demand.csv"
    depot_file = "/Users/zhaozhijie/PycharmProjects/VRP/heterogenerous/时间修正/3学校3车库30需求点/depot.csv"
    school_file = "/Users/zhaozhijie/PycharmProjects/VRP/heterogenerous/时间修正/3学校3车库30需求点/school.csv"
    vehicle_file = "/Users/zhaozhijie/PycharmProjects/VRP/heterogenerous/时间修正/3学校3车库30需求点/vehicle.csv"
    run(
        demand_file,
        depot_file,
        school_file,
        vehicle_file,
        epochs=200,
        v_speed=660,
        pc=0.8,
        pm=0.1,
        popsize=100,
        n_select=80,
        opt_type=0,
        vehicle_assignment_mode=1,
        distance_opt=1
    )
