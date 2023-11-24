"""
作者：zhaozhijie
日期：2022年03月12日10时37分
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
        self.route_list = []  # SBRP的解
        self.timetable = []  # 时间表
        self.vehicles_schedule = []  # 存储车辆调度信息
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
        # 0:一辆车在整个服务时间内最多只服务一次 1:先假设所有类型(不同容量)的车的数量是无限的，待算法优化完后再进行车辆分配优化。
        self.vehicle_assignment_mode = 0
        self.vehicle_id_list = []
        self.vehicles_capacity = []
        self.vehicle_dict = {}
        self.vehicle_num = 0
        self.vehicle_speed = 1
        self.alpha = 0      # GRASP 参数
        self.time = None

    @property
    def distanceOption(self):
        return self.distance_opt

    @distanceOption.setter
    def setDistanceOption(self, option):
        self.distance_opt = option


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
        # 节点-节点
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
        # 节点-车库
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


def Greedy_Randomized_Construction(model) -> object:
    sol = Sol()
    remained_nodes_dict = copy.deepcopy(model.demand_dict)
    current_node_index = random.randint(
        model.demand_id_list[0], model.demand_id_list[-1]
    )
    current_node = remained_nodes_dict[current_node_index]
    sol.node_id_list.append(current_node_index)
    del remained_nodes_dict[current_node_index]
    while remained_nodes_dict:
        RCL = Make_RCL(model, remained_nodes_dict, current_node)
        picked_node = random.choice(RCL)
        sol.node_id_list.append(picked_node.id)
        current_node = picked_node
        del remained_nodes_dict[current_node.id]
    return sol


def Make_RCL(model, remained_nodes_dict, current_node):
    """基于价值的标准,静态alpha策略"""
    cost = []
    RCL = []
    node_id_list = []
    for node_id, node in remained_nodes_dict.items():
        c = math.sqrt(
            (current_node.x_coord - node.x_coord) ** 2
            + (current_node.y_coord - node.y_coord) ** 2
        )
        cost.append(c)
        node_id_list.append(node_id)
    c_min = min(cost)
    c_max = max(cost)
    criteria = c_min + model.alpha * (c_max - c_min)        # !!!!!!!!!!!!
    for i in range(len(cost)):
        if cost[i] <= criteria:
            RCL.append(remained_nodes_dict[node_id_list[i]])
    return RCL


def cal_obj(sol, model):
    node_id_list = copy.deepcopy(sol.node_id_list)
    sol.num_vehicles, sol.route_list, sol.vehicles_schedule = split_routes(
        node_id_list, model
    )
    if not sol.num_vehicles:
        return 0
    sol.timetable, sol.cost_of_time, sol.cost_of_distance = cal_travel_cost(
        sol.route_list, model
    )
    if model.opt_type == 0:
        sol.obj = sol.cost_of_distance + sol.num_vehicles * 100000
    else:
        sol.obj = sol.cost_of_time
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
                            (departure - model.time_matrix[n_2, school])
                            - model.demand_dict[n_2].service_time
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
                    if cumulative_min_cost[4] + cost <= cumulative_min_cost[n_2]:
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
            label = labels[node_id]                               # 被字典坑惨了 用列表+元组代替字典
    routes.append(route_pure)
    for i in range(len(routes)):
        demand = 0
        for j in range(len(routes[i])):
            demand += model.demand_dict[routes[i][j]].demand
        demand_dict.append((demand, routes[i]))
    for t in demand_dict:
        if t[0] > 80:
            print('1')
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
    # print(len(vehicle_id_list))               # 车辆数量debug
    # print(len(demand_dict))
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
                print('没有足够的车辆可够分配。尽管最终优化结果可能不需要更多的车辆，但再算法优化过程中可能需要更多的车辆来计算'
                      '请尝试增加车辆数量或选择车辆分配模式1')
                return 0
            vehicle_id_list.pop(index)
            capacity_list.pop(index)

        elif model.vehicle_assignment_mode == 1:
            index = list(map(lambda x: x >= demand_dict[i][0], capacity_list)).index(True)
            vehicle_schedule.append(vehicle_id_list[index])
    #     print(vehicle_id_list)
    #     print(capacity_list)
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
        print("没有车辆可分配,增加车场容量")
        sys.exit(0)

    route.insert(0, index)
    route.extend(model.school_id_sequence)
    route.append(index)
    depot_dict[index].depot_capacity -= 1
    # print(depot_dict[index].depot_capacity)
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


def local_search(k, sol, model):
    T = 100
    current_sol = copy.deepcopy(sol)
    new_sol = copy.deepcopy(sol)
    if k == 4:  # remove-insert within a route
        while T > 0.01:
            perturbation_operator1(current_sol, model)
            delta_obj = current_sol.obj - new_sol.obj
            if delta_obj < 0 or random.random() < math.exp(-delta_obj / T):
                new_sol = copy.deepcopy(current_sol)
            T *= 0.98
        return new_sol
    elif k == 3:  # remove-insert between two routes
        while T > 0.01:
            perturbation_operator2(current_sol, model)
            delta_obj = current_sol.obj - new_sol.obj
            if delta_obj < 0 or random.random() < math.exp(-delta_obj / T):
                new_sol = copy.deepcopy(current_sol)
            T *= 0.98
        return new_sol
    elif k == 2:
        new_sol = perturbation_operator3(current_sol, model)
        return new_sol
    elif k == 1:  # 魔改的LKH算法
        new_sol = perturbation_operator4(current_sol, model)
        return new_sol


def perturbation_operator1(sol, model):
    while True:
        index_1 = random.randint(0, len(sol.node_id_list) - 1)
        index_2 = random.randint(0, len(sol.node_id_list) - 1)
        if index_1 != index_2:
            break
    removed_node_id = sol.node_id_list.pop(index_1)
    sol.node_id_list.insert(index_2, removed_node_id)
    cal_obj(sol, model)


def perturbation_operator2(sol, model):
    while True:
        index_1 = random.randint(0, len(sol.node_id_list) - 1)
        index_2 = random.randint(0, len(sol.node_id_list) - 1)
        if index_1 != index_2:
            break
    sol.node_id_list[index_1: index_2 + 1] = list(
        reversed(sol.node_id_list[index_1: index_2 + 1])
    )
    cal_obj(sol, model)


def perturbation_operator3(current_sol, model):
    node_id_list = copy.deepcopy(current_sol.node_id_list)
    distance = []
    two_side_distance = []
    two_side_distance1 = []
    for i in range(len(node_id_list) - 1):
        distance.append(model.distance_matrix[node_id_list[i], node_id_list[i + 1]])
    for i in range(len(distance) - 1):
        two_side_distance.append(distance[i] + distance[i + 1])
    index = two_side_distance.index(max(two_side_distance))
    node_index = index + 1
    node_id = node_id_list.pop(node_index)
    for i in range(len(node_id_list) - 1):
        two_side_distance1.append(
            model.distance_matrix[node_id_list[i], node_id]
            + model.distance_matrix[node_id_list[i + 1], node_id]
        )
    index_ = two_side_distance1.index(min(two_side_distance1))
    node_index_ = index_ + 1
    node_id_list.insert(node_index_, node_id)
    sol = Sol()
    sol.node_id_list = node_id_list
    cal_obj(sol, model)
    if sol.obj < current_sol.obj:
        current_sol = copy.deepcopy(sol)
    return current_sol


def perturbation_operator4(current_sol, model):
    sol = copy.deepcopy(current_sol)
    sol1 = copy.deepcopy(current_sol)
    distance_list = []
    distance_list2 = []
    sol2_list = []
    obj_list = []
    for i in range(len(current_sol.node_id_list) - 1):
        distance_list.append(
            model.distance_matrix[
                current_sol.node_id_list[i], current_sol.node_id_list[i + 1]
            ]
        )
    for i in range(1, len(current_sol.node_id_list) - 1):  # 从最后两个向前推
        current_sol = copy.deepcopy(sol)
        index1, index2 = (
            distance_list.index(sorted(distance_list)[-i]),
            distance_list.index(sorted(distance_list)[-(i + 1)]),
        )  # 距离最大的两个节点的index
        if index1 > index2:  # 2-opt
            index1, index2 = index2, index1
        reversed_seq = current_sol.node_id_list[index1 + 1: index2 + 1]
        reversed_seq.reverse()
        current_sol.node_id_list = (
            current_sol.node_id_list[: index1 + 1]
            + reversed_seq
            + current_sol.node_id_list[index2 + 1:]
        )
        cal_obj(current_sol, model)
        if current_sol.obj < sol1.obj:
            sol1 = copy.deepcopy(current_sol)
            sol2 = copy.deepcopy(sol1)
            for i in range(len(current_sol.node_id_list) - 1):
                distance_list2.append(
                    model.distance_matrix[
                        current_sol.node_id_list[i], current_sol.node_id_list[i + 1]
                    ]
                )
            for j in range(1, len(current_sol.node_id_list) - 2):
                current_sol = copy.deepcopy(sol1)
                index1, index2, index3 = (
                    distance_list2.index(sorted(distance_list2)[-j]),
                    distance_list2.index(sorted(distance_list2)[-(j + 1)]),
                    distance_list2.index(sorted(distance_list2)[-(j + 2)]),
                )
                if index1 > index2:  # 3-opt
                    index1, index2 = index2, index1
                if index2 > index3:
                    index2, index3 = index3, index2
                if index1 > index2:
                    index1, index2 = index2, index1
                reversed_seq1 = current_sol.node_id_list[index1 + 1: index2 + 1]
                reversed_seq1.reverse()
                reversed_seq2 = current_sol.node_id_list[index2 + 1: index3 + 1]
                reversed_seq2.reverse()
                current_sol.node_id_list = (
                    current_sol.node_id_list[: index1 + 1]
                    + reversed_seq1
                    + reversed_seq2
                    + current_sol.node_id_list[index3 + 1:]
                )
                cal_obj(current_sol, model)
                if current_sol.obj < sol1.obj:
                    if current_sol.obj < sol2.obj:
                        sol2 = copy.deepcopy(current_sol)
            sol2_list.append(sol2)
            obj_list.append(sol2.obj)
    if sol2_list:
        return sol2_list[obj_list.index(min(obj_list))]
    else:
        return sol1


def plot_obj(obj_list):
    plt.rcParams["axes.unicode_minus"] = False
    plt.plot(np.arange(1, len(obj_list) + 1), obj_list)
    plt.xlabel("Iterations")
    plt.ylabel("obj value")
    plt.grid()
    plt.xlim(1, len(obj_list) + 1)
    plt.savefig("resultGRASP.png")
    # plt.clf()
    plt.show()


def plot_routes(model):
    for route in model.best_sol.route_list:
        x_coord = [model.depot_dict[route[0]].x_coord]
        y_coord = [model.depot_dict[route[0]].y_coord]
        for node_id in route[1: -(1 + len(model.school_id_list))]:
            x_coord.append(model.demand_dict[node_id].x_coord)
            y_coord.append(model.demand_dict[node_id].y_coord)
        for node_id in route[-(1 + len(model.school_id_list)): -1]:
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
            x_coord[-(1 + len(model.school_id_list)): -1],
            y_coord[-(1 + len(model.school_id_list)): -1],
            marker="^",
            color="c",
            linewidth=2,
            markersize=10,
        )
    plt.xlabel("x_coord")
    plt.ylabel("y_coord")
    plt.savefig("routesGRASP.png")
    plt.show()


def output(model, epochs, history_best_list, k_list, GRASP):
    depot_count = []
    for route in model.best_sol.route_list:
        depot_count.append(route[0])
    c = collections.Counter(depot_count)

    work = xlsxwriter.Workbook(os.getcwd()+'/GRASP.xlsx')
    worksheet = work.add_worksheet()
    worksheet.write(0, 0, "时间cost")
    worksheet.write(0, 1, "距离cost")
    worksheet.write(0, 2, "优化目标(0:最短距离；1:最短时间)")
    worksheet.write(0, 3, "目标值")
    worksheet.write(0, 4, "迭代次数")
    worksheet.write(0, 5, "alpha参数")
    worksheet.write(0, 6, '校车使用数量')
    worksheet.write(0, 7, 'cpu时间')
    for i in range(len(model.depot_id_list)):
        worksheet.write(0, 8 + i, f"车场{model.depot_id_list[i]}的车辆数")
    worksheet.write(1, 0, model.best_sol.cost_of_time)
    worksheet.write(1, 1, model.best_sol.cost_of_distance)
    worksheet.write(1, 2, model.opt_type)
    worksheet.write(1, 3, model.best_sol.obj)
    worksheet.write(1, 4, epochs)
    worksheet.write(1, 5, model.alpha)
    worksheet.write(1, 6, model.best_sol.num_vehicles)
    worksheet.write(1, 7, model.time)
    for i in range(len(model.depot_id_list)):
        worksheet.write(1, 8 + i, c[model.depot_id_list[i]])
    worksheet.write(2, 0, "vehicleID")
    worksheet.write(2, 1, "route")
    worksheet.write(2, 2, "timetable")
    for row, route in enumerate(model.best_sol.vehicles_schedule):
        worksheet.write(row + 3, 0, route[0])
        r = [str(i) for i in route[1]]
        worksheet.write(row + 3, 1, ">".join(r))
        r = [str(i) for i in model.best_sol.timetable[row]]
        worksheet.write(row + 3, 2, ">".join(r))
    work.close()
    with open('/Users/zhaozhijie/PycharmProjects/VRP/heterogenerous/data_GRASP.txt', 'a', encoding='utf-8') as f:
        txt = []
        # for index in range(len(history_best_list)):
        #     txt.append(str(history_best_list[index])+',')
        txt.append(str(model.best_sol.num_vehicles)+',')
        txt.append(str(model.best_sol.cost_of_distance)+',')
        txt.append(str(model.best_sol.cost_of_time)+',')
        txt.append(str(model.time)+',')
        txt.append(str(GRASP)+',')
        txt.append(str(model.best_sol.obj)+',')
        f.write('\n')
        f.writelines(txt)
    with open('/Users/zhaozhijie/PycharmProjects/VRP/heterogenerous/operator_count.txt', 'a', encoding='utf-8') as f:
        txt = []
        for index in range(len(k_list)):
            txt.append(str(k_list[index])+',')
        f.write('\n')
        f.writelines(txt)


def run(
    demand_file,
    depot_file,
    school_file,
    vehicle_file,
    epochs,
    alpha,
    v_speed,
    k_max,
    opt_type,
    vehicle_assignment_mode,
    distance_opt
):
    begin = time.time()
    model = Model()
    model.opt_type = opt_type
    model.vehicle_speed = v_speed
    model.alpha = alpha
    model.vehicle_assignment_mode = vehicle_assignment_mode
    model.distance_opt = distance_opt
    sol = Sol()
    sol.obj = float("inf")
    model.best_sol = sol
    history_best_obj = []
    read_csv_file(demand_file, depot_file, school_file, vehicle_file, model)
    print('验证学校序列是否可行中>>>')
    cal_school_sequence(model)
    print('学校序列验证通过,计算时间距离矩阵中>>>')
    cal_time_distance_matrix(model)
    GRASP_list = []  # 存储GRASP生成的解
    k_list = np.zeros(k_max)  # 存储邻域动作使用次数
    print('开始求解')
    for ep in range(epochs):
        sol = Greedy_Randomized_Construction(model)
        indicator = cal_obj(sol, model)
        if not indicator:       # 车辆分配模式0中如果遇到有的解需求的车辆超出了最大车辆则跳出本次迭代进入下次迭代
            continue
        GRASP_list.append(sol.obj)
        k = 1
        # local_search_sol = []
        while k <= k_max:
            new_sol = local_search(k, sol, model)
            if new_sol.obj < sol.obj:
                k_list[k - 1] += 1
                sol = copy.deepcopy(new_sol)
                k = 1
            else:
                k += 1
            # local_search_sol.append(sol.cost_of_distance)
        # plot_obj(local_search_sol)
        model.sol_list.append(sol)
        if sol.obj < model.best_sol.obj:
            model.best_sol = copy.deepcopy(sol)
        history_best_obj.append(model.best_sol.obj)
        print(f"{ep}/{epochs}   距离cost:{model.best_sol.cost_of_distance}  车辆数量:{model.best_sol.num_vehicles}    时间cost:{model.best_sol.cost_of_time}    目标值:{model.best_sol.obj}")
    print(f"贪婪随机自适应构造的所有解中最优的是{min(GRASP_list)}")
    print(f"几种算子对当前解的改造情况{k_list}")
    # 车辆使用情况
    list01 = []
    for i in range(len(model.best_sol.vehicles_schedule)):
        list01.append(model.best_sol.vehicles_schedule[i][0])
    set01 = set(list01)
    dict01 = {}
    for item in set01:
        dict01.update({item: list01.count(item)})
    print(dict01)

    print(model.best_sol.vehicles_schedule)
    end = time.time()
    model.time = end - begin
    print(f"求解时间为:{model.time}")
    plot_obj(history_best_obj)
    plot_routes(model)
    output(model, epochs, history_best_obj, k_list, min(GRASP_list))


if __name__ == "__main__":
    demand_file = "/Users/zhaozhijie/PycharmProjects/VRP/heterogenerous/现实问题/demand.csv"
    depot_file = "/Users/zhaozhijie/PycharmProjects/VRP/heterogenerous/现实问题/depot.csv"
    school_file = "/Users/zhaozhijie/PycharmProjects/VRP/heterogenerous/现实问题/school.csv"
    vehicle_file = "/Users/zhaozhijie/PycharmProjects/VRP/heterogenerous/现实问题/vehicle.csv"
    run(
        demand_file,
        depot_file,
        school_file,
        vehicle_file,
        epochs=10,
        alpha=0.2,
        v_speed=660,  # 米/分钟
        k_max=4,
        opt_type=0,
        vehicle_assignment_mode=1,      # 0不允许优化过程中车辆超上限 1求解完再优化
        distance_opt=1      # 0欧氏距离 1曼哈顿距离
    )
