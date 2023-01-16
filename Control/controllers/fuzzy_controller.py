import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl


def fuzzy_initialization():
    # 定义模糊范围
    dspeed_range = np.arange(-30, 30, 1, np.float32)  # 速度差
    distance_range = np.arange(0, 200, 2, np.float32)  # 距离差
    acc_range = np.arange(0.2, 0.8, 0.01, np.float32)  # 输出加速度差
    # 创建模糊控制变量
    dspeed = ctrl.Antecedent(dspeed_range, 'dspeed')
    distance = ctrl.Antecedent(distance_range, 'distance')
    acc = ctrl.Consequent(acc_range, 'acc')
    # 定义模糊集和其隶属度函数
    dspeed['NL'] = fuzz.trimf(dspeed_range, [-30, -20, -10])
    dspeed['NB'] = fuzz.trimf(dspeed_range, [-20, -15, -5])
    dspeed['NS'] = fuzz.trimf(dspeed_range, [-10, -5, 0])
    dspeed['0'] = fuzz.trimf(dspeed_range, [-1, 0, 1])
    dspeed['PS'] = fuzz.trimf(dspeed_range, [0, 5, 10])
    dspeed['PB'] = fuzz.trimf(dspeed_range, [5, 10, 20])
    dspeed['PL'] = fuzz.trimf(dspeed_range, [10, 20, 30])

    distance['NL'] = fuzz.trimf(distance_range, [50, 60, 70])
    distance['NB'] = fuzz.trimf(distance_range, [65, 70, 75])
    distance['NS'] = fuzz.trimf(distance_range, [70, 75, 80])
    distance['0'] = fuzz.trimf(distance_range, [79, 80, 81])
    distance['PS'] = fuzz.trimf(distance_range, [80, 90, 100])
    distance['PB'] = fuzz.trimf(distance_range, [90, 120, 140])
    distance['PL'] = fuzz.trimf(distance_range, [130, 150, 200])

    acc['NL'] = fuzz.trimf(acc_range, [0.7, 0.75, 0.8])
    acc['NB'] = fuzz.trimf(acc_range, [0.6, 0.7, 0.75])
    acc['NS'] = fuzz.trimf(acc_range, [0.5, 0.55, 0.65])
    acc['0'] = fuzz.trimf(acc_range, [0.48, 0.5, 0.52])
    acc['PS'] = fuzz.trimf(acc_range, [0.35, 0.45, 0.5])
    acc['PB'] = fuzz.trimf(acc_range, [0.25, 0.3, 0.4])
    acc['PL'] = fuzz.trimf(acc_range, [0.2, 0.25, 0.3])

    acc.defuzzify_method = 'centroid'  # 设定输出powder的解模糊方法——质心解模糊方式
    # 输出规则
    ruleNL = ctrl.Rule(antecedent=((dspeed['PB'] & distance['NL']) |
                                        (dspeed['PL'] & distance['NL']) |
                                        (dspeed['PL'] & distance['NB'])),
                            consequent=acc['NL'], label='rule NL')
    ruleNB = ctrl.Rule(antecedent=((dspeed['0'] & distance['NL']) |
                                        (dspeed['PS'] & distance['NL']) |
                                        (dspeed['PS'] & distance['NB']) |
                                        (dspeed['PB'] & distance['NB']) |
                                        (dspeed['PB'] & distance['NS']) |
                                        (dspeed['PL'] & distance['NS']) |
                                        (dspeed['PL'] & distance['0'])),
                            consequent=acc['NB'], label='rule NB')
    ruleNS = ctrl.Rule(antecedent=((dspeed['NB'] & distance['NL']) |
                                        (dspeed['NS'] & distance['NL']) |
                                        (dspeed['0'] & distance['NB']) |
                                        (dspeed['PS'] & distance['NS']) |
                                        (dspeed['PB'] & distance['0']) |
                                        (dspeed['PB'] & distance['PS']) |
                                        (dspeed['PL'] & distance['PS']) |
                                        (dspeed['PL'] & distance['PB'])),
                            consequent=acc['NS'], label='rule NS')
    rule0 = ctrl.Rule(antecedent=((dspeed['NL'] & distance['NL']) |
                                       (dspeed['NB'] & distance['NB']) |
                                       (dspeed['NS'] & distance['NB']) |
                                       (dspeed['NS'] & distance['NS']) |
                                       (dspeed['0'] & distance['NS']) |
                                       (dspeed['0'] & distance['0']) |
                                       (dspeed['PS'] & distance['0']) |
                                       (dspeed['PS'] & distance['PS']) |
                                       (dspeed['PB'] & distance['PB']) |
                                       (dspeed['PL'] & distance['PL'])),
                           consequent=acc['0'], label='rule 0')
    rulePS = ctrl.Rule(antecedent=((dspeed['NL'] & distance['NB']) |
                                        (dspeed['NL'] & distance['NS']) |
                                        (dspeed['NB'] & distance['NS']) |
                                        (dspeed['NB'] & distance['0']) |
                                        (dspeed['NS'] & distance['0']) |
                                        (dspeed['NS'] & distance['PS']) |
                                        (dspeed['0'] & distance['PS']) |
                                        (dspeed['0'] & distance['PB']) |
                                        (dspeed['PS'] & distance['PB']) |
                                        (dspeed['PS'] & distance['PL']) |
                                        (dspeed['PB'] & distance['PL'])),
                            consequent=acc['PS'], label='rule PS')
    rulePB = ctrl.Rule(antecedent=((dspeed['NL'] & distance['0']) |
                                        (dspeed['NB'] & distance['PS']) |
                                        (dspeed['NB'] & distance['PB']) |
                                        (dspeed['NS'] & distance['PB']) |
                                        (dspeed['NS'] & distance['PL']) |
                                        (dspeed['0'] & distance['PL'])),
                            consequent=acc['PB'], label='rule PB')
    rulePL = ctrl.Rule(antecedent=((dspeed['NL'] & distance['NS']) |
                                   (dspeed['NL'] & distance['PB']) |
                                   (dspeed['NL'] & distance['PL']) |
                                   (dspeed['NB'] & distance['PL'])),
                            consequent=acc['PL'], label='rule PL')
    # 系统和运行环境初始化
    system = ctrl.ControlSystem(
        rules=[ruleNL, ruleNB, ruleNS, rule0, rulePS, rulePB, ruleNL])
    fuzzy_system = ctrl.ControlSystemSimulation(system)
    return fuzzy_system


def fuzzy_compute(fuzzy_system, dspeed, distance):
        fuzzy_system.input['dspeed'] = dspeed
        fuzzy_system.input['distance'] = distance
        fuzzy_system.compute()
        acc = fuzzy_system.output['acc']
        return acc
