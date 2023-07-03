import casadi as ca
import numpy as np
import time


def f(states, controls):
    return rhs


def shift_movement(T, t0, x0, u, f):
    # 小车运动到下一个位置
    f_value = f(x0, u[:, 0])
    st = x0 + T*f_value
    # 时间增加
    t = t0 + T
    # 准备下一个估计的最优控制，因为u[:, 0]已经采纳，我们就简单地把后面的结果提前
    u_end = ca.horzcat(u[:, 1:], u[:, -1])
    return t, st, u_end.T


if __name__ == '__main__':
    T = 0.2  # （模拟的）系统采样时间【秒】
    N = 100  # 需要预测的步长【超参数】
    rob_diam = 0.3  # 机器人自身直径【米，仅为展示所用】
    v_max = 0.6  # 最大前向速度【物理约束】
    omega_max = np.pi / 4.0  # 最大转动角速度 【物理约束】
    # 根据数学模型建模
    ## 系统状态
    x = ca.SX.sym('x')  # x轴状态
    y = ca.SX.sym('y')  # y轴状态
    theta = ca.SX.sym('theta')  # z轴转角　
    states = ca.vertcat(x, y)  # 构建小车状态向量 \bm{x} = [x, y, theta].T
    states = ca.vertcat(states, theta)  # 实际上也可以通过 states = ca.vertcat(*[x, y, theta])一步实现
    n_states = states.size()[0]  # 获得系统状态的尺寸，向量以（n_states, 1）的格式呈现 【这点很重要】
    ## 控制输入
    v = ca.SX.sym('v')  # 前向速度
    omega = ca.SX.sym('omega')  # 转动速度
    controls = ca.vertcat(v, omega)  # 控制向量
    n_controls = controls.size()[0]  # 控制向量尺寸
    # 运动学模型
    rhs = ca.vertcat(v * np.cos(theta), v * np.sin(theta))
    rhs = ca.vertcat(rhs, omega)
    # 利用CasADi构建一个函数
    f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

    # 开始构建MPC
    ## 相关变量，格式(状态长度， 步长)
    U = ca.SX.sym('U', n_controls, N)  # N步内的控制输出
    X = ca.SX.sym('X', n_states, N + 1)  # N+1步的系统状态，通常长度比控制多1
    P = ca.SX.sym('P', n_states + n_states)  # 构建问题的相关参数
    # 在这里每次只需要给定当前/初始位置和目标终点位置
    ## Single Shooting 约束条件
    X[:, 0] = P[:3]  # 初始状态希望相等
    ### 剩余N状态约束条件
    for i in range(N):
        # 通过前述函数获得下个时刻系统状态变化。
        # 这里需要注意引用的index为[:, i]，因为X为(n_states, N+1)矩阵
        f_value = f(X[:, i], U[:, i])
        X[:, i + 1] = X[:, i] + f_value * T
    ## 获得输入（控制输入，参数）和输出（系统状态）之间关系的函数ff
    ff = ca.Function('ff', [U, P], [X], ['input_U', 'target_state'], ['horizon_states'])
    ## NLP问题
    ### 惩罚矩阵
    Q = np.array([[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, .1]])
    R = np.array([[0.5, 0.0], [0.0, 0.05]])
    ### 优化目标
    obj = 0  # 初始化优化目标值
    for i in range(N):
        # 在N步内对获得优化目标表达式
        obj = obj + ca.mtimes([(X[:, i] - P[3:]).T, Q, X[:, i] - P[3:]]) + ca.mtimes([U[:, i].T, R, U[:, i]])

    ### 约束条件定义
    g = []  # 用list来存储优化目标的向量
    for i in range(N + 1):
        # 这里的约束条件只有小车的坐标（x,y）必须在-2至2之间
        # 由于xy没有特异性，所以在这个例子中顺序不重要（但是在更多实例中，这个很重要）
        g.append(X[0, i])
        g.append(X[1, i])
    ### 定义NLP问题，'f'为目标函数，'x'为需寻找的优化结果（优化目标变量），'p'为系统参数，'g'为约束条件
    ### 需要注意的是，用SX表达必须将所有表示成标量或者是一维矢量的形式
    nlp_prob = {'f': obj, 'x': ca.reshape(U, -1, 1), 'p': P, 'g': ca.vertcat(*g)}
    ### ipot设置
    opts_setting = {'ipopt.max_iter': 30, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                    'ipopt.acceptable_obj_change_tol': 1e-6}
    ## 最终目标，获得求解器
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

    # 开始仿真
    ## 定义约束条件，实际上CasADi需要在每次求解前更改约束条件。不过我们这里这些条件都是一成不变的
    ## 因此我们定义在while循环外，以提高效率
    ### 状态约束
    lbg = -2.0  # x，y不得小于-2
    ubg = 2.0  # x，y不得大于2
    ### 控制约束
    lbx = []  # 最低约束条件
    ubx = []  # 最高约束条件
    for _ in range(N):
        #### 记住这个顺序，不可搞混！
        #### U是以(n_controls, N)存储的，但是在定义问题的时候被改变成(n_controlsxN,1)的向量
        #### 实际上，第一组控制v0和omega0的index为U_0为U_1，第二组为U_2和U_3
        #### 因此，在这里约束必须在一个循环里连续定义。
        lbx.append(-v_max)
        ubx.append(v_max)
        lbx.append(-omega_max)
        ubx.append(omega_max)
    ## 仿真条件和相关变量
    t0 = 0.0  # 仿真时间
    x0 = np.array([0.0, 0.0, 0.0]).reshape(-1, 1)  # 小车初始状态
    xs = np.array([1.5, 1.5, 0.0]).reshape(-1, 1)  # 小车末了状态
    u0 = np.array([0.0, 0.0] * N).reshape(-1, 2)  # 系统初始控制状态，为了统一本例中所有numpy有关
    # 变量都会定义成（N,状态）的形式方便索引和print
    x_c = []  # 存储系统的状态
    u_c = []  # 存储控制全部计算后的控制指令
    t_c = []  # 保存时间
    xx = []  # 存储每一步机器人位置
    sim_time = 20.0  # 仿真时长
    index_t = []  # 存储时间戳，以便计算每一步求解的时间

    ## 开始仿真
    mpciter = 0  # 迭代计数器
    start_time = time.time()  # 获取开始仿真时间
    ### 终止条件为小车和目标的欧式距离小于0.01或者仿真超时
    while (np.linalg.norm(x0 - xs) > 1e-2 and mpciter - sim_time / T < 0.0):
        ### 初始化优化参数
        c_p = np.concatenate((x0, xs))
        ### 初始化优化目标变量
        init_control = ca.reshape(u0, -1, 1)
        ### 计算结果并且
        t_ = time.time()
        res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
        index_t.append(time.time() - t_)
        ### 获得最优控制结果u
        u_sol = ca.reshape(res['x'], n_controls, N)  # 记住将其恢复U的形状定义
        ###
        ff_value = ff(u_sol, c_p)  # 利用之前定义ff函数获得根据优化后的结果
        # 小车之后N+1步后的状态（n_states, N+1）
        ### 存储结果
        x_c.append(ff_value)
        u_c.append(u_sol[:, 0])
        t_c.append(t0)
        ### 根据数学模型和MPC计算的结果移动小车并且准备好下一个循环的初始化目标
        t0, x0, u0 = shift_movement(T, t0, x0, u_sol, f)
        ### 存储小车的位置
        x0 = ca.reshape(x0, -1, 1)
        xx.append(x0.full())
        ### 计数器+1
        mpciter = mpciter + 1

        print(index_t)
