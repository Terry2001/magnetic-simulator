# -*- coding: utf-8 -*-
import math
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st
from io import BytesIO

# 设置 Matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


u = constants.mu_0

# Streamlit 界面设置
st.set_page_config(layout="wide")

# Streamlit 界面
st.title("磁感线可视化")

st.sidebar.header("参数设置")

# 采样点数量设置
num_points = st.sidebar.number_input("采样点数量", min_value=1000, max_value=100000, value=15000, step=1000, help="增加采样点可提高图像密度，但会增加计算时间")

# 高度设置（Z轴截面）
z_height = st.sidebar.slider("观察高度 Z/m", min_value=-0.1, max_value=0.1, value=0.0, step=0.005, format="%.3f")

# 交互式拖动模式开关（默认启用）
enable_drag = st.sidebar.checkbox("启用拖动模式", value=True, help="启用后可在下方使用滑块拖动磁铁位置")

# 磁铁数量
num_magnets = st.sidebar.number_input("磁铁数量", min_value=1, max_value=10, value=1, step=1)

# 初始化 session_state 用于保存磁铁位置
for i in range(10):
    for axis in ['x', 'y', 'z']:
        key = f"magnet_{i}_{axis}"
        if key not in st.session_state:
            st.session_state[key] = 0.0

# 初始化磁铁列表
magnets = []

# 动态生成磁铁输入
for i in range(num_magnets):
    st.sidebar.subheader(f"磁铁 {i+1}")

    # 默认电压（29.4V 对应原 Pm = 0.5695312497）
    default_U = [29.4, 29.4, 29.4, 29.4, 29.4, 29.4, 29.4, 29.4, 29.4, 29.4]

    # 默认方向向量（全部向右，即X轴正方向）
    default_vectors = [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ]

    U_default = default_U[i] if i < len(default_U) else 29.4
    vector_default = default_vectors[i] if i < len(default_vectors) else [0.0, 1.0, 0.0]

    # 根据是否启用拖动模式选择输入方式（key绑定到session_state）
    if enable_drag:
        # 拖动模式：使用滑块
        st.sidebar.slider(f"X坐标/m", min_value=-0.1, max_value=0.1, value=float(st.session_state[f"magnet_{i}_x"]), step=0.001, format="%.3f", key=f"magnet_{i}_x")
        st.sidebar.slider(f"Y坐标/m", min_value=-0.1, max_value=0.1, value=float(st.session_state[f"magnet_{i}_y"]), step=0.001, format="%.3f", key=f"magnet_{i}_y")
        st.sidebar.slider(f"Z坐标/m", min_value=-0.1, max_value=0.1, value=float(st.session_state[f"magnet_{i}_z"]), step=0.001, format="%.3f", key=f"magnet_{i}_z")
    else:
        # 默认模式：使用输入框
        st.sidebar.number_input(f"X坐标/m", value=float(st.session_state[f"magnet_{i}_x"]), format="%.4f", key=f"magnet_{i}_x")
        st.sidebar.number_input(f"Y坐标/m", value=float(st.session_state[f"magnet_{i}_y"]), format="%.4f", key=f"magnet_{i}_y")
        st.sidebar.number_input(f"Z坐标/m", value=float(st.session_state[f"magnet_{i}_z"]), format="%.4f", key=f"magnet_{i}_z")

    # 从 session_state 读取当前坐标
    coord = [st.session_state[f"magnet_{i}_x"], st.session_state[f"magnet_{i}_y"], st.session_state[f"magnet_{i}_z"]]

    st.sidebar.markdown(f"**磁铁{i + 1}强度（默认为钕磁铁）**")
    U_input = st.sidebar.number_input(f"磁铁{i + 1} 等效输入电压/mV", value=U_default, format="%.1f", key=f"U_{i}")

    st.sidebar.markdown(f"**磁铁{i+1}方向**")
    v = [
        st.sidebar.number_input(f"X", value=vector_default[0], format="%.1f", key=f"vec_x_{i}"),
        st.sidebar.number_input(f"Y", value=vector_default[1], format="%.1f", key=f"vec_y_{i}"),
        st.sidebar.number_input(f"Z", value=vector_default[2], format="%.1f", key=f"vec_z_{i}")
    ]
    
    magnets.append({"coord": coord, "Pm": U_input * 1.9527 / 100.8, "vector": v})


@np.vectorize
def Bx_vec(x, y, z, Pm):
    """向量化Bx计算"""
    r2 = x**2 + y**2 + z**2
    return 3 * u * Pm * z * x / (4 * constants.pi * r2**2.5)

@np.vectorize
def Bz_vec(x, y, z, Pm):
    """向量化Bz计算"""
    r2 = x**2 + y**2 + z**2
    return u * Pm * (2 * z**2 - x**2 - y**2) / (4 * constants.pi * r2**2.5)

def compute_magnetic_field_batch(points, magnets):
    """批量计算磁场，使用向量化操作"""
    N = points.shape[0]
    total_field = np.zeros((N, 3))

    for magnet in magnets:
        cm = np.array(magnet["coord"])
        v_a = np.array(magnet["vector"])
        Pm = magnet["Pm"]
        mol_v_a = np.linalg.norm(v_a)
        if mol_v_a == 0:
            continue

        m = points - cm
        dis = np.linalg.norm(m, axis=1)

        # 点到直线距离
        cross_prod = np.cross(m, v_a)
        d_dot_line = np.linalg.norm(cross_prod, axis=1) / mol_v_a
        h = np.sqrt(np.maximum(dis**2 - d_dot_line**2, 0))

        # 符号
        dp = np.dot(m, v_a)
        plmi = np.sign(dp)
        plmi[dp == 0] = 1

        # 单位向量 unit_x
        tp = dp / (mol_v_a ** 2)
        shadow = cm + np.outer(tp, v_a)
        dis_x = points - shadow
        norm_x = np.linalg.norm(dis_x, axis=1)
        norm_x[norm_x == 0] = 1
        unit_x = dis_x / norm_x[:, np.newaxis]

        # 单位向量 unit_h
        unit_h = v_a / mol_v_a

        # 计算并累加磁场
        Bx_vals = Bx_vec(d_dot_line, np.zeros(N), h * plmi, Pm)
        Bz_vals = Bz_vec(d_dot_line, np.zeros(N), h, Pm)
        total_field += unit_x * Bx_vals[:, np.newaxis] + unit_h * Bz_vals[:, np.newaxis]

    return total_field


def compute_force_torque(pos, m, magnets, h_diff=0.0001):
    """计算试探磁铁的受力和力矩"""
    # 中心点磁场
    B_center = compute_magnetic_field_batch(pos.reshape(1, -1), magnets)[0]

    # X方向梯度
    B_xp = compute_magnetic_field_batch((pos + [h_diff, 0, 0]).reshape(1, -1), magnets)[0]
    B_xm = compute_magnetic_field_batch((pos - [h_diff, 0, 0]).reshape(1, -1), magnets)[0]
    dB_dx = (B_xp - B_xm) / (2 * h_diff)

    # Y方向梯度
    B_yp = compute_magnetic_field_batch((pos + [0, h_diff, 0]).reshape(1, -1), magnets)[0]
    B_ym = compute_magnetic_field_batch((pos - [0, h_diff, 0]).reshape(1, -1), magnets)[0]
    dB_dy = (B_yp - B_ym) / (2 * h_diff)

    # Z方向梯度
    B_zp = compute_magnetic_field_batch((pos + [0, 0, h_diff]).reshape(1, -1), magnets)[0]
    B_zm = compute_magnetic_field_batch((pos - [0, 0, h_diff]).reshape(1, -1), magnets)[0]
    dB_dz = (B_zp - B_zm) / (2 * h_diff)

    # 受力 F = ∇(m·B)
    F = np.array([np.dot(m, dB_dx), np.dot(m, dB_dy), np.dot(m, dB_dz)])

    # 力矩 τ = m × B
    torque = np.cross(m, B_center)

    return F, torque, B_center

# 创建随机均匀采样点（在指定高度Z的XY平面上）
np.random.seed(42)  # 固定随机种子，保证结果可复现
# num_points 和 z_height 由用户在侧边栏设置
x_range = (-0.1, 0.1)
y_range = (-0.1, 0.1)

# 生成随机均匀分布的点
X_random = np.random.uniform(x_range[0], x_range[1], num_points)
Y_random = np.random.uniform(y_range[0], y_range[1], num_points)

# 排除半径
exclude_radius = 0.005  # 0.005m

# 向量化排除计算：检查每个点是否在任一磁铁的排除范围内
# 使用用户选择的高度 z_height
points = np.column_stack([X_random, Y_random, np.full(num_points, z_height)])  # (N, 3)

# 计算每个点到所有磁铁的距离，使用广播
valid_mask = np.ones(num_points, dtype=bool)
for magnet in magnets:
    cm = np.array(magnet["coord"])
    dists = np.linalg.norm(points - cm, axis=1)
    valid_mask &= (dists >= exclude_radius)

# 只保留有效点
valid_indices = np.where(valid_mask)[0]
points_valid = points[valid_mask]
X_valid = X_random[valid_mask]
Y_valid = Y_random[valid_mask]

# 批量计算磁场
total_fields = compute_magnetic_field_batch(points_valid, magnets)

# 计算磁场强度和方向
mag_norms_array = np.linalg.norm(total_fields, axis=1)
nonzero_mask = mag_norms_array > 0

# 只保留非零点
mag_norms_array = mag_norms_array[nonzero_mask]
U_array = np.zeros(np.sum(nonzero_mask))
V_array = np.zeros(np.sum(nonzero_mask))

if np.sum(nonzero_mask) > 0:
    U_array = total_fields[nonzero_mask, 0] / mag_norms_array
    V_array = total_fields[nonzero_mask, 1] / mag_norms_array
    X_valid = X_valid[nonzero_mask]
    Y_valid = Y_valid[nonzero_mask]

# 使用剩余数据的极值作为量程
B_min = mag_norms_array.min()
B_max = mag_norms_array.max()

# 对磁场强度进行对数变换，压缩动态范围
mag_norms_log = np.log10(mag_norms_array + 1e-10)
log_min = mag_norms_log.min()
log_max = mag_norms_log.max()
mag_norms_mapped = (mag_norms_log - log_min) / (log_max - log_min + 1e-10)

# 箭头长度：统一设为相同长度（缩短）
uniform_arrow_length = 0.003
U_scaled = U_array * uniform_arrow_length
V_scaled = V_array * uniform_arrow_length


# ==================== 试探磁铁受力设置 ====================
st.sidebar.markdown("---")
st.sidebar.header("试探磁铁设置")

# 复选框：在图中显示试探磁铁受力
show_force = st.sidebar.checkbox("在图中显示试探磁铁受力", value=False, key="show_force_checkbox")

# 初始化试探磁铁位置的 session_state
for axis, default_val in [('x', 0.05), ('y', 0.0), ('z', 0.0)]:
    key = f"test_magnet_{axis}"
    if key not in st.session_state:
        st.session_state[key] = default_val

# 试探磁铁位置（根据是否启用拖动模式选择输入方式，key绑定到session_state）
if enable_drag:
    st.sidebar.slider("试探磁铁 X坐标/m", min_value=-0.1, max_value=0.1, value=float(st.session_state["test_magnet_x"]), step=0.001, format="%.3f", key="test_magnet_x")
    st.sidebar.slider("试探磁铁 Y坐标/m", min_value=-0.1, max_value=0.1, value=float(st.session_state["test_magnet_y"]), step=0.001, format="%.3f", key="test_magnet_y")
    st.sidebar.slider("试探磁铁 Z坐标/m", min_value=-0.1, max_value=0.1, value=float(st.session_state["test_magnet_z"]), step=0.001, format="%.3f", key="test_magnet_z")
else:
    st.sidebar.number_input("试探磁铁 X坐标/m", value=float(st.session_state["test_magnet_x"]), format="%.4f", key="test_magnet_x")
    st.sidebar.number_input("试探磁铁 Y坐标/m", value=float(st.session_state["test_magnet_y"]), format="%.4f", key="test_magnet_y")
    st.sidebar.number_input("试探磁铁 Z坐标/m", value=float(st.session_state["test_magnet_z"]), format="%.4f", key="test_magnet_z")

# 从 session_state 读取试探磁铁位置
test_magnet_pos = [st.session_state["test_magnet_x"], st.session_state["test_magnet_y"], st.session_state["test_magnet_z"]]

# 试探磁铁磁矩大小（等效电压）
test_magnet_U = st.sidebar.number_input("试探磁铁 等效输入电压/mV", value=29.4, format="%.1f", key="test_U")
test_magnet_Pm = test_magnet_U * 1.9527 / 100.8

# 试探磁铁磁矩方向
test_magnet_dir = [
    st.sidebar.number_input("试探磁铁 磁矩方向 X", value=1.0, format="%.1f", key="test_dir_x"),
    st.sidebar.number_input("试探磁铁 磁矩方向 Y", value=0.0, format="%.1f", key="test_dir_y"),
    st.sidebar.number_input("试探磁铁 磁矩方向 Z", value=0.0, format="%.1f", key="test_dir_z")
]

# 归一化磁矩方向
test_magnet_dir_norm = np.array(test_magnet_dir)
test_magnet_dir_norm = test_magnet_dir_norm / (np.linalg.norm(test_magnet_dir_norm) + 1e-10)

# 自动执行计算（每次进入/刷新网页后自动运行）
with st.spinner("正在计算..."):
        fig, ax = plt.subplots(figsize=(10, 9))
        # 使用高饱和度彩虹色：弱磁场(红) -> 强磁场(紫)
        from matplotlib.colors import ListedColormap
        import matplotlib.cm as cm
        
        # 获取rainbow色图并调整
        rainbow = cm.get_cmap('rainbow', 256)
        rainbow_colors = rainbow(np.linspace(0, 1, 256))
        # 降低明度（使颜色更深沉）
        rainbow_dark = rainbow_colors.copy()
        rainbow_dark[:, :3] = rainbow_colors[:, :3] * 0.8  # 明度降低20%
        dark_cmap = ListedColormap(rainbow_dark)
        
        Q = ax.quiver(X_valid, Y_valid, U_scaled, V_scaled, mag_norms_mapped,
                       cmap=dark_cmap,       # 低明度彩虹色
                       angles='xy',          # 确保箭头方向在数据坐标系中准确
                       scale_units='xy',     # 缩放单位与数据单位一致
                       scale=1,              # 缩放因子为1，因为长度已经计算好
                       width=0.001,          # 箭头杆的宽度
                       alpha=1.0,            # 完全不透明
                       clim=[0, 1])          # 颜色范围

        # 绘制磁铁位置和N/S极方向（红蓝箭头）
        arrow_scale = 0.005  # 箭头长度（缩短）
        for idx, magnet in enumerate(magnets):
            coord = magnet["coord"]
            vec = magnet["vector"]
            
            # 归一化方向向量
            vec_norm = np.linalg.norm(vec)
            if vec_norm > 0:
                vec_unit = np.array(vec) / vec_norm
            else:
                vec_unit = np.array([0, 0, 0])
            
            # 绘制磁铁位置（灰色圆点）
            ax.plot(coord[0], coord[1], 'o', color='gray', markersize=10, zorder=5)
            
            # 绘制N极方向箭头（红色）- 指向向量方向
            ax.arrow(coord[0], coord[1], 
                    vec_unit[0] * arrow_scale, vec_unit[1] * arrow_scale,
                    head_width=0.003, head_length=0.003, 
                    fc='red', ec='red', linewidth=2, zorder=6)
            
            # 绘制S极方向箭头（蓝色）- 与N极相反
            ax.arrow(coord[0], coord[1], 
                    -vec_unit[0] * arrow_scale, -vec_unit[1] * arrow_scale,
                    head_width=0.003, head_length=0.003, 
                    fc='blue', ec='blue', linewidth=2, zorder=6)
            
            # 添加标签
            ax.text(coord[0], coord[1] + 0.005, f'M{idx+1}', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 如果启用了受力计算，绘制试探磁铁和受力矢量
        if show_force:
            # 计算受力和力矩
            pos = np.array(test_magnet_pos)
            m = test_magnet_dir_norm * test_magnet_Pm
            F_3d, torque, B_center = compute_force_torque(pos, m, magnets)
            F = F_3d[:2]  # XY平面分量
            F_magnitude = np.linalg.norm(F)
            torque_xy = torque[:2]
            torque_magnitude = np.linalg.norm(torque)
            torque_xy_magnitude = np.linalg.norm(torque_xy)
            
            # 绘制试探磁铁位置（灰色圆点，与M1相同风格）
            ax.plot(test_magnet_pos[0], test_magnet_pos[1], 'o', 
                   color='gray', markersize=10, zorder=10)
            # 添加P标签
            ax.text(test_magnet_pos[0], test_magnet_pos[1] + 0.005, 'P', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold', zorder=12)
            
            # 绘制试探磁铁的N极方向（红色箭头，与M1相同）
            arrow_scale_p = 0.005  # 与M1相同的箭头长度
            ax.arrow(test_magnet_pos[0], test_magnet_pos[1], 
                    test_magnet_dir_norm[0] * arrow_scale_p, 
                    test_magnet_dir_norm[1] * arrow_scale_p,
                    head_width=0.003, head_length=0.003, 
                    fc='red', ec='red', linewidth=2, zorder=11)
            
            # 绘制受力矢量（绿色粗长箭头）
            if F_magnitude > 1e-15:
                # 固定较长的箭头长度
                force_arrow_length = 0.015
                # 归一化方向
                F_dir = F / F_magnitude if F_magnitude > 0 else np.array([0, 0])
                ax.arrow(test_magnet_pos[0], test_magnet_pos[1],
                        F_dir[0] * force_arrow_length, F_dir[1] * force_arrow_length,
                        head_width=0.004, head_length=0.004,
                        fc='black', ec='black', linewidth=2, zorder=11)
                # 添加受力标注（在箭头尖旁边）
                arrow_tip_x = test_magnet_pos[0] + F_dir[0] * force_arrow_length
                arrow_tip_y = test_magnet_pos[1] + F_dir[1] * force_arrow_length
                # 根据方向调整标注位置
                offset_x = 0.003 if F_dir[0] >= 0 else -0.003
                offset_y = 0.003 if F_dir[1] >= 0 else -0.003
                ax.text(arrow_tip_x + offset_x, arrow_tip_y + offset_y,
                       f'F', fontsize=10, color='black',
                       fontweight='bold', zorder=12, 
                       ha='left' if F_dir[0] >= 0 else 'right',
                       va='bottom' if F_dir[1] >= 0 else 'top')
            
            # 绘制力矩矢量（深紫色）- 偏移显示避免重叠
            # 判断力矩主要方向
            torque_ratio = abs(torque[2]) / (torque_xy_magnitude + abs(torque[2]) + 1e-10)
            
            if torque_ratio > 0.9:  # 力矩主要沿Z轴（垂直于XY平面）
                # 绘制顺时针或逆时针圆弧箭头
                from matplotlib.patches import Arc, FancyArrowPatch
                arc_radius = 0.012
                
                if torque[2] > 0:  # 逆时针（从Z轴正方向看）↺
                    # 绘制270度圆弧，从底部(270°)逆时针旋转到右侧(540°=180°)
                    arc = Arc((test_magnet_pos[0], test_magnet_pos[1]), 
                             arc_radius * 2, arc_radius * 2,
                             angle=0, theta1=90, theta2=360,
                             color='indigo', linewidth=3, zorder=11)
                    ax.add_patch(arc)
                    # 箭头头部在圆弧末端（右侧），指向逆时针方向（向左）
                    arrow_tip_x = test_magnet_pos[0] + arc_radius
                    arrow_tip_y = test_magnet_pos[1]
                    # 箭头方向：向上（逆时针切线方向）
                    ax.arrow(arrow_tip_x, arrow_tip_y,
                            0.00, 0.004,
                            head_width=0.004, head_length=0.004,
                            fc='indigo', ec='indigo', linewidth=1, zorder=11)
                else:  # 顺时针（从Z轴正方向看）↻
                    # 绘制270度圆弧，从底部(270°)顺时针旋转到左侧(0°=-90°)
                    arc = Arc((test_magnet_pos[0], test_magnet_pos[1]), 
                             arc_radius * 2, arc_radius * 2,
                             angle=0, theta1=180, theta2=450,
                             color='indigo', linewidth=3, zorder=11)
                    ax.add_patch(arc)
                    # 箭头头部在圆弧末端（左侧），指向顺时针方向（向上）
                    arrow_tip_x = test_magnet_pos[0] - arc_radius
                    arrow_tip_y = test_magnet_pos[1]
                    # 箭头方向：向上（顺时针切线方向）
                    ax.arrow(arrow_tip_x, arrow_tip_y,
                            0, 0.004,
                            head_width=0.004, head_length=0.004,
                            fc='indigo', ec='indigo', linewidth=1, zorder=11)
                
                # 添加力矩标注（位于箭头尖端附近）
                if torque[2] > 0:  # 逆时针，箭头在右侧
                    label_x = test_magnet_pos[0] + arc_radius + 0.003
                    label_y = test_magnet_pos[1] + 0.008
                else:  # 顺时针，箭头在左侧
                    label_x = test_magnet_pos[0] - arc_radius - 0.003
                    label_y = test_magnet_pos[1] + 0.008
                ax.text(label_x, label_y,
                       f'τz', fontsize=10, color='indigo',
                       fontweight='bold', zorder=12, ha='center', va='bottom')
            
            elif torque_xy_magnitude > 1e-15:  # 力矩主要在XY平面内
                # 固定箭头长度
                torque_arrow_length = 0.015
                # 归一化方向
                torque_dir = torque_xy / torque_xy_magnitude if torque_xy_magnitude > 0 else np.array([0, 0])
                # 偏移位置显示力矩（向右下方偏移）
                offset_x = 0.008
                offset_y = -0.008
                ax.arrow(test_magnet_pos[0] + offset_x, test_magnet_pos[1] + offset_y,
                        torque_dir[0] * torque_arrow_length, torque_dir[1] * torque_arrow_length,
                        head_width=0.003, head_length=0.003,
                        fc='indigo', ec='indigo', linewidth=3, zorder=11)
                # 添加力矩标注（位于箭头尖端附近，字号加大）
                ax.text(test_magnet_pos[0] + offset_x + torque_dir[0] * torque_arrow_length * 1.5, 
                       test_magnet_pos[1] + offset_y + torque_dir[1] * torque_arrow_length * 1.5,
                       f'τ', fontsize=10, color='indigo',
                       fontweight='bold', zorder=12, ha='center', va='center')
        
        ax.set_title('Magnetic Field Visualization')
        ax.set_xlabel('X Axis (m)')
        ax.set_ylabel('Y Axis (m)')
        ax.axis('equal')  # 保证X轴和Y轴比例相同，图形不变形
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        
        # 添加颜色条
        cbar = plt.colorbar(Q, ax=ax, shrink=0.8)
        cbar.set_label('log10(B) Relative Magnetic Induction', rotation=270, labelpad=25)
        # 设置颜色条刻度为对数空间的相对位置
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        # 计算对应实际磁感应强度值的标签
        tick_values = [log_min + t * (log_max - log_min) for t in [0, 0.25, 0.5, 0.75, 1.0]]
        tick_labels = [f'{10**v:.2e}' for v in tick_values]
        cbar.set_ticklabels(tick_labels)
        
        # 将图表保存到内存
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        
        # 显示图片和受力分析结果（全宽布局）
        st.success("计算完成！")
        
        # 使用全宽布局，图片和参数并排（两列：大图+参数列）
        col_img, col_data = st.columns([3, 1])
        
        with col_img:
            base_caption = f"Z={z_height:.3f}m 高度截面 - 磁感线分布图"
            caption_text = f"{base_caption}（灰点=磁铁/P，红/蓝=N/S极，黑箭头=受力，紫箭头=力矩）" if show_force else f"{base_caption}（灰点表示磁铁位置，红色箭头表示N极方向）"
            st.image(buf, caption=caption_text, use_container_width=True)
        
        with col_data:
            # 显示磁铁参数信息（字号加大）
            st.markdown("<h4 style='font-size:16px; margin-bottom:8px;'>磁铁参数</h4>", unsafe_allow_html=True)
            for idx, magnet in enumerate(magnets):
                st.markdown(f"<p style='font-size:13px; margin:3px 0;'><b>M{idx+1}:</b> ({magnet['coord'][0]:.4f}, {magnet['coord'][1]:.4f}, {magnet['coord'][2]:.4f})<br>({magnet['vector'][0]:.1f}, {magnet['vector'][1]:.1f}, {magnet['vector'][2]:.1f})</p>", unsafe_allow_html=True)
            
            # 如果启用了受力计算，显示详细的受力分析结果
            if show_force:
                st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)
                st.markdown("<h4 style='font-size:16px; margin-bottom:8px;'>试探磁铁受力分析</h4>", unsafe_allow_html=True)

                # 计算3D受力和力矩（用于显示）
                pos_disp = np.array(test_magnet_pos)
                m_disp = test_magnet_dir_norm * test_magnet_Pm
                F_disp, torque_disp, B_c = compute_force_torque(pos_disp, m_disp, magnets)

                st.markdown(f"<p style='font-size:12px; margin:2px 0;'><b>位置:</b> ({test_magnet_pos[0]:.4f}, {test_magnet_pos[1]:.4f}, {test_magnet_pos[2]:.4f}) m</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:12px; margin:2px 0;'><b>磁矩:</b> {test_magnet_Pm:.6f} A·m²</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:12px; margin:2px 0;'><b>方向:</b> ({test_magnet_dir_norm[0]:.3f}, {test_magnet_dir_norm[1]:.3f}, {test_magnet_dir_norm[2]:.3f})</p>", unsafe_allow_html=True)

                st.markdown(f"<p style='font-size:12px; margin:2px 0;'><b>B:</b> ({B_c[0]:.3e}, {B_c[1]:.3e}, {B_c[2]:.3e}) T</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:12px; margin:2px 0;'><b>|B|:</b> {np.linalg.norm(B_c):.3e} T</p>", unsafe_allow_html=True)

                st.markdown(f"<p style='font-size:12px; margin:2px 0;'><b>F:</b> ({F_disp[0]:.3e}, {F_disp[1]:.3e}, {F_disp[2]:.3e}) N</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:12px; margin:2px 0;'><b>|F|:</b> {np.linalg.norm(F_disp):.3e} N</p>", unsafe_allow_html=True)

                st.markdown(f"<p style='font-size:12px; margin:2px 0;'><b>τ:</b> ({torque_disp[0]:.3e}, {torque_disp[1]:.3e}, {torque_disp[2]:.3e}) N·m</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:12px; margin:2px 0;'><b>|τ|:</b> {np.linalg.norm(torque_disp):.3e} N·m</p>", unsafe_allow_html=True)
            else:
                st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)
                st.markdown("<h4 style='font-size:16px; margin-bottom:8px;'>受力分析</h4>", unsafe_allow_html=True)
                st.markdown('<p style="font-size:13px; color:gray;">勾选"显示试探磁铁受力"查看详情</p>', unsafe_allow_html=True)
