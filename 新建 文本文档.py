import cv2
import numpy as np
import math

def get_angle_radar_mode(image_path):
    # 1. 读取
    img = cv2.imread(image_path)
    if img is None: return None
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    # 2. 预处理 (取V通道)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    
    # ==============================
    # 核心参数
    RADIUS = 50       # 扫描半径 (根据你的截图大小调整，要避开中心箭头)
    THRESHOLD = 140   # 亮度阈值 (大于这个值认为是白色扇形)
    MIN_FOV = 60      # 最小视角跨度 (过滤道路噪点)
    MAX_FOV = 100     # 最大视角跨度
    # ==============================

    # 3. 环形采样
    values = []
    for i in range(360):
        # 0度是向右，顺时针
        rad = math.radians(i)
        x = int(cx + RADIUS * math.cos(rad))
        y = int(cy + RADIUS * math.sin(rad))
        
        # 边界检查
        val = 0
        if 0 <= x < w and 0 <= y < h:
            val = v_channel[y, x]
        values.append(val)

    # 4. 寻找连续的高亮片段
    # 拼接两圈处理跨0度问题
    scan_data = values + values
    
    best_segment = None # (start, end, length)
    
    in_block = False
    start_idx = 0
    
    for i, val in enumerate(scan_data):
        if val >= THRESHOLD:
            if not in_block:
                in_block = True
                start_idx = i
        else:
            if in_block:
                in_block = False
                length = i - start_idx
                
                # 过滤逻辑：只取长度合理的片段 (大概是FOV的角度)
                if MIN_FOV <= length <= MAX_FOV:
                    # 这里的逻辑是：取最长的那个片段(最稳)
                    if best_segment is None or length > best_segment[2]:
                        best_segment = (start_idx, i, length)
                        
    if best_segment is None:
        print("未检测到扇形边缘")
        return None
        
    start_deg = best_segment[0] % 360
    end_deg = best_segment[1] % 360
    
    # 5. 计算中心
    # 如果跨越了0度 (比如 start=330, end=30)
    calc_end = end_deg
    if end_deg < start_deg:
        calc_end += 360
        
    center_deg = (start_deg + calc_end) / 2
    center_deg = center_deg % 360
    
    # 6. 坐标转换 (OpenCV 0度向右 -> 地图 0度向上)
    # 我们的扫描是以图像X轴正向为0度，顺时针扫描
    # 结果: 0=右, 90=下, 180=左, 270=上
    # 目标: 0=上, 90=右, 180=下, 270=左
    final_angle = (center_deg + 90) % 360

    # --- 可视化调试 ---
    debug_img = img.copy()
    # 画出扫描圈
    cv2.circle(debug_img, (cx, cy), RADIUS, (255, 0, 0), 1)
    
    # 画出检测到的两条边
    s_rad = math.radians(start_deg)
    e_rad = math.radians(end_deg)
    sx = int(cx + RADIUS * math.cos(s_rad))
    sy = int(cy + RADIUS * math.sin(s_rad))
    ex = int(cx + RADIUS * math.cos(e_rad))
    ey = int(cy + RADIUS * math.sin(e_rad))
    
    cv2.line(debug_img, (cx, cy), (sx, sy), (0, 0, 255), 2) # 起始边(红)
    cv2.line(debug_img, (cx, cy), (ex, ey), (0, 0, 255), 2) # 结束边(红)
    
    # 画出计算出的方向
    c_rad = math.radians(center_deg)
    mx = int(cx + (RADIUS+20) * math.cos(c_rad))
    my = int(cy + (RADIUS+20) * math.sin(c_rad))
    cv2.line(debug_img, (cx, cy), (mx, my), (0, 255, 0), 2) # 中心线(绿)
    
    cv2.imwrite('radar_debug.png', debug_img)
    
    return final_angle

if __name__ == "__main__":
    angle = get_angle_radar_mode('map.png')
    if angle is not None:
        print(f"当前视角: {angle:.1f} 度")