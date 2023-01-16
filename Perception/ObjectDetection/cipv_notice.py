def cipv_notice(obstacles, bev_lanes):
    cipv = None
    if obstacles is not None:
        ego_left = None
        ego_right = None
        adjacent_left = None
        adjacent_right = None
        for bev_lane in bev_lanes:
            if bev_lane.position_type == -1:
                ego_left = bev_lane
            elif bev_lane.position_type == 1:
                ego_right = bev_lane
            elif bev_lane.position_type == -2:
                adjacent_left = bev_lane
            elif bev_lane.position_type == 2:
                adjacent_right = bev_lane
        for obstacle in obstacles:
            x = obstacle.position[0] + obstacle.position[2] / 2
            y = obstacle.position[1] + obstacle.position[3]  # 计算尾部中心
            if y >= 320:  # 过滤监控范围之外的物体
                if y <= 470:
                    if ego_left is not None and ego_right is not None:  # 前方车道的CIPV
                        if ego_left.fit[0]*y**3 + ego_left.fit[1]*y**2 + ego_left.fit[2]*y + ego_left.fit[3] < x and ego_right.fit[0]*y**3 + ego_right.fit[1]*y**2 + ego_right.fit[2]*y + ego_right.fit[3] > x:
                            obstacle.lane = 0
                            if cipv is None:
                                cipv = obstacle
                            else:
                                if cipv.position[1] + cipv.position[3] <= y:
                                    cipv = obstacle
                            continue
                    else:
                        if x >= 390 and x <= 410:
                            if cipv is None:
                                cipv = obstacle
                            else:
                                if cipv.position[1] + cipv.position[3] <= y:
                                    cipv = obstacle
                            obstacle.lane = 0
                            continue
                if adjacent_left is not None and ego_left is not None:  # 左车道的CIPV
                    if adjacent_left.fit[0]*y**3 + adjacent_left.fit[1]*y**2 + adjacent_left.fit[2]*y + adjacent_left.fit[3] < x and ego_left.fit[0]*y**3 + ego_left.fit[1]*y**2 + ego_left.fit[2]*y + ego_left.fit[3] > x:
                        obstacle.lane = -1
                        continue
                else:
                    if x >= 370 and x <= 390:
                        obstacle.lane = -1
                        continue
                if ego_right is not None and adjacent_right is not None:  # 右车道的CIPV
                    if ego_right.fit[0]*y**3 + ego_right.fit[1]*y**2 + ego_right.fit[2]*y + ego_right.fit[3] < x and adjacent_right.fit[0]*y**3 + adjacent_right.fit[1]*y**2 + adjacent_right.fit[2]*y + adjacent_right.fit[3] > x:
                        obstacle.lane = 1
                        continue
                else:
                    if x >= 410 and x <= 430:
                        obstacle.lane = 1
                        continue
        return obstacles, cipv
    else:
        return None, None


