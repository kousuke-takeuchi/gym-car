import math
import copy

import gym
import numpy as np


env = gym.make('CarRacing-v0')
observation = env.reset()

car_color = np.array([204, 0, 0])
wall_color = np.array([102, 204, 102])
road_color = np.array([105, 105, 105])


def histogram(observation):
    """
    画面の色素ヒストグラムを計算
    """
    result = {}
    for row in observation:
        for rgb in row:
            hexrgb = hex(rgb[0])[2:] + hex(rgb[1])[2:] + hex(rgb[2])[2:]
            result[hexrgb] = result.get(hexrgb, 0) + 1
    return result

def is_wall(position, observation):
    """
    position:位置にあるピクセルが、壁かどうか
    """
    a = observation[position[0]][position[1]] == wall_color
    if a.all():
        return True
    return False

def current_direction(prev_position, current_position):
    """
    以前の位置と現在の位置から、現在向いている方角を計算する
    """
    d_x = current_position[0] - prev_position[0]
    d_y = current_position[1] - prev_position[1]
    return (d_x, d_y)

def heuristic_f(position, observation, prev_position):
    """
    周りの壁の位置を探索して、一番壁から遠い方角にハンドルを向ける
    ハンドルの方向に壁がない場合、アクセルを踏む
    壁がある場合、ブレーキを踏む
    """
    width, height = observation.shape[:2]
    # ８方向を探索する (horizontal, vertical)
    directions = {
        'n': (0, 1),
        'ne': (1, 1),
        'e': (1, 0),
        'se': (1, -1),
        's': (0, -1),
        'sw': (-1, -1),
        'w': (-1, 0),
        'nw': (-1, 1),
    }

    # 探索する方向
    OPEN = [dict(name=n, direction=d, score=0) for n, d in directions.items()]
    CLOSE = []
    for o in OPEN:
        direction = o['direction']
        score = o['score']
        # 探索中のピクセル位置
        p = copy.deepcopy(position)
        s_x, s_y = p[0], p[1]
        # 終了条件
        def flag(s_x, s_y):
            return s_x > 0 and s_x < width and s_y > 0 and s_y < height
        while flag(s_x, s_y) and not is_wall((s_x, s_y), observation):
            score += 1
            s_x += direction[0]
            s_y += direction[1]
        o['score'] = score
        CLOSE.append(o)

    # 探索結果から、一番スコアの高い(壁が遠い)方向にハンドルを切る
    max_d = max(CLOSE, key=lambda c: c['score'])

    # 現在の方角
    if not prev_position:
        handle = 0.0
    else:
        c_d = current_direction(position, prev_position)
        # 現在の方角とハンドルを切る方角のtan(θ)を計算し、θをハンドルのパワーとする
        d_x = max_d['direction'][0] - c_d[0]
        d_y = max_d['direction'][1] - c_d[1]
        handle = math.atan2(d_y, d_x) / 10.0

    accel, brake = 0, 0
    # スコアが45以上の場合、アクセルを踏む
    if max_d['score'] > 45:
        accel = max_d['score'] / 1000.0
    else:
        brake = 1.0 / max_d['score']
    print(max_d['name'])
    return [handle, accel, brake]


if __name__ == '__main__':
    done = False
    w = 0
    wait = 50
    position = None
    while not done:
        # 画面を描画
        env.render()

        if w < wait:
            w += 1
            observation, reward, done, info = env.step([0.0, 0.0, 0.0])
            continue

        # 赤色のピクセル位置を検索
        pixels = np.where(observation[:,:,0]==car_color[0])

        # ピクセルの位置を平均して、現在の車体の場所を特定する
        prev_position = copy.copy(position)
        x = int(sum(pixels[0])/len(pixels[0]))
        y = int(sum(pixels[1])/len(pixels[1]))
        position = (x, y)

        # ヒューリスティック関数から、ハンドル方向、アクセル、ブレーキを決める
        # action = [right, accel, brake]
        action = heuristic_f(position, observation, prev_position)

        observation, reward, done, info = env.step(action)
