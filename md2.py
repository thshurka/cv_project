import torch
import cv2
import numpy as np
import os
import time
import sqlite3


# Загрузка модели
model_path = 'yolov5/runs/train/exp8/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Классы объектов
classes = ['closed-door', 'helmet', 'human', 'opened-door', 'robot']
colors = {'robot': (0, 255, 0), 'human': (255, 0, 0), 'opened-door': (0, 0, 255), 'helmet': (255, 255, 0), 'closed-door': (255, 0, 255)}

# Создание таблицы нарушений
def create_database():
    conn = sqlite3.connect('violations.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            violation TEXT,
            screenshot_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

create_database()

# Функция для логирования нарушений
def log_violation(message, frame):
    screenshot_dir = 'violations_screenshots'
    os.makedirs(screenshot_dir, exist_ok=True)
    screenshot_path = os.path.join(screenshot_dir, f"violation_{int(time.time())}.jpg")
    cv2.imwrite(screenshot_path, frame)
    
    conn = sqlite3.connect('violations.db')
    cursor = conn.cursor()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('INSERT INTO violations (timestamp, violation, screenshot_path) VALUES (?, ?, ?)', 
                    (timestamp, message, screenshot_path))
    conn.commit()
    print(f"Нарушение записано: {message}, скриншот: {screenshot_path}")
    conn.close()


# Определение движения робота
movement_threshold = 10  # Порог для определения движения
robot_position = None
robot_last_seen = 0  # Время, когда последний раз робот был зафиксирован
activity_timeout = 5  # Порог времени для определения "неактивности"

def has_robot_moved(new_position, old_position):
    if old_position is None:
        return True  # Если старой позиции нет, робот "двигается"
    
    x_old, y_old, x_old2, y_old2 = old_position
    x_new, y_new, x_new2, y_new2 = new_position
    # Расчет центров
    center_old = ((x_old + x_old2) / 2, (y_old + y_old2) / 2)
    center_new = ((x_new + x_new2) / 2, (y_new + y_new2) / 2)
    # Смещение по X и Y
    delta_x = abs(center_new[0] - center_old[0])
    delta_y = abs(center_new[1] - center_old[1])
    # Определяем, произошло ли значительное смещение
    return delta_x > movement_threshold or delta_y > movement_threshold

def update_robot_status(new_position):
    global robot_position, robot_last_seen
    current_time = time.time()
    
    if has_robot_moved(new_position, robot_position):
        robot_position = new_position  # Обновляем позицию робота
        robot_last_seen = current_time  # Обновляем время последней активности
        return True  # Робот активен
    
    # Проверяем, не прошло ли слишком много времени с последней активности
    if current_time - robot_last_seen > activity_timeout:
        return False  # Робот неактивен
    
    return True  # Робот всё еще активен, хотя не двигается

# Основная функция для обработки кадров
def process_frame(frame):
    global robot_position

    results = model(frame)
    detections = results.pred[0]
    robot_active = False
    human_in_robot_area = False
    opened_door_while_robot_active = False
    helmet_worn = False

    for det in detections:
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
        if conf > 0.2:
            label = classes[int(cls)]
            color = colors.get(label, (255, 255, 255))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            text = f"{label}: {conf:.2f}"
            cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if label == 'robot':
                new_position = (x1, y1, x2, y2)
                # Определяем статус робота с помощью обновленной функции
                robot_active = update_robot_status(new_position)

            elif label == 'human':
                human_center = ((x1 + x2) // 2, y2)
                if is_point_in_robot_area(human_center):
                    human_in_robot_area = True

            elif label == 'helmet':
                helmet_worn = is_helmet_worn(detections)

            elif label == 'opened-door' and robot_active:
                opened_door_while_robot_active = True
    
    # Отображаем полигон рабочей зоны робота
    cv2.polylines(frame, [robot_area_polygon], isClosed=True, color=(0, 255, 255), thickness=2)
    # Обрабатываем возможные нарушения
    process_violations(robot_active, human_in_robot_area, helmet_worn, opened_door_while_robot_active, frame)

    return frame

# Проверка на наличие каски
def is_helmet_worn(detections):
    helmets = []
    humans = []
    
    for det in detections:
        cls = int(det[5])
        if classes[cls] == 'human':
            humans.append((det[0], det[1], det[2], det[3]))
        elif classes[cls] == 'helmet':
            helmets.append((det[0], det[1], det[2], det[3]))

    for human in humans:
        human_x1, human_y1, human_x2, human_y2 = human
        for helmet in helmets:
            helmet_x1, helmet_y1, helmet_x2, helmet_y2 = helmet
            if (helmet_y1 < human_y1 + (human_y2 - human_y1) / 2) and \
            (helmet_x1 < human_x2 and helmet_x2 > human_x1):
                return True
    return False

# Обработка нарушений
def process_violations(robot_active, human_in_robot_area, helmet_worn, opened_door_while_robot_active, frame):
    print(f"Статус робота: {'Активен' if robot_active else 'Неактивен'}")
    print(f"Человек в зоне робота: {'Да' if human_in_robot_area else 'Нет'}")
    print(f"Каска надета: {'Да' if helmet_worn else 'Нет'}")
    print(f"Дверь открыта при активном роботе: {'Да' if opened_door_while_robot_active else 'Нет'}")
    
    if human_in_robot_area and robot_active:
        if not helmet_worn:
            log_violation("helmet", frame)
        else:
            log_violation("working_robot")
    if opened_door_while_robot_active:
        log_violation("opened_door", frame)
# Определение рабочей зоны робота
robot_area_polygon = np.array([[1020,1074],[1918,1073],[1918,764],[1078,668],[46,1027],[98,1078]])

def is_point_in_robot_area(point):
    return cv2.pointPolygonTest(robot_area_polygon, point, False) >= 0

# Основная функция для обработки кадров
def process_frame(frame):
    global robot_position

    results = model(frame)
    detections = results.pred[0]
    robot_active = False
    human_in_robot_area = False
    opened_door_while_robot_active = False
    helmet_worn = False

    for det in detections:
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
        if conf > 0.2:
            label = classes[int(cls)]
            color = colors.get(label, (255, 255, 255))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            text = f"{label}: {conf:.2f}"
            cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if label == 'robot':
                new_position = (x1, y1, x2, y2)
                if has_robot_moved(new_position, robot_position):
                    robot_active = True
                    robot_position = new_position

            elif label == 'human':
                human_center = ((x1 + x2) // 2, y2)
                if is_point_in_robot_area(human_center):
                    human_in_robot_area = True
            elif label == 'helmet':
                helmet_worn = is_helmet_worn(detections)

            elif label == 'opened-door' and robot_active:
                opened_door_while_robot_active = True
                
    cv2.polylines(frame, [robot_area_polygon], isClosed=True, color=(0, 255, 255), thickness=2)
    process_violations(robot_active, human_in_robot_area, helmet_worn, opened_door_while_robot_active, frame)

    return frame

# Функция для обработки видео
def monitor_safety(video_source):
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)
        cv2.imshow('Safety Monitoring', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Запуск мониторинга
monitor_safety()