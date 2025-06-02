import csv
import time
from pynput import keyboard

LOG_FILE = "test_ds_z.csv"

key_press_times = {}

last_event_time = None

row_count = 1

def init_csv():
    with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_press",   
            "timestamp_release", 
            "key_code",         
            "key_char",        
            "hold_time",         
            "latency"           
        ])


def on_press(key):
    global last_event_time
    try:
        key_char = key.char if hasattr(key, 'char') else str(key)

        if key_char.startswith('Key.'):
            return

        key_press_times[key] = time.time()

    except Exception as e:
        print(f"Ошибка в on_press: {e}")


def on_release(key):
    global last_event_time, row_count
    try:
        key_char = key.char if hasattr(key, 'char') else str(key)

        if key_char.startswith('Key.'):
            return

        release_time = time.time()
        press_time = key_press_times.get(key)

        if press_time:
            hold_time = (release_time - press_time) * 1000  # ms
            latency = (press_time - last_event_time) * 1000 if last_event_time else 0
            last_event_time = release_time

            with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    int(press_time * 1_000_000),
                    int(release_time * 1_000_000),
                    str(key),
                    key_char,
                    round(hold_time, 2),
                    round(latency, 2)
                ])
                row_count += 1

            # Проверка: если достигли 513 строк (1 заголовок + 512 данных)
            if row_count >= 513:
                print("Достигнуто 513 строк. Завершаем логирование.")
                return False 

    except Exception as e:
        print(f"Ошибка в on_release: {e}")

init_csv()

with keyboard.Listener(
    on_press=on_press,
    on_release=on_release
) as listener:
    print("Логирование запущено. Для остановки нажмите Ctrl+C")
    listener.join()

print("Программа завершена.")
