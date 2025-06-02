import csv
import time
from pynput import keyboard

# Путь для сохранения логов
LOG_FILE = "test_ds_z.csv"

# Словарь для отслеживания времени нажатия клавиш
key_press_times = {}

# Переменная для хранения времени последнего события
last_event_time = None

# Счётчик записей
row_count = 1  # начинаем с 1, потому что первая строка — заголовок


# Функция инициализации CSV файла
def init_csv():
    with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_press",   # Время нажатия (микросекунды)
            "timestamp_release", # Время отпускания (микросекунды)
            "key_code",          # Код клавиши (например, 'a', 'Key.shift')
            "key_char",          # Символ (если доступен)
            "hold_time",         # Время удержания (мс)
            "latency"            # Интервал между нажатиями (мс)
        ])


# Обработка нажатия клавиши
def on_press(key):
    global last_event_time
    try:
        key_char = key.char if hasattr(key, 'char') else str(key)

        if key_char.startswith('Key.'):
            return

        key_press_times[key] = time.time()

    except Exception as e:
        print(f"Ошибка в on_press: {e}")


# Обработка отпускания клавиши
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
                return False  # остановит listener

    except Exception as e:
        print(f"Ошибка в on_release: {e}")


# Инициализация CSV
init_csv()

# Запуск слушателя клавиатуры
with keyboard.Listener(
    on_press=on_press,
    on_release=on_release
) as listener:
    print("Логирование запущено. Для остановки нажмите Ctrl+C")
    listener.join()

print("Программа завершена.")
