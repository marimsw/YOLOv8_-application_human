import tkinter as tk
from tkinter import Canvas, PhotoImage, messagebox
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tkinterdnd2 import DND_FILES, TkinterDnD

# Загрузка модели YOLOv8
model = YOLO('yolov8n.pt')  # Замените 'yolov8n.pt' на нужную вам модель


class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Обнаружение людей")

        self.canvas = Canvas(root, width=800, height=600)
        self.canvas.pack()

        self.button = tk.Button(root, text="Найти человека", command=self.detect_person)
        self.button.pack()

        self.image_path = None

        # Перетаскивание изображения
        self.canvas.drop_target_register(DND_FILES)
        self.canvas.dnd_bind('<<Drop>>', self.on_drop)

    def on_drop(self, event):
        # Удаляем фигурные скобки из пути
        self.image_path = event.data.strip('{}')
        self.load_image(self.image_path)

    def load_image(self, path):
        self.img = PhotoImage(file=path)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

    def detect_person(self):
        if not self.image_path:
            messagebox.showerror("Ошибка", "Пожалуйста, перетащите изображение.")
            return

        # Обработка изображения с помощью YOLOv8
        img = cv2.imread(self.image_path)
        results = model(img)

        # Получение результатов
        person_count = 0
        for result in results[0].boxes.data:  # Получаем данные о предсказаниях
            if int(result[5]) == 0:  # 0 - это класс "человек"
                person_count += 1
                x1, y1, x2, y2 = map(int, result[:4])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Вычисление процента нахождения человека
        total_objects = len(results[0].boxes.data)
        if total_objects > 0:
            percentage = (person_count / total_objects) * 100
        else:
            percentage = 0

        # Отображение результата с помощью matplotlib
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'Человек: {percentage:.2f}%')
        plt.axis('off')  # Отключить оси
        plt.show()


if __name__ == "__main__":
    root = TkinterDnD.Tk()  # Используем TkinterDnD вместо обычного Tk
    app = ImageApp(root)
    root.mainloop()
