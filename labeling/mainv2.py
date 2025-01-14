import os
import shutil
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

class ImageLabelingApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Image Labeling App")

        # Załaduj konfigurację z pliku config.txt
        self.load_config()

        # Inicjalizacja zmiennych
        self.rectangles = []  # Lista prostokątów dla aktualnego obrazu
        self.current_image_idx = 0
        self.start_x = None
        self.start_y = None
        self.current_rectangle = None
        self.selected_category = None

        # Ustawienie interfejsu
        self.canvas = tk.Canvas(self.window, width=800, height=600)
        self.canvas.pack()

        self.category_label = tk.Label(self.window, text="Wybierz kategorię: Brak")
        self.category_label.pack()

        self.image_paths = [os.path.join(self.image_folder, f) for f in os.listdir(self.image_folder) if f.endswith(('jpg', 'jpeg', 'png'))]
        if not self.image_paths:
            messagebox.showerror("Błąd", "Brak obrazów w folderze do etykietowania.")
            self.window.destroy()
            return

        self.load_image(self.image_paths[self.current_image_idx])

        # Przyciski nawigacyjne
        self.prev_button = tk.Button(self.window, text="Poprzednie", command=self.prev_image)
        self.prev_button.pack(side=tk.LEFT)
        self.next_button = tk.Button(self.window, text="Następne", command=self.next_image)
        self.next_button.pack(side=tk.RIGHT)

        self.window.bind("<KeyPress>", self.on_key_press)
        self.canvas.bind("<ButtonPress-1>", self.start_rect)
        self.canvas.bind("<B1-Motion>", self.draw_rect)
        self.canvas.bind("<ButtonRelease-1>", self.finish_rect)

        self.window.mainloop()

    def load_config(self):
        """Ładowanie ustawień z pliku konfiguracyjnego"""
        self.categories = {}
        self.keybinds = {}
        self.colors = {}  # Mapowanie kategorii na kolory
        available_colors = ["red", "green", "blue", "orange", "purple", "cyan", "magenta"]
        color_index = 0

        with open('config.txt', 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue  # Pomijamy komentarze i puste linie
                if line.startswith("image_folder="):
                    self.image_folder = line.split('=')[1]
                elif line.startswith("output_folder="):
                    self.output_folder = line.split('=')[1]
                elif line.startswith("deleted_folder="):
                    self.deleted_folder = line.split('=')[1]
                elif ',' in line:  # Format numer,kategoria,skróty
                    category_id, category_name, keybind = line.split(',')
                    category_id = int(category_id)
                    self.categories[category_name] = category_id
                    self.keybinds[keybind] = category_id
                    self.colors[category_id] = available_colors[color_index % len(available_colors)]
                    color_index += 1

        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.deleted_folder, exist_ok=True)

    def load_image(self, image_path):
        """Ładowanie i wyświetlanie obrazu na canvasie"""
        self.image = Image.open(image_path)
        self.image.thumbnail((800, 600))
        self.image_tk = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)

    def start_rect(self, event):
        """Inicjalizowanie rysowania prostokąta"""
        if self.selected_category is None:
            messagebox.showerror("Błąd", "Nie wybrano kategorii!")
            return
        self.start_x = event.x
        self.start_y = event.y

    def draw_rect(self, event):
        """Rysowanie prostokąta na canvasie"""
        if self.current_rectangle:
            self.canvas.delete(self.current_rectangle)
        self.current_rectangle = self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline=self.colors[self.selected_category])

    def finish_rect(self, event):
        """Zakończenie rysowania prostokąta"""
        if self.start_x is None or self.start_y is None:
            return
        end_x = event.x
        end_y = event.y
        x1, x2 = sorted([self.start_x, end_x])
        y1, y2 = sorted([self.start_y, end_y])
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2 / self.image.width
        center_y = (y1 + y2) / 2 / self.image.height
        self.rectangles.append((self.selected_category, center_x, center_y, width / self.image.width, height / self.image.height))
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=self.colors[self.selected_category], width=2)
        self.start_x = None
        self.start_y = None
        self.current_rectangle = None

    def on_key_press(self, event):
        """Wybór kategorii za pomocą skrótu klawiszowego"""
        if event.char in self.keybinds:
            self.selected_category = self.keybinds[event.char]
            category_name = next(key for key, value in self.categories.items() if value == self.selected_category)
            self.category_label.config(text=f"Wybierz kategorię: {category_name}")
        elif event.char == 'z':
            self.undo_last_rectangle()
        elif event.keysym == 'space':
            self.next_image()

    def undo_last_rectangle(self):
        """Cofnięcie ostatnio utworzonej etykiety"""
        if self.rectangles:
            self.rectangles.pop()
            self.canvas.delete("all")
            self.load_image(self.image_paths[self.current_image_idx])
            for rect in self.rectangles:
                category, center_x, center_y, width, height = rect
                x1 = (center_x - width / 2) * self.image.width
                y1 = (center_y - height / 2) * self.image.height
                x2 = (center_x + width / 2) * self.image.width
                y2 = (center_y + height / 2) * self.image.height
                self.canvas.create_rectangle(x1, y1, x2, y2, outline=self.colors[category], width=2)

    def prev_image(self):
        """Przejście do poprzedniego zdjęcia"""
        if self.current_image_idx > 0:
            self.save_labels()
            self.move_image()
            self.current_image_idx -= 1
            self.load_image(self.image_paths[self.current_image_idx])
            self.rectangles.clear()

    def next_image(self):
        """Przejście do następnego zdjęcia"""
        if self.current_image_idx < len(self.image_paths):
            self.save_labels()
            self.move_image()
            self.current_image_idx += 1
            if self.current_image_idx < len(self.image_paths):
                self.load_image(self.image_paths[self.current_image_idx])
                self.rectangles.clear()
            else:
                messagebox.showinfo("Koniec", "Wszystkie obrazy zostały zetykietowane.")
                self.window.quit()

    def save_labels(self):
        """Zapisanie etykiet do pliku txt"""
        image_name = os.path.basename(self.image_paths[self.current_image_idx])
        label_file = os.path.join(self.output_folder, image_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt'))
        with open(label_file, 'w') as f:
            for label in self.rectangles:
                f.write(f"{label[0]} {label[1]} {label[2]} {label[3]} {label[4]}\n")

    def move_image(self):
        """Przeniesienie zetykietowanego obrazu do folderów"""
        image_name = os.path.basename(self.image_paths[self.current_image_idx])
        shutil.copy(self.image_paths[self.current_image_idx], os.path.join(self.output_folder, image_name))
        shutil.move(self.image_paths[self.current_image_idx], os.path.join(self.deleted_folder, image_name))

if __name__ == "__main__":
    app = ImageLabelingApp()
