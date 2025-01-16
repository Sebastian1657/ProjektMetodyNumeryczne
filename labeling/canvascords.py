import os
import glob
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import random

class ImageViewer:
    def __init__(self, master, image_folder):
        self.master = master
        master.title("Przeglądarka Obrazów")
        self.image_folder = image_folder
        self.image_files = sorted(glob.glob(os.path.join(self.image_folder, "*.jpg")))
        self.txt_files = sorted(glob.glob(os.path.join(self.image_folder, "*.txt")))
        self.current_index = 0

        self.txt_map = {}
        for txt_file in self.txt_files:
            base_name = os.path.splitext(os.path.basename(txt_file))[0]
            self.txt_map[base_name] = txt_file

        self.canvas = tk.Canvas(master)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.prev_button = ttk.Button(master, text="Poprzedni", command=self.show_previous)
        self.prev_button.pack(side=tk.LEFT)
        self.next_button = ttk.Button(master, text="Następny", command=self.show_next)
        self.next_button.pack(side=tk.RIGHT)

        self.show_image()

    def show_image(self):
        if not self.image_files:
            self.canvas.delete("all")
            self.canvas.create_text(self.canvas.winfo_width()/2, self.canvas.winfo_height()/2, text="Brak obrazów w folderze.", font=("Arial", 16), fill="gray")
            return

        image_path = self.image_files[self.current_index]
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        try:
            image = Image.open(image_path)
            self.canvas.delete("all")
            self.tk_image = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

            if base_name in self.txt_map:
                txt_path = self.txt_map[base_name]
                try:
                    with open(txt_path, 'r') as file:
                        for line in file:
                            data = line.strip().split()
                            try:
                                category = int(data[0])
                                center_x = float(data[1]) * image.width
                                center_y = float(data[2]) * image.height
                                width = float(data[3]) * image.width
                                height = float(data[4]) * image.height
                                x_min = center_x - width / 2
                                y_min = center_y - height / 2

                                random.seed(category)
                                r = random.randint(0, 255)
                                g = random.randint(0, 255)
                                b = random.randint(0, 255)
                                hex_color = '#%02x%02x%02x' % (r, g, b)

                                self.canvas.create_rectangle(x_min, y_min, x_min + width, y_min + height, outline=hex_color, width=2)
                            except (IndexError, ValueError) as e:
                                print(f"Błąd odczytu danych w wierszu pliku TXT: {e} w pliku {txt_path}. Sprawdź format pliku.")

                except FileNotFoundError:
                    print(f"Brak pliku TXT dla obrazu: {image_path}. Pomijam adnotacje.")
            
        except FileNotFoundError:
            print(f"Nie znaleziono pliku: {image_path}")
            return
        except Exception as e:
            print(f"Wystąpił błąd: {e}")

    def show_next(self):
        if self.image_files:
            self.current_index = (self.current_index + 1) % len(self.image_files)
            self.show_image()

    def show_previous(self):
        if self.image_files:
            self.current_index = (self.current_index - 1) % len(self.image_files)
            self.show_image()

def main():
    root = tk.Tk()
    output_folder = "labeling\output"
    viewer = ImageViewer(root, output_folder)
    root.mainloop()

if __name__ == "__main__":
    main()