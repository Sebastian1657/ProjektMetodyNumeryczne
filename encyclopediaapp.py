import os
import glob
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

class EncyclopediaViewer:
    def __init__(self, master, base_folder):
        self.master = master
        master.title("Prosta Encyklopedia")
        self.base_folder = base_folder
        self.categories = self.load_categories()
        self.current_image = None
        self.current_image_index = 0
        self.image_files = [] # Inicjalizacja pustej listy

        # Widgety interfejsu
        self.category_label = ttk.Label(master, text="Wybierz kategorię:")
        self.category_label.pack()

        self.category_combobox = ttk.Combobox(master, values=list(self.categories.keys()))
        self.category_combobox.pack()
        self.category_combobox.bind("<<ComboboxSelected>>", self.show_images)

        self.image_canvas = tk.Canvas(master)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)

        # Przyciski nawigacyjne
        self.prev_button = ttk.Button(master, text="Poprzedni", command=self.show_previous_image)
        self.prev_button.pack(side=tk.LEFT)
        self.next_button = ttk.Button(master, text="Następny", command=self.show_next_image)
        self.next_button.pack(side=tk.RIGHT)

    def load_categories(self):
        categories = {}
        for folder_name in os.listdir(self.base_folder):
            folder_path = os.path.join(self.base_folder, folder_name)
            if os.path.isdir(folder_path):
                categories[folder_name] = folder_path
        return categories

    def show_images(self, event=None):
            selected_category = self.category_combobox.get()
            if not selected_category:
                return

            category_path = self.categories[selected_category]
            self.image_files = glob.glob(os.path.join(category_path, "*.jpg")) + glob.glob(os.path.join(category_path, "*.png"))
            if not self.image_files:
                self.image_canvas.delete("all")
                self.image_canvas.create_text(self.image_canvas.winfo_width()/2, self.image_canvas.winfo_height()/2, text="Brak obrazów w tej kategorii.", font=("Arial", 16), fill="gray")
                return

            self.current_image_index = 0  # Resetujemy indeks przy zmianie kategorii
            self.show_current_image()

    def show_current_image(self):
        if not self.image_files:
            return

        image_path = self.image_files[self.current_image_index]
        self.image_canvas.delete("all")
        try:
            image = Image.open(image_path)
            image.thumbnail((700, 700))
            self.current_image = ImageTk.PhotoImage(image)
            self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image)
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można otworzyć pliku: {image_path}\n{e}")

    def show_next_image(self):
        if self.image_files:
            self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
            self.show_current_image()

    def show_previous_image(self):
        if self.image_files:
            self.current_image_index = (self.current_image_index - 1) % len(self.image_files)
            self.show_current_image()

def main():
    root = tk.Tk()
    base_folder = "encyclopedia"  # Nazwa folderu nadrzędnego
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
        messagebox.showinfo("Informacja", f"Utworzono folder 'encyklopedia'. Umieść w nim foldery z kategoriami i obrazami.")
        return

    app = EncyclopediaViewer(root, base_folder)
    root.mainloop()

if __name__ == "__main__":
    main()