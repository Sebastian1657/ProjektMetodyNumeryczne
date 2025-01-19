import os

def renumber_files(folder_path):
    # Pobierz listę plików w folderze
    files = os.listdir(folder_path)
    
    # Filtruj tylko te pliki, które pasują do schematu "IMG_fokaXXXX" z odpowiednimi rozszerzeniami
    matching_files = sorted([f for f in files if f.startswith("IMG_foka") and f.endswith((".txt", ".png", ".jpg", ".jpeg"))])
    
    # Nowa numeracja
    for new_index, old_file in enumerate(matching_files):
        # Rozpoznanie rozszerzenia pliku
        extension = os.path.splitext(old_file)[1]
        
        # Generowanie nowej nazwy pliku
        new_name = f"IMG_foka{new_index:04d}{extension}"
        
        # Stare i nowe ścieżki
        old_path = os.path.join(folder_path, old_file)
        new_path = os.path.join(folder_path, new_name)
        
        # Zmień nazwę pliku
        os.rename(old_path, new_path)
        print(f"Renamed: {old_file} -> {new_name}")

# Ścieżka do folderu z plikami
folder_path = "raw_data/labels/"  # Zmień na swoją ścieżkę

# Wywołanie funkcji
renumber_files(folder_path)
