import PIL
import os
import PIL.Image
from tkinter.filedialog import *
from pillow_heif import register_heif_opener

register_heif_opener()

CONST_HEIGHT = 32
CONST_WIDTH = 32

compressed_file_count = 0

directory_path = "./images/raw"
compressed_path = "./images/compressed"
file_base_extension = "HEIC"
compressed_file_extension = "jpg"
dir_path_len = len(directory_path) + 1
for (dirpath, dirnames, filenames) in os.walk(directory_path):
    folder_name = dirpath[dir_path_len:]
    index = 0
    for filename in filenames:
        index += 1
        img_path = f"{folder_name}/{folder_name}_{index}"
        # Check if file does not match naming conventions
        if(not filename.startswith(folder_name)):
            os.rename(f"{dirpath}/{filename}", f"{dirpath}/{folder_name}_{index}.{file_base_extension}")
        # Check if file has not been compressed yet
        if(not os.path.exists(f"{compressed_path}/{img_path}.{compressed_file_extension}")):
            compressed_file_count += 1
            print(f"{compressed_path}/{folder_name}/{folder_name}_{index}.{compressed_file_extension} DNE!")
            img = PIL.Image.open(f"{dirpath}/{folder_name}_{index}.{file_base_extension}")
            img = img.resize((CONST_HEIGHT, CONST_WIDTH), PIL.Image.Resampling.LANCZOS)
            img = img.convert("RGB")
            if(not os.path.exists(f"{compressed_path}/{folder_name}")):
                os.makedirs(f"{compressed_path}/{folder_name}")
            img.save(f"{compressed_path}/{img_path}.{compressed_file_extension}")
            print(f"Compressed {folder_name}_{index}!")

if(compressed_file_count == 0):
    print("Program finished. No files compressed.")
else:
    print(f"Program finished. Compressed {compressed_file_count} files.")

exit(0)
