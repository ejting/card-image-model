import os

# Verify the file exists
file_path = 'C:\\Users\\eting\\AppData\\Local\\Programs\\Python\\Python39\\Lib\\site-packages\\tensorflow_decision_forests\\tensorflow\\ops\\inference\\inference.so'
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")