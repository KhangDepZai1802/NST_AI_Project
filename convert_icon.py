# convert_icon.py
from PIL import Image
img = Image.open("assets/logoNST.png").convert("RGBA")
img.save("assets/logoNST.ico", format="ICO", sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])
print("Done → assets/logoNST.ico")