from PIL import Image

def resize(path):
    image = Image.open(path)    
    resize = min(image.size)
    result = image.resize((resize, resize))
    result.save(path)

resize("dog.jpg")