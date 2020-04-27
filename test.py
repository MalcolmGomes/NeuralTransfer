import time
from neural_transfer import *

# style_path = input("Enter path to style image: ")
# content_path = input("Enter path to content image: ")
style_path = "starry.jpg"
content_path = "fox.jpg"
resize_img(style_path)
resize_img(content_path)
print("Loading images", style_path, 'and', content_path)
style_img = image_loader(style_path)
content_img = image_loader(content_path)
assert style_img.size() == content_img.size()

input_img = content_img.clone()
# input_img = torch.randn(content_img.data.size(), device=device) # For white noise
print("Performing style transfer of", style_path, "on", content_path)
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img)
print("Style Transfer Complete! Images being displayed.")
# plt.figure(), imshow(style_img, title='Style Image')
# plt.figure(), imshow(content_img, title='Content Image')
plt.figure(), imshow(output, title='Output Image'), plt.ioff(), plt.show()
input("Press any key to exit.")