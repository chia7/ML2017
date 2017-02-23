import sys
from PIL import Image

im0 = Image.open(sys.argv[1])
im1 = Image.open(sys.argv[2])

w, h = im0.size

im_out = Image.new( "RGBA", (w, h), (0, 0, 0, 0) )

for i in range(w):
	for j in range(h):
		if im0.getpixel((i, j)) != im1.getpixel((i, j)):
			im_out.putpixel( (i, j), im1.getpixel((i, j)))

im_out.save("ans_two.png")