import sys
sys.path = ['/home/lbh/workspace'] + sys.path
import face_alignment
from skimage import io
import time
import cProfile, pstats
import torch

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')


input = [f'test/assets/000000{i}.jpeg' for i in range(10,99)][:70]
input = [io.imread(e)  for e in input]
#input = input * 3 

prof = cProfile.Profile()
prof.enable()
fa.get_facebox_form_imagelist(input[:3])
print('init done')
s1 = time.time()
for i in range(50):
    preds1 = [fa.get_facebox_form_image(e) for e in input]

print('for loop done')
s2 = time.time()
print(f'for loop: {s2-s1}s')
for i in range(50):
    preds2 = fa.get_facebox_form_imagelist(input)
s3 = time.time()
print(f'batch: {s3-s2}')
import ipdb; ipdb.set_trace()

prof.disable()
p = pstats.Stats(prof, stream=sys.stdout)
p.sort_stats('cumulative').print_stats(100)
