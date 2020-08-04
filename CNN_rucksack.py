%%Define your own dataset:
path = Path('data')
!mkdir data
!mv rucksack.txt data/rucksack.txt
!mv handbag.txt data/handbag.txt

folder_1 = 'rucksack'
file_1 = 'rucksack.txt'
dest_1 = path/folder_1
dest_1.mkdir(parents = True, exist_ok = True)

# help(dest_1.mkdir)
folder_2 = 'handbag'
file_2 = 'handbag.txt'
dest_2 = path/folder_2
dest_2.mkdir(parents = True, exist_ok = True)

classes = ['rucksack', 'handbag']

download_images(path/file_1, dest_1, max_pics=200)
download_images(path/file_2, dest_2, max_pics=200)

for c in classes:
  print(c)
  verify_images(path/c, delete=True, max_size=500)

np.random.seed(42)
data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.2, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)

learn = cnn_learner(data, models.resnet34, metrics=accuracy)

learn.fit_one_cycle(1, 2)

