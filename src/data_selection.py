""" DATA SELECTION """
from imports import *
warnings.simplefilter(action = 'ignore', category = FutureWarning)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print("Num GPUs available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Organize data into train, valid, test dirs
os.chdir('../data/dogs-vs-cats')

if os.path.isdir('train/dog') is False:
    os.makedirs('train/dog')
    os.makedirs('train/cat')

    os.makedirs('valid/dog')
    os.makedirs('valid/cat')

    os.makedirs('test/dog')
    os.makedirs('test/cat')

    # 1000 images for train
    for i in random.sample(glob.glob('cat*'), 500):
        shutil.move(i, 'train/cat')
    for i in random.sample(glob.glob('dog*'), 500):
        shutil.move(i, 'train/dog')

    # 200 images for validation
    for i in random.sample(glob.glob('cat*'), 100):
        shutil.move(i, 'valid/cat')
    for i in random.sample(glob.glob('dog*'), 100):
        shutil.move(i, 'valid/dog')

    # 100 images for test
    for i in random.sample(glob.glob('cat*'), 50):
        shutil.move(i, 'test/cat')
    for i in random.sample(glob.glob('dog*'), 50):
        shutil.move(i, 'test/dog')

os.chdir('../../')

