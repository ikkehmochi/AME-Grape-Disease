import os
import random
import shutil

VAL_SIZE = 0.2
TEST_SIZE = 0.2
RANDOM_SEED = 42

if __name__ == "__main__":
    random.seed(RANDOM_SEED)
versions = os.listdir("data")
for version in versions:
    path = os.path.join("data", version)
    if os.path.isdir(os.path.join(path, "split")) is True:
        shutil.rmtree(os.path.join(path, "split"))
    if os.path.isdir(os.path.join(path, "split", "val")) is True:
        shutil.rmtree(os.path.join(path, "split", "val"))
    if os.path.isdir(os.path.join(path, "split", "test")) is True:
        shutil.rmtree(os.path.join(path, "split", "test"))
    if os.path.isdir(os.path.join(path, "split", "train")) is True:
        shutil.rmtree(os.path.join(path, "split", "train"))
    if os.path.isdir(os.path.join(path, "split")) is True:
        shutil.rmtree(os.path.join(path, "split"))
    os.mkdir(os.path.join(path, "split"))
    os.mkdir(os.path.join(path, "split", "val"))
    os.mkdir(os.path.join(path, "split", "test"))
    os.mkdir(os.path.join(path, "split", "train"))
    all_class = os.listdir(path=path)
    print(all_class)
    for x in all_class:
        if x != "split":
            images_path = os.path.join(path, x)
            class_images = os.listdir(images_path)
            val_images = random.sample(class_images, int(len(class_images) * VAL_SIZE))
            class_images = [image for image in class_images if image not in val_images]
            test_images = random.sample(
                class_images, int(len(class_images) * TEST_SIZE)
            )
            train_images = [image for image in class_images if image not in test_images]
            os.mkdir(os.path.join(path, "split", "val", x))
            for file in val_images:
                source = os.path.join(path, x, file)
                destination = os.path.join(path, "split", "val", x, file)
                shutil.copy(src=source, dst=destination)
            os.mkdir(os.path.join(path, "split", "test", x))
            for file in test_images:
                source = os.path.join(path, x, file)
                destination = os.path.join(path, "split", "test", x, file)
                shutil.copy(src=source, dst=destination)
            os.mkdir(os.path.join(path, "split", "train", x))
            print(x)
            for file in train_images:
                source = os.path.join(path, x, file)
                print(source)
                destination = os.path.join(path, "split", "train", x, file)
                shutil.copy(src=source, dst=destination)
