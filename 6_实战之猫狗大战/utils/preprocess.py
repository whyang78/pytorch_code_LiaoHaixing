import os
import shutil
import random

random.seed(78)

def get_dataset(file_path,output_path,train_ratio=0.9):
    cat_images=[]
    dog_images=[]
    for image in os.listdir(file_path):
        if image[:3]=='cat':
            cat_images.append(image)
        elif image[:3]=='dog':
            dog_images.append(image)

    random.shuffle(cat_images)
    random.shuffle(dog_images)

    for i in range(len(cat_images)):
        origin_path=os.path.join(file_path,cat_images[i])
        if i<int(len(cat_images)*train_ratio):
            pic_path=os.path.join(output_path,'train','cat',cat_images[i])
        else:
            pic_path=os.path.join(output_path,'val','cat',cat_images[i])
        shutil.move(origin_path,pic_path)

    for i in range(len(dog_images)):
        origin_path=os.path.join(file_path,dog_images[i])
        if i<int(len(dog_images)*train_ratio):
            pic_path=os.path.join(output_path,'train','dog',dog_images[i])
        else:
            pic_path=os.path.join(output_path,'val','dog',dog_images[i])
        shutil.move(origin_path,pic_path)

if __name__ == '__main__':
    file_path='../../dataset/cat_vs_dog/train'
    output_path='../../dataset/cat_vs_dog/data'
    get_dataset(file_path,0.9,output_path)
