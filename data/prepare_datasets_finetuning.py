import os
import random
import shutil

synth_dataset = '/synth-colon'
real_dataset = '/polyp-data/TestDataset'
dataset_names = [x for x in os.listdir(real_dataset) if x[0] != '.']
train_size_synth = 20000
val_size_synth = 20

for target_name in dataset_names:
    target_name = 'finetune_' + target_name

    os.makedirs('/cut/datasets/{}/trainA'.format(target_name), exist_ok=True)
    os.makedirs('/cut/datasets/{}/valA'.format(target_name), exist_ok=True)
    os.makedirs('/cut/datasets/{}/valA_seg'.format(target_name), exist_ok=True)
    os.makedirs('/cut/datasets/{}/trainA_seg'.format(target_name), exist_ok=True)
    os.makedirs('/cut/datasets/{}/trainB'.format(target_name), exist_ok=True)
    os.makedirs('/cut/datasets/{}/trainB_seg'.format(target_name), exist_ok=True)
    os.makedirs('/cut/datasets/{}/valB'.format(target_name), exist_ok=True)
    os.makedirs('/cut/datasets/{}/valB_seg'.format(target_name), exist_ok=True)

    synth_images = os.listdir(os.path.join(synth_dataset, 'images'))
    synth_images = synth_images[:train_size_synth+val_size_synth]
    random.shuffle(synth_images)

    trainA_images = random.sample(synth_images, len(synth_images)-val_size_synth)
    valA_images = [x for x in synth_images if x not in trainA_images]

    real_images = os.listdir(os.path.join(real_dataset, target_name.split('_')[1], 'images'))
    random.shuffle(real_images)

    trainB_images = real_images
    valB_images = real_images


    for f in trainA_images:
        shutil.copy(
            os.path.join(synth_dataset, 'images', f),
            os.path.join('/cut/datasets',target_name,'trainA')
        )

    for f in trainA_images:
        shutil.copy(
            os.path.join(synth_dataset, 'masks', f),
            os.path.join('/cut/datasets',target_name,'trainA_seg')
        )

    for f in valA_images:
        shutil.copy(
            os.path.join(synth_dataset, 'images', f),
            os.path.join('/cut/datasets',target_name,'valA')
        )

    for f in valA_images:
        shutil.copy(
            os.path.join(synth_dataset, 'masks', f),
            os.path.join('/cut/datasets',target_name,'valA_seg')
        )


    for f in trainB_images:
        shutil.copy(
            os.path.join(real_dataset, target_name.split('_')[1], 'images', f),
            os.path.join('/cut/datasets',target_name,'trainB')
        )

    for f in trainB_images:
        shutil.copy(
            os.path.join(real_dataset, target_name.split('_')[1],'masks', f),
            os.path.join('/cut/datasets',target_name,'trainB_seg')
        )

    for f in valB_images:
        shutil.copy(
            os.path.join(real_dataset, target_name.split('_')[1],'images', f),
            os.path.join('/cut/datasets',target_name,'valB')
        )

    for f in valB_images:
        shutil.copy(
            os.path.join(real_dataset,target_name.split('_')[1], 'masks', f),
            os.path.join('/cut/datasets',target_name,'valB_seg')
        )