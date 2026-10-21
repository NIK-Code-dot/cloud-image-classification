import os
import shutil
import random

SOURCE_DIR = "ml/data/raw/RealWaste"
OUTPUT_DIR = "ml/data/processed"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

classes = [
    d for d in os.listdir(SOURCE_DIR)
    if os.path.isdir(os.path.join(SOURCE_DIR, d))
]

for class_name in classes:
    source_class_dir = os.path.join(SOURCE_DIR, class_name)

    images = [
        f for f in os.listdir(source_class_dir)
        if os.path.isfile(os.path.join(source_class_dir, f))
    ]

    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for split_name, split_images in [
        ("train", train_images),
        ("val", val_images),
        ("test", test_images),
    ]:
        split_class_dir = os.path.join(
            OUTPUT_DIR,
            split_name,
            class_name
        )

        os.makedirs(split_class_dir, exist_ok=True)

        for image_name in split_images:
            source_path = os.path.join(source_class_dir, image_name)
            destination_path = os.path.join(split_class_dir, image_name)

            shutil.copy2(source_path, destination_path)

    print(
        f"{class_name}: "
        f"train={len(train_images)}, "
        f"val={len(val_images)}, "
        f"test={len(test_images)}"
    )

print("\nDataset split completed successfully.")
