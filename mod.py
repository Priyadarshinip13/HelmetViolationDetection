import os

name_to_id = {
    "driver_with_helmet": 0,
    "passenger_with_helemt": 0,

    "driver_without_helmet": 1,
    "passenger_without_helemt": 1,

    "driver": 2,
    "passenger": 2,

    "number_plate": 3
}

label_dirs = [
    "C:/Users/PC-1/Desktop/project/data/train/labels",
    "C:/Users/PC-1/Desktop/project/data/test/labels",
    "C:/Users/PC-1/Desktop/project/data/vaid/labels"
]

for label_dir in label_dirs:
    for file in os.listdir(label_dir):
        if not file.endswith(".txt"):
            continue

        path = os.path.join(label_dir, file)

        with open(path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            class_name = parts[0]

            if class_name in name_to_id:
                parts[0] = str(name_to_id[class_name])
                new_lines.append(" ".join(parts))

        with open(path, "w") as f:
            f.write("\n".join(new_lines))