import os

label_dir = "data/labels"
max_id = -1

for file in os.listdir(label_dir):
    if file.endswith(".txt"):
        with open(os.path.join(label_dir, file)) as f:
            for line in f:
                if line.strip():
                    cid = int(line.split()[0])
                    if cid > max_id:
                        max_id = cid

print("Maximum class ID found:", max_id)
print("Recommended nc value:", max_id + 1)
