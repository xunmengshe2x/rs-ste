import os
import pickle

data = {
    "image1_paths": [],
    "image2_paths": [],
    "image1_rec": [],
    "image2_rec":[],
}

def save_data(image_dir, target_txt, output_file="data/annotation/inference_annotations.pkl"):
    with open(target_txt, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        source_img_name, target_text = line.strip().split()
        data["image1_paths"].append(os.path.join(image_dir, source_img_name))
        data["image2_rec"].append(target_text)

    with open(output_file, "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    image_dir = "example_data"
    target_txt = "example_data/i_t.txt"
    save_data(image_dir, target_txt)
    print("Completed annotation preparation.")