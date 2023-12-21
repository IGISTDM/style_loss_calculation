import json
from constants import loss_list, style_list, json_output_folder
from path import path_join

method_list = ["AdaIN", "IGISTDM", "RAASN"]
content_amount = 20
style_amount = 100
required_amount = content_amount * style_amount

for method in method_list:
    print(f"{method}:")
    for style in style_list:
        for loss in loss_list:
            file_name = f"{style}-{loss}.json"
            print(f"    {file_name}")
            json_file = path_join(json_output_folder, method, file_name)
            with open(json_file, 'r') as json_file:
                data = json.load(json_file)
            clip_loss_amount = 0
            for clip_loss in data:
                clip_loss_amount = clip_loss_amount + 1
            if clip_loss_amount != required_amount:
                raise Exception(
                    f"Data amount in {style}-{loss}.json is below {required_amount}")

print("JSON amount is correct!")
