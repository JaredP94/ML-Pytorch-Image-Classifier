import utils

import json
import pandas as pd
import torch

# Main program function defined below
def main():
    in_args = utils.get_predict_input_args()

    with open(in_args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    device = torch.device("cuda:0" if (in_args.gpu and torch.cuda.is_available()) else "cpu")

    loaded_model, loaded_mappings = utils.load_checkpoint(in_args.checkpoint)

    probs, classes = utils.predict(in_args.dir, loaded_model, loaded_mappings, device, in_args.top_k)
    result = pd.DataFrame({'probability': probs}, index=[cat_to_name[class_] for class_ in classes])
    print(result)

if __name__ == "__main__":
    main()

# Demo: python predict.py flowers/test/1/image_06743.jpg checkpoint_squeezenet1_1.pth --gpu --top_k 10 --category_names cat_to_name.json