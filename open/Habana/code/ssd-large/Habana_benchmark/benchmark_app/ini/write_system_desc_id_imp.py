import json 
import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description='Parsing Arguments')
    parser.add_argument('-id', '--input_data_types',  help='Model Directory', type=str, default='INT8')
    parser.add_argument('-re', '--retraining',  help='Model Name', type=str, default='No')
    parser.add_argument('-swf', '--starting_weights_filename',  help='starting_weights_filename', type=str)
    parser.add_argument('-wd', '--weight_data_types',  help='weight_data_types', type=str, default='INT8')
    parser.add_argument('-wt', '--weight_transformations',  help='weight_transformations', type=str, default='Asymmetric Quantization')
    parser.add_argument('-o', '--output',  help='output_json', type=str, required=True)

    return parser.parse_args()


def main(args):
    #with open('system_desc_id_imp.json') as json_file:
    #    data = json.load(json_file)
    data={}
    data["input_data_types"]  = args.input_data_types
    data["retraining"]  = args.retraining
    data["starting_weights_filename"]  = args.starting_weights_filename
    data["weight_data_types"]  = args.weight_data_types
    data["weight_transformations"]  = args.weight_transformations
    with open(args.output, 'w') as outfile:
        json.dump(data, outfile,sort_keys=True,indent=3)

if __name__ == "__main__":
    args=arg_parser()
    main(args)