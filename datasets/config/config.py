import sys, os, argparse, yaml

def load_kitti_parameters(args,yaml_file_path):
    opt = yaml.load(open(yaml_file_path), Loader=yaml.FullLoader)
    #load the parameters from the yaml file
    for key, value in opt.items():
        setattr(args, key, value)
    args.rgb_path = args.dataset_path + args.seq +"/image_2"
    args.flow_path_noc = args.dataset_path + args.seq + "/flow_noc"
    args.flow_path_occ = args.dataset_path + args.seq + "/flow_occ"
    return args

def load_vkitti_parameters(args,yaml_file_path):
    opt = yaml.load(open(yaml_file_path), Loader=yaml.FullLoader)
    #load the parameters from the yaml file
    for key, value in opt.items():
        setattr(args, key, value)
    args.rgb_path = args.dataset_path + "vkitti_1.3.1_rgb/" + args.seq 
    args.flow_path_noc = args.dataset_path + "vkitti_1.3.1_flowgt/" + args.seq
    args.flow_path_occ = args.dataset_path + "vkitti_1.3.1_flowgt/" + args.seq
    return args

def data_analysis_parameters(dataset, yaml_file_path):
    parser = argparse.ArgumentParser()
    """
    add parameters here
    """
    parser.add_argument('-dataset_path', help="for ex. dataset/kitti/", type=str, required=False, default='datasets/kitti/')
    parser.add_argument('-seq', help="seq", type=str, required=False, default="training")
    parser.add_argument('-dataset', help="kitti/vkitti/sintel", type=str, required=False, default='kitti')
    parser.add_argument('-rgb_path', help="datasets/kitti/training/image_2", type=str, required=False, default='datasets/kitti/training/image_2')
    parser.add_argument('-flow_path_noc', help="datasets/kitti/training/flow_noc", type=str, required=False, default='datasets/kitti/training/flow_noc')
    parser.add_argument('-flow_path_occ', help="datasets/kitti/training/flow_occ", type=str, required=False, default='datasets/kitti/training/flow_occ')
    parser.add_argument('-height', help="height", type=int, required=False, default=256)
    parser.add_argument('-width', help="width", type=int, required=False, default=512)
    # # # training parameters
    # parser.add_argument('-epochs', help="num of epochs for train", type=int, required=False, default=100)
    # parser.add_argument('-lr', help="learning rate", type=float, required=False, default=0.00005)
    # parser.add_argument('-batch_size', help="batch size", type=int, required=False, default=64)
    args = parser.parse_args()
    
    # loading args from yaml file given as input
    if(dataset == "kitti"):
        args = load_kitti_parameters(args,yaml_file_path)
    
    elif(dataset == "vkitti"):
        args = load_vkitti_parameters(args,yaml_file_path)
         
    return args
    



# print("arguments: {}".format(str(args)))
# with open(args["config"], 'w') as file_descriptor:
#         yaml.safe_dump(args, file_descriptor)