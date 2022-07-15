args.gpus
args.restore_ckpt
args.stage


aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}

train_dataset = Carla_Dataset(aug_params, split='training', seq= ["MidRainyNoon"], setup_type = ['camera_0']) # , 'camera_1','camera_2','camera_3', 'camera_4'


args.batch_size


args.lr
args.wdecay
args.epsilon
optimizer =  ADAMW


