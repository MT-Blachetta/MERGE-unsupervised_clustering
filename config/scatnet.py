[{ 'train_method': 'scan', 'loss_type': 'scan_loss' ,'save_data': True, 'run_id': 0  },
 { 'train_method': 'multitwist', 'loss_type': 'twist_loss', 'setup':'multitwist','augmentation_type':'twist', 'augmentation_strategy':'barlow' ,'model_args.batch_norm': True, 'save_data': True, 'run_id': 1  },
 { 'train_method': 'multidouble', 'loss_type': 'double_loss' , 'setup':'double','augmentation_type':'twist', 'augmentation_strategy':'barlow','model_args.batch_norm': True, 'save_data': True, 'run_id': 2  } ]