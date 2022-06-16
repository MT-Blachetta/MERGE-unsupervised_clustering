from functionality import collate_custom
import torch


def INFO(mode,arguments):
    
    if mode == '00_main_command_list':
        print('\n')
        print(mode)
        print('arguments.gpu = ',arguments.gpu)
        print('arguments.prefix = ',arguments.prefix)
        print('arguments.config = ',arguments.config)


def TEST_initial_model(model,dataset,transformation):

    print('-----------------START_TEST:initial_model------------------')

    dataset.transform = transformation

    model.eval()
    model = model.cuda()

    batch_loader = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=256, pin_memory=True, collate_fn=collate_custom, drop_last=False, shuffle=True)

    batch = next(iter(batch_loader))

    print('type of batch: ',type(batch))
    if isinstance(batch, dict):
        images = batch['image']
        target = batch['target']
    else:
        images = batch[0]
        target = batch[1]

    images = images.cuda()

    print('images_type: ',type(images))
    print('images_shape: ',images.shape)
    print('images[0]_type: ',type(images[0]))

    print('dataset-target_type',type(target))
    print('dataset-target_shape: ',target.shape)
    print('dataset-target[0]_type: ',type(target[0]))
    print('dataset-target_max',target.max())
    

    features = model(images,forward_pass='features')
    print('features_shape: ', features.shape)
    print('features[0].shape: ',features[0].shape)
    preds = model(features,forward_pass='head')
    print('preds.shape: ',preds.shape)
    print('preds[0]: ',preds[0])
    truck = preds[:,9]
    print('truck_class max: ',truck.max())

    max_confidence, prediction = torch.max(preds,dim=1)
    prediction = prediction.type(torch.LongTensor)
    print('prediction.type: ',type(prediction))
    print('prediction.shape: ',prediction.shape)
    print('prediction_max: ',prediction.max())

    loss_fn = torch.nn.CrossEntropyLoss()
    loss_test = loss_fn(preds,prediction)
    print(loss_test)

    print('-------------END_TEST---------------')

    #type(batch)
    #(batch.shape)



    return
