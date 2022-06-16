import torch
import numpy as np
from utils.utils import AverageMeter, ProgressMeter



def simclr_train(train_loader, model, criterion, optimizer, epoch, train_args, second_criterion=None):
    """ 
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image']
        images_augmented = batch['image_augmented']
        b, c, h, w = images.size()
        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
        input_ = input_.view(-1, c, h, w) 
        input_ = input_.cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)

        output = model(input_).view(b, 2, -1)
        loss = criterion(output)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)

    return loss.item(), 0


def scan_train(train_loader, model, criterion, optimizer, epoch, train_args, second_criterion=None):
    """ 
    Train w/ SCAN-Loss
    """
    total_losses = AverageMeter('Total Loss', ':.4e')
    consistency_losses = AverageMeter('Consistency Loss', ':.4e')
    entropy_losses = AverageMeter('Entropy', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [total_losses, consistency_losses, entropy_losses],
        prefix="Epoch: [{}]".format(epoch))

    if train_args['update_cluster_head_only']:
        model.eval() # No need to update BN
    else:
        model.train() # Update BN

    model.to('cuda:'+str(train_args['gpu_id']))

    for i, batch in enumerate(train_loader):
        # Forward pass
        anchors = batch['anchor']
        if isinstance(anchors,list):
            anchors = [im.to('cuda:'+str(train_args['gpu_id']), non_blocking=True) for im in anchors]
        else: anchors = anchors.to('cuda:'+str(train_args['gpu_id']),non_blocking=True)

        neighbors = batch['neighbor']
        if isinstance(neighbors,list):
            neighbors = [im.to('cuda:'+str(train_args['gpu_id']), non_blocking=True) for im in neighbors]
        else: neighbors = neighbors.to('cuda:'+str(train_args['gpu_id']),non_blocking=True)


        if train_args['update_cluster_head_only']: # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_features = model(anchors, forward_pass='backbone')
                neighbors_features = model(neighbors, forward_pass='backbone')
            anchors_output = model(anchors_features, forward_pass='head')
            neighbors_output = model(neighbors_features, forward_pass='head')

        else: # Calculate gradient for backprop of complete network
            anchors_output = model(anchors)
            neighbors_output = model(neighbors)

        # Loss for every head
        total_loss, consistency_loss, entropy_loss = [], [], []
        for anchors_output_subhead, neighbors_output_subhead in zip(anchors_output, neighbors_output):
            total_loss_, consistency_loss_, entropy_loss_ = criterion(anchors_output_subhead,
                                                                         neighbors_output_subhead)
            total_loss.append(total_loss_)
            consistency_loss.append(consistency_loss_)
            entropy_loss.append(entropy_loss_)

        # Register the mean loss and backprop the total loss to cover all subheads
        total_losses.update(np.mean([v.item() for v in total_loss]))
        consistency_losses.update(np.mean([v.item() for v in consistency_loss]))
        entropy_losses.update(np.mean([v.item() for v in entropy_loss]))

        total_stack = torch.stack(total_loss, dim=0)
        best_head = total_stack.argmin(dim=0)
        total_loss = torch.sum(total_stack)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)

    return total_loss.item(), best_head


def selflabel_train(train_loader, model, criterion, optimizer, epoch, train_args, second_criterion=None):
    """ 
    Self-labeling based on confident samples
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [losses],
                                prefix="Epoch: [{}]".format(epoch))
    model.train()
    model.to('cuda:'+str(train_args['gpu_id']))


    for i, batch in enumerate(train_loader):
        images = batch['image']
        images.to('cuda:'+str(train_args['gpu_id']),non_blocking=True)
        images_augmented = batch['image_augmented']
        images_augmented.to('cuda:'+str(train_args['gpu_id']),non_blocking=True)


        with torch.no_grad(): 
            output = model(images)[0]
        output_augmented = model(images_augmented)[0]

        loss = criterion(output, output_augmented)
        losses.update(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if train_args['ema'] is not None: # Apply EMA to update the weights of the network
            train_args['ema'].update_params(model)
            train_args['ema'].apply_shadow(model)
        
        if i % 25 == 0:
            progress.display(i)

    return loss.item(), 0

def twist_training(train_loader, model, criterion, optimizer, epoch, train_args, second_criterion=None):

    device = train_args['device']+':'+str(train_args['gpu_id'])
    model.to(device)

    for i, return_object in enumerate(train_loader):
        if isinstance(return_object,dict):
            imgs = return_object['image']
        else:
            imgs = return_object[0]

        print('epoch: ',epoch,' / batch: ',i)
        optimizer.zero_grad()
        imgs = [im.to(device, non_blocking=True) for im in imgs]

        if train_args['aug'] != 'multicrop':
            train_args['local_crops_number'] = 0

        feat = model(imgs)
        all_feats = feat.chunk(2+train_args['local_crops_number'])
            #all_probs = [torch.nn.functional.softmax(f/args['tau'], dim=-1) for f in all_feats]
        n_views = len(all_feats)

        loss = 0
        all_loss = []
        for i1 in range(2):
            for i2 in range(i1+1, n_views):
                all_loss.append(criterion(all_feats[i1], all_feats[i2]))
        loss = sum([single_loss/len(all_loss) for single_loss in all_loss])

        loss.backward()
        optimizer.step()

    return loss.item(), 0

def double_training(train_loader, model, criterion, optimizer, epoch, train_args, second_criterion=None):

    if train_args['update_cluster_head_only']:
        model.eval() # No need to update BN
    else:
        model.train() # Update BN

    device = train_args['device']+':'+str(train_args['gpu_id'])
    model.to(device)

    current_loss = 0

    for i, batch in enumerate(train_loader):
        # Forward pass
        print('epoch: ',epoch,' / batch: ',i)
        anchors = batch['anchor']
        neighbors = batch['neighbor']
        anchors = [im.to(device, non_blocking=True) for im in anchors]
        neighbors = [im.to(device, non_blocking=True) for im in neighbors]
        #print('outputType: ',type(anchors))
        #print('outputLen: ',len(anchors))
        #print('ImageType: ',type(anchors[0]))
        #print('ImagesShape: ',anchors[0].shape) 

        if train_args['update_cluster_head_only']: # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_features = model(anchors, forward_pass='backbone')
                neighbors_features = model(neighbors, forward_pass='backbone')
            anchors_output = model(anchors_features, forward_pass='head')
            neighbors_output = model(neighbors_features, forward_pass='head')

        else: # Calculate gradient for backprop of complete network
            anchors_output = model(anchors)
            neighbors_output = model(neighbors)

        #print('modelOutput: ',anchors_output.shape)
        all_feats = anchors_output.chunk(2+train_args['local_crops_number'])
        #all_probs = [torch.nn.functional.softmax(f/args['tau'], dim=-1) for f in all_feats]
        n_views = len(all_feats)
        twist_loss = 0
        all_loss = []
        for i1 in range(2):
            for i2 in range(i1+1, n_views):
                all_loss.append(criterion(all_feats[i1], all_feats[i2]))

        twist_loss = sum([single_loss/len(all_loss) for single_loss in all_loss])


        # Loss for every head
        #total_loss =  []
        #for anchors_output_subhead, neighbors_output_subhead in zip(anchors_output, neighbors_output):
            #print('anchors_output_subhead: ',anchors_output_subhead.shape)
        scan_loss = second_criterion(anchors_output, neighbors_output)



        #total_loss = torch.sum(torch.stack(total_loss, dim=0))
        final_loss = 0.5*(twist_loss + scan_loss)
        print('LOSS\ntwist: ',twist_loss.item(),' scan: ',scan_loss.item(),' total: ',final_loss.item())
        current_loss = final_loss.item()

        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

    return current_loss.item(), 0


def multidouble_training(train_loader, model, criterion, optimizer, epoch, train_args, second_criterion=None):

    if train_args['update_cluster_head_only']:
        model.eval() # No need to update BN
    else:
        model.train() # Update BN

    device = train_args['device']+':'+str(train_args['gpu_id'])
    model.to(device)

    current_loss = 0

    for i, batch in enumerate(train_loader):
        # Forward pass
        print('epoch: ',epoch,' / batch: ',i)
        anchors = batch['anchor']
        neighbors = batch['neighbor']
        anchors = [im.to(device, non_blocking=True) for im in anchors]
        neighbors = [im.to(device, non_blocking=True) for im in neighbors]
        #print('outputType: ',type(anchors))
        #print('outputLen: ',len(anchors))
        #print('ImageType: ',type(anchors[0]))
        #print('ImagesShape: ',anchors[0].shape) 

        if train_args['update_cluster_head_only']: # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_features = model(anchors, forward_pass='backbone')
                neighbors_features = model(neighbors, forward_pass='backbone')
            anchors_output = model(anchors_features, forward_pass='head')
            neighbors_output = model(neighbors_features, forward_pass='head')

        else: # Calculate gradient for backprop of complete network
            anchors_output = model(anchors)
            neighbors_output = model(neighbors)

        # Loss for every head
        scan_loss, twist_loss = [], []
        for anchors_output_subhead, neighbors_output_subhead in zip(anchors_output, neighbors_output):
            scan_loss_ = second_criterion(anchors_output_subhead,neighbors_output_subhead)
            scan_loss.append(scan_loss_)

        #print('modelOutput: ',anchors_output.shape)   

            all_feats = anchors_output_subhead.chunk(2+train_args['local_crops_number'])
            #all_probs = [torch.nn.functional.softmax(f/args['tau'], dim=-1) for f in all_feats]
            n_views = len(all_feats)
            #twist_loss_ = 0
            all_loss = []
            for i1 in range(2):
                for i2 in range(i1+1, n_views):
                    all_loss.append(criterion(all_feats[i1], all_feats[i2]))

            twist_loss_ = sum([single_loss/len(all_loss) for single_loss in all_loss])
            twist_loss.append(twist_loss_)

        
        ttemp = torch.stack(twist_loss, dim=0)
        best_head = ttemp.argmin(dim=0)
       

# Loss for every head
        #total_loss =  []
        #for anchors_output_subhead, neighbors_output_subhead in zip(anchors_output, neighbors_output):
        #print('anchors_output_subhead: ',anchors_output_subhead.shape)
        twist_all  = torch.sum(ttemp)
        scan_all = torch.sum(torch.stack(scan_loss, dim=0))

        #total_loss.append(total_loss_)


        #total_loss = torch.sum(torch.stack(total_loss, dim=0))
        final_loss = 0.5*(twist_all + scan_all)
        current_loss = final_loss

        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

    return current_loss.item(), best_head

def multihead_twist_train(train_loader, model, criterion, optimizer, epoch, train_args, second_criterion=None):


    device = 'cuda:'+str(train_args['gpu_id'])
    model.to(device)
    if train_args['update_cluster_head_only']:
        model.eval() # No need to update BN
    else:
        model.train() # Update BN


    for i, return_object in enumerate(train_loader):
        if isinstance(return_object,dict):
            imgs = return_object['image']
        else:
            imgs = return_object[0]
        
        print('epoch: ',epoch,' / batch: ',i)
        optimizer.zero_grad()
        imgs = [im.to(device, non_blocking=True) for im in imgs]

        if train_args['aug'] != 'multicrop':
            train_args['local_crops_number'] = 0

        if train_args['update_cluster_head_only']: # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                x = model(imgs,forward_pass='backbone')

            features = model(x,forward_pass='head')

        else:
            features = model(imgs)

        twist_heads = []

        for feat in features:
            all_feats = feat.chunk(2+train_args['local_crops_number'])
                #all_probs = [torch.nn.functional.softmax(f/args['tau'], dim=-1) for f in all_feats]
            n_views = len(all_feats)

            #loss = 0
            all_loss = []
            for i1 in range(2):
                for i2 in range(i1+1, n_views):
                    all_loss.append(criterion(all_feats[i1], all_feats[i2]))
            loss = sum([single_loss/len(all_loss) for single_loss in all_loss])

            twist_heads.append(loss)
        
        ttemp = torch.stack(twist_heads, dim=0)
        best_head = ttemp.argmin(dim=0)
        twist_all  = torch.sum(ttemp)

        twist_all.backward()
        optimizer.step()

    return twist_all.item(), best_head


def pseudolabel_train(train_loader, model, criterion, optimizer, epoch, train_args, second_criterion=None):

    device = 'cuda:'+str(train_args['gpu_id'])

    model.train()
    model = model.to(device) # OK(%-cexp_00)
    #print('MODEL AFTER CRITICAL EXPRESSION 2: ',type(model))

    for i, batch in enumerate(train_loader):
        images = batch['image']
        targets = batch['target']

        images = images.to(device) # OK(%-cexp_00)
        print('images(type): ', type(images))
        print('images.shape: ',images.shape)


        features = model(images)
        print('features(type): ', type(features))
        print('features.shape: ',features.shape)
        print('feature[0]',str(feature[0]))
        print('targets(type): ', type(targets))
        print('targets.shape: ',targets.shape)
        print('targets[0]: ',str(targets[0]))

        loss = criterion(features, targets)
        print('epoch: ',epoch,' / batch: ',i)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #criterion.to(device)
