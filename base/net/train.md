1.多GPU并行训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = multiscale_resnet(4)
model = torch.nn.DataParallel(model)
model.to(device)
CUDA_VISIBLE_DEVICES=0,1,3  python train.py


2.恢复训练
if opt.resume:
    model.eval()
    print('resuming finetune from %s' % opt.resume)
    try:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(opt.resume))
    except KeyError:
        model.load_state_dict(torch.load(opt.resume))
        model = torch.nn.DataParallel(model)
    else:
        model = torch.nn.DataParallel(model)
    model = model.cuda()