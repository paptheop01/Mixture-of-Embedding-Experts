import pickle

with open('/mnt/disks/disk-1/train_loader_ant.pkl', 'rb') as file:
    loader =pickle.load(file)
    dataloader_iter = iter(loader)
    first_batch = next(dataloader_iter)
    first_batch = next(dataloader_iter)
    print(first_batch['feats'])
   