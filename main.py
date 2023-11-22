import torch 
import torch.nn as nn
import argparse
from utils import *
import torch.nn.functional as F
from models import *
from tqdm import tqdm


torch.set_default_dtype(torch.float32)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")


parser.add_argument(
    "-wz",
    "--window_size",
    default=25
)


parser.add_argument(
    "-gd",
    "--get_data",
    default=False
)


parser.add_argument(
    "-sp",
    "--save_path",
    default="./results/"
)


parser.add_argument(
    "-ls",
    "--loss",
    default="crossentropy"
)


parser.add_argument(
    "-mn",
    "--modelname", 
    default='GRU', 
    type=str,
    help=['Transformer', 'RNN', 'LSTM', 'GRU']
)


parser.add_argument(
    "-nc",
    "--num_class", 
    default=3
)


parser.add_argument(
    "-bs",
    "--batch_size",
    default=1,
    type=int
)


parser.add_argument(
    "-e",
    "--epochs", 
    default=20
)


def train(args, model, loss_fn, loader, optimizer, epoch):
    model.train()
    y_pred = []
    y_true = []
    losses = 0
    loop = tqdm(loader)
    model = model.to(DEVICE)
    
    for i, (data, _, _, label) in enumerate(loop):
        data = data.to(DEVICE).to(torch.float32)
        output = model(data)
        optimizer.zero_grad()
        mask = (label != -1)
        label = label[mask].to(DEVICE)
        
        loss = loss_fn(output, label)
        y_pred.append(F.softmax(output, dim=1).detach().cpu().numpy())
        y_true.extend(label.detach().cpu().numpy())
        losses += loss.item()
        loss.backward()
        optimizer.step()
        
    y_pred = np.concatenate(y_pred, axis=0)
    metrics = evaluate(args, y_pred, y_true)
    metrics['loss'] = losses / (i+1)
    
    progress = ProgressMeter(
        args,
        mode='train',
        meters=metrics,
        prefix= f'Epoch [{epoch}] --->  '
    )
    
    progress.display()



def validate(args, model, loss_fn, loader):
    model.eval()
    y_pred = []
    y_true = []
    losses = 0
    loop = tqdm(loader)
    with torch.no_grad():
        for i, (data, _, _, label)in enumerate(loop):
                data = data.to(DEVICE).to(torch.float32)
                output = model(data)
                mask = (label != -1)
                label = label[mask].to(DEVICE)
                output = output.squeeze(0)
                
                loss = loss_fn(output, label)
                y_pred.append(F.softmax(output, dim=1).detach().cpu().numpy())
                y_true.extend(label.detach().cpu().numpy())
                losses += loss.item()


    y_pred = np.concatenate(y_pred, axis=0)
    metrics = evaluate(args, y_pred, y_true)
    metrics['loss'] = losses / (i+1)

    progress = ProgressMeter(
        args,
        mode='test',
        meters=metrics,
        prefix= f'Testing --->  '
    )
    
    progress.display()
    return metrics['loss']



def main(args):
    if args.get_data:
        retrieve_data(args, output_path='./data')
        calculate_residual_label(data_path='./data')
        
    train_loader, val_loader = load_data(args, './data')
    for data, _ , _, _ in train_loader:
        input_size = data.size()[3]
        break
    
    hidden_size = 32
    num_stacked_layers = 3
    
    # Prepared model
    model = prepared_model(args, input_size, hidden_size, num_stacked_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience = 2, threshold=1e-4, verbose=2)
    loss_fn = nn.CrossEntropyLoss()
    
    best_loss = 100000
    for epoch in range(args.epochs):
        train(args, model, loss_fn, train_loader, optimizer, epoch)
        torch.cuda.empty_cache()
        val_loss = validate(args, model, loss_fn, val_loader)
        scheduler.step(val_loss)
        best_loss = save_model(val_loss, best_loss, model, args)



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
