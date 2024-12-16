import torch 
import torch.utils
import torch.utils.tensorboard
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def create_writer(experiment_name:str,
                 model_name:str,
                 extra:str=None)->torch.utils.tensorboard.writer.SummaryWriter:
    from datetime import datetime
    import os
    timestamp=datetime.now().strftime("%Y-%m-%d")
    if extra:
        log_dir=os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir=os.path.join("runs", timestamp, experiment_name, model_name)
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return torch.utils.tensorboard.writer.SummaryWriter(log_dir=log_dir)

def train_step(
        model:torch.nn.Module,
        dataloader:torch.utils.data.DataLoader,
        loss_fn:torch.nn.Module,
        optimizer:torch.optim.Optimizer,
        device:torch.device,
        regularization_type:str="None",
        lambda_reg:float=0.0
) -> Tuple[float, float]:
    model.train()
    train_loss, train_acc=0,0
    for batch,(X, y) in enumerate (dataloader):
        X,y=X.to(device), y.to(device)
        y_pred=model(X)
        loss=loss_fn(y_pred, y)
        if regularization_type=="L1":
            l1_norm=sum(p.abs().sum() for p in model.parameters())
            loss+=lambda_reg*l1_norm
        elif regularization_type=="L2":
            l2_norm=sum(p.pow(2).sum() for p in model.parameters())
            loss+=lambda_reg*l2_norm
        train_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class=torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc+=(y_pred_class==y).sum().item()/len(y_pred)
    train_loss=train_loss/len(dataloader)
    train_acc=train_acc/len(dataloader)
    return train_loss, train_acc

def val_step(model:torch.nn.Module,
              dataloader:torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device:torch.device)->Tuple[float, float]:
    model.eval()
    val_loss, val_acc= 0, 0
    with torch.no_grad():
        for batch, (X,y) in enumerate(dataloader):
            X, y= X.to(device), y.to(device)
            val_pred_logits=model(X)
            loss = loss_fn(val_pred_logits, y)
            val_loss += loss.item()
            test_pred_labels=torch.argmax(val_pred_logits, dim=1)
            val_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    return val_loss, val_acc

def test_step(model:torch.nn.Module,
              dataloader:torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device:torch.device)->Tuple[float, float]:
    model.eval()
    test_loss, test_acc= 0, 0
    with torch.inference_mode():
        for batch, (X,y) in enumerate(dataloader):
            X, y= X.to(device), y.to(device)
            test_pred_logits=model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_pred_labels=torch.argmax(test_pred_logits, dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(
        model:torch.nn.Module,
        train_dataloader:torch.utils.data.DataLoader,
        val_dataloader:torch.utils.data.DataLoader,
        test_dataloader:torch.utils.data.DataLoader,
        optimizer:torch.optim.Optimizer,
        loss_fn:torch.nn.Module,
        epochs:int,
        writer:torch.utils.tensorboard.writer.SummaryWriter=None,
        device:torch.device=torch.device('cpu'),
        regularization_type:str="None",
        lambda_reg:float=0.0
        
)->Dict[str, List]: 
    results={
        "train_loss":[],
        "train_acc":[],
        "val_loss":[],
        "val_acc":[],
        "test_loss":[],
        "test_acc":[]
    }
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc=train_step(model=model,
                                         dataloader=train_dataloader,
                                         loss_fn=loss_fn,
                                         optimizer=optimizer,
                                         device=device,
                                         regularization_type=regularization_type,
                                         lambda_reg=lambda_reg)
        val_loss, val_acc=val_step(model=model,
                                      dataloader=val_dataloader,
                                      loss_fn=loss_fn,
                                      device=device)
        test_loss, test_acc=test_step(model=model,
                                      dataloader=test_dataloader,
                                      loss_fn=loss_fn,
                                      device=device)
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"val_loss: {val_loss:.4f} | "
          f"val_acc: {val_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        if writer:
            writer.add_scalars(main_tag="Loss",
                               tag_scalar_dict={
                                   "train_loss":train_loss,
                                   "val_loss":val_loss,
                                   "test_loss":test_loss
                               },
                               global_step=epoch)
    
            writer.add_scalars(main_tag="Accuracy",
                               tag_scalar_dict={
                                   "train_acc":train_acc,
                                   "val_acc":val_acc,
                                   "test_acc":test_acc
                               },
                               global_step=epoch)
            writer.close()
        else:
            pass

    return results