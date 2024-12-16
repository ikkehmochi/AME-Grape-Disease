import torch
import torchvision
import utils, data_setup, engine, ensembleModel
import os
from torchvision.transforms import v2 

DATASET_TYPE="Original"
MODEL="EfficientNet_B7"
SCHEME="FineTune"
NUM_EPOCHS = 10
BATCH_SIZE = 16
OUTPUT_SHAPE=4
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset_path=os.path.join(os.getcwd(), "data")

train_dir=os.path.join(dataset_path, DATASET_TYPE, "split", "train")
val_dir=os.path.join(dataset_path, DATASET_TYPE, "split", "val")
test_dir=os.path.join(dataset_path, DATASET_TYPE, "split", "test")

if DATASET_TYPE!="Augmented":
    data_transform=v2.Compose([ v2.Resize(size=(256, 256)),
                                v2.ToImage(), 
                                v2.ToDtype(torch.float32, scale=True),
                                    v2.Normalize(mean=[0.4685, 0.5424, 0.4491], std=[0.2337, 0.2420,0.2531]),
                                ])
else:
    data_transform=v2.Compose([ v2.Resize(size=(256, 256)),
                                v2.ToImage(), 
                                v2.ToDtype(torch.float32, scale=True),
                                v2.Normalize(mean=[0.4683, 0.5414, 0.4477], std=[0.2327, 0.2407,0.2521]),
                                ])
if __name__=="__main__":
    models_dir=os.path.join(os.getcwd(), "models","EndtoEnd")
    model_name=r"EfficientNet_B7_Original_EndtoEnd.pth"
    model_path=os.path.join(models_dir, model_name)
    regularizations=["L1", "L2"]
    lambda_reg=[0, 1e-4, 5e-4, 1e-3]
    for regularization in regularizations:
        for reg in lambda_reg:
            model=torchvision.models.efficientnet_b7().to(DEVICE)
            model.classifier=torch.nn.Sequential(torch.nn.Dropout(p=0.5, inplace=True), torch.nn.Linear(in_features=2560, out_features=OUTPUT_SHAPE, bias=True).to(DEVICE))
            model.load_state_dict(torch.load(model_path, weights_only=True))
            for param in model.features.parameters():
                param.requires_grad=False
            model.classifier.requires_grad=True
            train_dataloader, val_dataloader,test_dataloader, class_names=data_setup.create_dataloaders(train_dir=train_dir,
                                                                                                        val_dir=val_dir,
                                                                                                        test_dir=test_dir,
                                                                                                        transform=data_transform,
                                                                                                        batch_size=BATCH_SIZE,
                                                                                                        num_workers=os.cpu_count())
            loss_fn=torch.nn.CrossEntropyLoss()
            optimizer=torch.optim.SGD(params=model.parameters(), lr=3e-3, momentum=0.9)
            engine.train(model=model,
                            train_dataloader=train_dataloader,
                            val_dataloader=val_dataloader,
                            test_dataloader=test_dataloader,
                            loss_fn=loss_fn,
                            optimizer=optimizer,
                            epochs=NUM_EPOCHS,
                            writer=engine.create_writer(experiment_name=MODEL,
                                                        model_name=DATASET_TYPE,
                                                        extra=f"{SCHEME}_{regularization}_{reg}_{model_name.strip(".pth")}"),
                            device=DEVICE,
                            regularization_type=regularization,
                            lambda_reg=reg)
            utils.save_model(model=model,
                                target_dir=f"Models/{SCHEME}",
                                model_name=f"{MODEL}_{DATASET_TYPE}_{SCHEME}_{regularization}_{reg}_{model_name.strip(".pth")}.pth")
            del model