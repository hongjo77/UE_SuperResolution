import torch
import json
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from model_class_model_cnn6 import *
import os
from dataset_module import SRDataset, SRDataAugmentation

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"


def load_config_and_data(config_path, data_loaders_path):
    """
    설정 및 DataLoader를 로드합니다.
    Args:
        config_path (str): JSON 형식의 설정 파일 경로.
        data_loaders_path (str): DataLoader가 저장된 .pth 파일 경로.

    Returns:
        dict: 설정 정보.
        dict: DataLoader 정보.
    """
    # JSON 설정 로드
    with open(config_path, "r") as f:
        config = json.load(f)

    # DataLoader 로드
    data_loaders = torch.load(data_loaders_path)

    return config, data_loaders


if __name__ == "__main__":
    # 설정 파일 및 DataLoader 파일 경로
    config_path = "config.json"
    data_loaders_path = "data_loaders.pth"

    # 설정 및 DataLoader 로드
    config, data_loaders = load_config_and_data(config_path, data_loaders_path)

    # DataLoader 추출
    train_loader = data_loaders["train_loader"]
    val_loader = data_loaders["val_loader"]

    # 설정 정보 추출
    model_save_path = config["model_save_path"]
    learning_rate = config["learning_rate"]
    max_epochs = config["max_epochs"]
    devices = config["devices"]

    # 모델 생성
    model = CustomSRModel()
    lightning_model = LightningCustomSRModel(
        model, learning_rate=learning_rate)

    csv_logger = CSVLogger(save_dir="lightning_logs", name="super_resolution")

    # Trainer 설정
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=devices,
        strategy=DDPStrategy(find_unused_parameters=True),
        logger=csv_logger,
    )

    # 모델 훈련
    trainer.fit(lightning_model, train_loader, val_loader)

    # 모델 저장
    torch.save(lightning_model.state_dict(), model_save_path)
    print(f"모델이 '{model_save_path}'에 저장되었습니다.")
