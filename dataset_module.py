import os
import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

class SRDataAugmentation:
    def __init__(self, p=1):
        """
        데이터 증강 클래스
        :param p: 증강 적용 확률
        """
        self.p = p

    def __call__(self, lr_img, hr_img):
        if random.random() < self.p:
            # 랜덤 수평 뒤집기
            if random.random() < 0.5:
                lr_img = TF.hflip(lr_img)
                hr_img = TF.hflip(hr_img)

            # 랜덤 수직 뒤집기
            if random.random() < 0.5:
                lr_img = TF.vflip(lr_img)
                hr_img = TF.vflip(hr_img)

            # 랜덤 회전
            angle = random.choice([0, 90, 180, 270])
            lr_img = TF.rotate(lr_img, angle)
            hr_img = TF.rotate(hr_img, angle)

            # 색상 조정
            lr_img = TF.adjust_brightness(lr_img, random.uniform(0.8, 1.2))
            lr_img = TF.adjust_contrast(lr_img, random.uniform(0.8, 1.2))
            lr_img = TF.adjust_saturation(lr_img, random.uniform(0.8, 1.2))

        return lr_img, hr_img


class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None, augment=True, aug_params=None):
        """
        SRDataset 클래스
        :param lr_dir: LR 이미지 디렉토리 경로
        :param hr_dir: HR 이미지 디렉토리 경로
        :param transform: 이미지 변환을 위한 transform
        :param augment: 데이터 증강 여부
        :param aug_params: 증강 파라미터 (SRDataAugmentation 초기화 파라미터)
        """
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform
        self.augment = augment

        # 디렉토리 내 이미지 파일만 필터링
        self.image_files = [
            f for f in os.listdir(lr_dir)
            if os.path.isfile(os.path.join(lr_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        # 증강 클래스 초기화
        if augment:
            aug_params = aug_params or {}
            self.aug_transform = SRDataAugmentation(**aug_params)

    def __len__(self):
        return len(self.image_files) * (2 if self.augment else 1)

    def __getitem__(self, idx):
        try:
            # 원본 데이터 인덱스 계산
            original_idx = idx // 2 if self.augment else idx

            # 파일 경로 생성
            lr_path = os.path.join(self.lr_dir, self.image_files[original_idx])
            hr_path = os.path.join(self.hr_dir, self.image_files[original_idx])

            # 이미지 로드
            lr_image = Image.open(lr_path).convert('RGB')
            hr_image = Image.open(hr_path).convert('RGB')

            # 증강 적용 (짝수 인덱스일 때만)
            if self.augment and idx % 2 == 1:
                lr_image, hr_image = self.aug_transform(lr_image, hr_image)

            # 변환 적용
            if self.transform:
                lr_image = self.transform(lr_image)
                hr_image = self.transform(hr_image)

            return lr_image, hr_image
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            raise e