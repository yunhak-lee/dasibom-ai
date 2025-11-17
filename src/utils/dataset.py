"""
PyTorch Dataset 클래스
얼굴 이미지 학습용 데이터셋
"""
import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class FaceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): 이미지 폴더 경로
            transform: 이미지 변환 (torchvision.transforms)
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # 이미지 파일 목록
        self.image_files = [
            f for f in os.listdir(data_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        print(f"✅ 데이터셋 로드 완료: {len(self.image_files)}개 이미지")
    
    def __len__(self):
        """데이터셋 크기"""
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        인덱스에 해당하는 이미지 로드
        
        Returns:
            dict: {"image": tensor, "filename": str}
        """
        # 이미지 경로
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        
        # 이미지 읽기
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # PIL Image로 변환 (transform 적용 위해)
        from PIL import Image
        image = Image.fromarray(image)
        
        # Transform 적용
        if self.transform:
            image = self.transform(image)
        
        return {
            "image": image,
            "filename": img_name
        }

class AgeDataset(Dataset):
    """
    나이 레이블이 있는 데이터셋
    파일명 형식: person1_age_25.jpg
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # 이미지 파일 및 나이 추출
        self.samples = []
        
        for filename in os.listdir(data_dir):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            # 파일명에서 나이 추출 시도
            try:
                # 예: "person1_age_25_bright.jpg" → 25
                if '_age_' in filename:
                    age_str = filename.split('_age_')[1].split('_')[0].split('.')[0]
                    age = int(age_str)
                    
                    self.samples.append({
                        'filename': filename,
                        'age': age
                    })
            except:
                # 나이 추출 실패 시 건너뛰기
                pass
        
        print(f"✅ 나이 데이터셋 로드: {len(self.samples)}개")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 이미지 로드
        img_path = os.path.join(self.data_dir, sample['filename'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # PIL로 변환
        from PIL import Image
        image = Image.fromarray(image)
        
        # Transform 적용
        if self.transform:
            image = self.transform(image)
        
        return {
            "image": image,
            "age": sample['age'],
            "filename": sample['filename']
        }

if __name__ == "__main__":
    print("=" * 50)
    print("Dataset 테스트")
    print("=" * 50)
    
    # 기본 데이터셋
    dataset = FaceDataset("data/augmented")
    print(f"총 샘플 수: {len(dataset)}")
    
    # 첫 번째 샘플 확인
    sample = dataset[0]
    print(f"샘플: {sample['filename']}")