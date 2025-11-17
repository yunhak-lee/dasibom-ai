"""
Dataset 테스트
"""
from src.utils.dataset import FaceDataset
from torchvision import transforms

# Transform 정의
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 데이터셋 로드
dataset = FaceDataset("data/augmented", transform=transform)

print(f"\n📊 데이터셋 정보:")
print(f"  - 총 이미지 수: {len(dataset)}")

# 첫 번째 샘플
sample = dataset[0]
print(f"\n✅ 첫 번째 샘플:")
print(f"  - 파일명: {sample['filename']}")
print(f"  - 이미지 shape: {sample['image'].shape}")
print(f"  - 이미지 타입: {type(sample['image'])}")

# DataLoader 테스트
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f"\n📦 DataLoader 테스트:")
for batch in dataloader:
    print(f"  - 배치 크기: {batch['image'].shape}")
    print(f"  - 파일명: {batch['filename']}")
    break

print("\n✅ Dataset 준비 완료!")