"""
데이터 증강 테스트
"""
from src.preprocessing.augmentation import DataAugmentor

augmentor = DataAugmentor()

# processed 폴더의 정렬된 이미지들을 증강
augmentor.augment_batch(
    input_dir="data/processed",
    output_dir="data/augmented",
    num_variations=5
)

print("\n✅ data/augmented/ 폴더를 확인하세요!")
print("원본 1장 → 7장으로 증가 (원본 + 6가지 변형)")