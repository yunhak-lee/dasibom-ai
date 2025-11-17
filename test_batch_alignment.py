"""
배치 얼굴 정렬 테스트
"""
from src.preprocessing.face_alignment import FaceAligner

aligner = FaceAligner(output_size=512)

# 폴더 내 모든 이미지 일괄 정렬
aligner.batch_align(
    input_dir="data/raw",
    output_dir="data/processed"
)

print("\n✅ data/processed/ 폴더를 확인하세요!")