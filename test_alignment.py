"""
얼굴 정렬 테스트
"""
from src.preprocessing.face_alignment import FaceAligner

aligner = FaceAligner(output_size=512)

# 단일 이미지 테스트
aligner.align_face(
    "data/test_face.jpg",
    "data/aligned_face.jpg"
)

print("\n✅ data/aligned_face.jpg 파일을 확인하세요!")