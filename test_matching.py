"""
얼굴 매칭 테스트
"""
from src.models.face_matcher import FaceMatcher

matcher = FaceMatcher()

# 같은 사람 사진으로 테스트
result = matcher.match_faces(
    "data/test_face.jpg",
    "data/test_face.jpg"  # 같은 이미지로 테스트
)

print(f"\n최종 결과: {result}")