"""
얼굴 감지 테스트
"""
from src.preprocessing.face_detection import FaceDetector

# 얼굴 감지기 생성
detector = FaceDetector()

# 테스트 이미지 경로
input_image = "data/test_face.jpg"
output_image = "data/result_face.jpg"

# 얼굴 감지 실행
print("\n🔍 얼굴 감지 시작...")
faces = detector.detect_faces(input_image)

# 결과 출력
print(f"\n📊 감지 결과: {len(faces)}개의 얼굴")

# 박스 그려서 저장
detector.draw_faces(input_image, output_image)

print("\n✅ 완료! result_face.jpg 파일을 확인하세요!")