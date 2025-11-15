"""
얼굴 감지 모듈 - insightface 사용
"""
import cv2
from insightface.app import FaceAnalysis

class FaceDetector:
    def __init__(self):
        """insightface 초기화"""
        print("🔧 FaceDetector 초기화 중...")
        self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ FaceDetector 준비 완료!")
    
    def detect_faces(self, image_path):
        """얼굴 감지"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ 이미지 없음: {image_path}")
            return []
        
        faces = self.app.get(img)
        print(f"✅ {len(faces)}개 얼굴 감지")
        return faces
    
    def draw_faces(self, image_path, output_path):
        """얼굴 박스 그리기"""
        img = cv2.imread(image_path)
        faces = self.detect_faces(image_path)
        
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        cv2.imwrite(output_path, img)
        print(f"💾 저장: {output_path}")