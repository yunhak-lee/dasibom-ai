"""
얼굴 정렬 모듈
insightface 랜드마크를 사용한 얼굴 정렬 및 크롭
"""
import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceAligner:
    def __init__(self, output_size=512):
        """
        Args:
            output_size (int): 출력 이미지 크기 (기본 512x512)
        """
        print("🔧 FaceAligner 초기화 중...")
        self.output_size = output_size
        self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ FaceAligner 준비 완료!")
    
    def align_face(self, image_path, output_path=None):
        """
        얼굴 정렬 및 크롭
        
        Args:
            image_path (str): 입력 이미지 경로
            output_path (str): 출력 이미지 경로 (None이면 반환만)
        
        Returns:
            numpy.ndarray: 정렬된 얼굴 이미지 (512x512)
        """
        # 이미지 읽기
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ 이미지 없음: {image_path}")
            return None
        
        # 얼굴 검출
        faces = self.app.get(img)
        
        if len(faces) == 0:
            print("❌ 얼굴이 감지되지 않았습니다")
            return None
        
        # 첫 번째 얼굴 선택
        face = faces[0]
        
        # 바운딩 박스 추출
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # 얼굴 영역 확장 (20% 여유)
        margin = 0.2
        w = x2 - x1
        h = y2 - y1
        x1 = max(0, int(x1 - w * margin))
        y1 = max(0, int(y1 - h * margin))
        x2 = min(img.shape[1], int(x2 + w * margin))
        y2 = min(img.shape[0], int(y2 + h * margin))
        
        # 얼굴 크롭
        face_img = img[y1:y2, x1:x2]
        
        # 정사각형으로 패딩
        face_h, face_w = face_img.shape[:2]
        if face_h > face_w:
            padding = (face_h - face_w) // 2
            face_img = cv2.copyMakeBorder(
                face_img, 0, 0, padding, padding,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        elif face_w > face_h:
            padding = (face_w - face_h) // 2
            face_img = cv2.copyMakeBorder(
                face_img, padding, padding, 0, 0,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        
        # 512x512 리사이즈
        aligned_face = cv2.resize(face_img, (self.output_size, self.output_size))
        
        print(f"✅ 얼굴 정렬 완료: {self.output_size}x{self.output_size}")
        
        # 저장
        if output_path:
            cv2.imwrite(output_path, aligned_face)
            print(f"💾 저장: {output_path}")
        
        return aligned_face
    
    def batch_align(self, input_dir, output_dir):
        """
        폴더 내 모든 이미지 일괄 정렬
        
        Args:
            input_dir (str): 입력 폴더
            output_dir (str): 출력 폴더
        """
        import os
        
        # 출력 폴더 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 이미지 파일 찾기
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"\n📂 {len(image_files)}개 이미지 처리 시작...")
        
        success_count = 0
        for i, filename in enumerate(image_files, 1):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"aligned_{filename}")
            
            print(f"\n[{i}/{len(image_files)}] {filename}")
            
            result = self.align_face(input_path, output_path)
            if result is not None:
                success_count += 1
        
        print(f"\n✅ 완료! {success_count}/{len(image_files)} 성공")

if __name__ == "__main__":
    print("=" * 50)
    print("얼굴 정렬 테스트")
    print("=" * 50)
    
    aligner = FaceAligner(output_size=512)