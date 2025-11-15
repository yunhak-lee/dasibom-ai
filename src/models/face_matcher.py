"""
얼굴 매칭 모듈
ArcFace 임베딩을 사용한 얼굴 유사도 계산
"""
import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceMatcher:
    def __init__(self):
        """ArcFace 모델 초기화"""
        print("🔧 FaceMatcher 초기화 중...")
        self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ FaceMatcher 준비 완료!")
    
    def extract_embedding(self, image_path):
        """
        얼굴에서 특징 벡터(임베딩) 추출
        
        Args:
            image_path (str): 이미지 경로
        
        Returns:
            numpy.ndarray: 512차원 특징 벡터
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ 이미지 없음: {image_path}")
            return None
        
        faces = self.app.get(img)
        
        if len(faces) == 0:
            print("❌ 얼굴이 감지되지 않았습니다")
            return None
        
        # 첫 번째 얼굴의 임베딩 반환
        embedding = faces[0].embedding
        print(f"✅ 임베딩 추출 완료 (차원: {len(embedding)})")
        return embedding
    
    def calculate_similarity(self, embedding1, embedding2):
        """
        두 얼굴 임베딩의 코사인 유사도 계산
        
        Args:
            embedding1, embedding2: 얼굴 특징 벡터
        
        Returns:
            float: 유사도 (0~1, 높을수록 유사)
        """
        # 코사인 유사도 계산
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        return float(similarity)
    
    def match_faces(self, image1_path, image2_path):
        """
        두 이미지의 얼굴 유사도 계산
        
        Returns:
            dict: {"similarity": float, "is_match": bool}
        """
        print("\n🔍 얼굴 매칭 시작...")
        
        # 임베딩 추출
        emb1 = self.extract_embedding(image1_path)
        emb2 = self.extract_embedding(image2_path)
        
        if emb1 is None or emb2 is None:
            return {"similarity": 0.0, "is_match": False}
        
        # 유사도 계산
        similarity = self.calculate_similarity(emb1, emb2)
        
        # 임계값 0.6 이상이면 동일인으로 판단
        is_match = similarity > 0.6
        
        print(f"\n📊 유사도: {similarity:.4f}")
        print(f"{'✅ 동일인입니다!' if is_match else '❌ 다른 사람입니다.'}")
        
        return {
            "similarity": similarity,
            "is_match": is_match
        }

if __name__ == "__main__":
    print("=" * 50)
    print("얼굴 매칭 테스트")
    print("=" * 50)
    
    matcher = FaceMatcher()