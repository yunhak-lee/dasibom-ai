"""
데이터 증강 모듈
학습 데이터 확장을 위한 이미지 변형
"""
import cv2
import numpy as np
import os
from pathlib import Path

class DataAugmentor:
    def __init__(self):
        """데이터 증강기 초기화"""
        print("🔧 DataAugmentor 초기화 완료!")
    
    def flip_horizontal(self, image):
        """좌우 반전"""
        return cv2.flip(image, 1)
    
    def adjust_brightness(self, image, factor=1.2):
        """
        밝기 조절
        factor > 1: 밝게, factor < 1: 어둡게
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def rotate_image(self, image, angle=5):
        """
        이미지 회전
        angle: 회전 각도 (양수: 반시계, 음수: 시계방향)
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 회전 행렬
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 회전 적용
        rotated = cv2.warpAffine(image, M, (w, h), 
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
        return rotated
    
    def add_noise(self, image, noise_level=10):
        """
        가우시안 노이즈 추가
        noise_level: 노이즈 강도 (0-50)
        """
        noise = np.random.normal(0, noise_level, image.shape)
        noisy_image = image.astype(np.float32) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image
    
    def augment_single(self, image_path, output_dir, num_variations=5):
        """
        단일 이미지 증강
        
        Args:
            image_path (str): 입력 이미지 경로
            output_dir (str): 출력 폴더
            num_variations (int): 생성할 변형 개수
        
        Returns:
            list: 생성된 파일 경로 리스트
        """
        # 이미지 읽기
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 이미지 없음: {image_path}")
            return []
        
        # 출력 폴더 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 원본 파일명
        filename = Path(image_path).stem
        ext = Path(image_path).suffix
        
        output_paths = []
        
        # 원본 저장
        original_path = os.path.join(output_dir, f"{filename}_original{ext}")
        cv2.imwrite(original_path, image)
        output_paths.append(original_path)
        print(f"✅ 원본 저장")
        
        # 1. 좌우 반전
        flipped = self.flip_horizontal(image)
        flip_path = os.path.join(output_dir, f"{filename}_flip{ext}")
        cv2.imwrite(flip_path, flipped)
        output_paths.append(flip_path)
        print(f"✅ 좌우 반전")
        
        # 2. 밝게
        bright = self.adjust_brightness(image, factor=1.2)
        bright_path = os.path.join(output_dir, f"{filename}_bright{ext}")
        cv2.imwrite(bright_path, bright)
        output_paths.append(bright_path)
        print(f"✅ 밝게 조정")
        
        # 3. 어둡게
        dark = self.adjust_brightness(image, factor=0.8)
        dark_path = os.path.join(output_dir, f"{filename}_dark{ext}")
        cv2.imwrite(dark_path, dark)
        output_paths.append(dark_path)
        print(f"✅ 어둡게 조정")
        
        # 4. 회전 (+5도)
        rotated_p = self.rotate_image(image, angle=5)
        rotate_p_path = os.path.join(output_dir, f"{filename}_rotate_p5{ext}")
        cv2.imwrite(rotate_p_path, rotated_p)
        output_paths.append(rotate_p_path)
        print(f"✅ +5도 회전")
        
        # 5. 회전 (-5도)
        rotated_n = self.rotate_image(image, angle=-5)
        rotate_n_path = os.path.join(output_dir, f"{filename}_rotate_n5{ext}")
        cv2.imwrite(rotate_n_path, rotated_n)
        output_paths.append(rotate_n_path)
        print(f"✅ -5도 회전")
        
        # 6. 노이즈
        noisy = self.add_noise(image, noise_level=10)
        noise_path = os.path.join(output_dir, f"{filename}_noise{ext}")
        cv2.imwrite(noise_path, noisy)
        output_paths.append(noise_path)
        print(f"✅ 노이즈 추가")
        
        return output_paths
    
    def augment_batch(self, input_dir, output_dir, num_variations=5):
        """
        폴더 내 모든 이미지 증강
        
        Args:
            input_dir (str): 입력 폴더
            output_dir (str): 출력 폴더
            num_variations (int): 이미지당 변형 개수
        """
        # 이미지 파일 찾기
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"\n📂 {len(image_files)}개 이미지 증강 시작...")
        print(f"각 이미지당 {num_variations+1}개 생성 (원본 포함)")
        
        total_generated = 0
        
        for i, filename in enumerate(image_files, 1):
            input_path = os.path.join(input_dir, filename)
            
            print(f"\n[{i}/{len(image_files)}] {filename}")
            
            outputs = self.augment_single(input_path, output_dir, num_variations)
            total_generated += len(outputs)
        
        print(f"\n✅ 완료! 총 {total_generated}개 이미지 생성")
        print(f"📁 저장 위치: {output_dir}")

if __name__ == "__main__":
    print("=" * 50)
    print("데이터 증강 테스트")
    print("=" * 50)
    
    augmentor = DataAugmentor()