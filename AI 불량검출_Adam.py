import os
import numpy as np
import random
from shutil import copyfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

# 경로 지정 변환 필요

# 데이터셋의 루트 디렉토리와 각 클래스 디렉토리를 지정합니다.
ROOT_DIR = './PW Imgae(AIR)_230317'
CLASS_DIRS = ['BackgroundNoise', 'LoopNoise', 'Normal(Air)']

# 각 클래스 디렉토리에 있는 모든 파일명을 읽어들입니다.
data = []
for class_dir in CLASS_DIRS:
    files = os.listdir(os.path.join(ROOT_DIR, class_dir))
    for file in files:
        data.append((os.path.join(ROOT_DIR, class_dir, file), class_dir))

# 데이터를 랜덤하게 train, validation, test로 분할합니다.
train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=42)

# 분할된 데이터셋을 저장할 디렉토리를 생성합니다.
for dataset in ['train', 'val', 'test']:
    for class_dir in CLASS_DIRS:
        os.makedirs(os.path.join(dataset, class_dir), exist_ok=True)

# 분할된 데이터셋을 디렉토리에 저장합니다.
for data, dataset in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
    for file, class_dir in data:
        filename = os.path.basename(file)
        dest_dir = os.path.join(dataset, class_dir)
        dest_file = os.path.join(dest_dir, filename)
        os.makedirs(dest_dir, exist_ok=True)
        os.replace(file, dest_file)

# 데이터셋 경로
data_dir = './Newnew data'

# 이미지 크기
IMG_SIZE = (224, 224)

# 배치 사이즈
BATCH_SIZE = 32

# 클래스 개수
NUM_CLASSES = 3

# 데이터셋 제너레이터
data_generator = ImageDataGenerator(rescale=1./255)

train_data = data_generator.flow_from_directory(directory=os.path.join(data_dir, 'train'),
                                                target_size=IMG_SIZE,
                                                batch_size=BATCH_SIZE,
                                                class_mode='categorical')

val_data = data_generator.flow_from_directory(directory=os.path.join(data_dir, 'val'),
                                              target_size=IMG_SIZE,
                                              batch_size=BATCH_SIZE,
                                              class_mode='categorical')

test_data = data_generator.flow_from_directory(directory=os.path.join(data_dir, 'test'),
                                               target_size=IMG_SIZE,
                                               batch_size=BATCH_SIZE,
                                               class_mode='categorical')
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')
])

NUM_EPOCHS = 10

# 학습 데이터셋 크기 계산
NUM_TRAIN_STEPS = train_data.n // BATCH_SIZE

# 검증 데이터셋 크기 계산
NUM_VAL_STEPS = val_data.n // BATCH_SIZE

optimizer = tf.keras.optimizers.Adam()
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
model.compile(optimizer=Adam(learning_rate=0.005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data,
                    epochs=NUM_EPOCHS,
                    steps_per_epoch=NUM_TRAIN_STEPS,
                    validation_data=val_data,
                    validation_steps=NUM_VAL_STEPS)
checkpoint.save('./ck3') # 모델 저장