import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from django.shortcuts import render
from django.http import HttpResponse
import datetime
import numpy as np
from PIL import Image, UnidentifiedImageError
import imagehash
from django.core.files.storage import default_storage
from django.conf import settings
from tensorflow.keras.preprocessing import image
from scipy.stats import entropy
import pickle

# === ABSOLUTE MODEL PATH ===
model_path = r"fruit_cnn_model.h5"
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    model = None  # Will be trained if not found

IMG_SIZE = 100

# Relative paths based on MEDIA_ROOT
train_dir = r"media\Fruit-Images-Dataset-master\Fruit-Images-Dataset-master\Training"
test_dir = r"media\Fruit-Images-Dataset-master\Fruit-Images-Dataset-master\Test"
class_labels = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])


def training(request):
    try:
        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            return HttpResponse("Error: Training or Test directories not found.")

        NUM_CLASSES = len([name for name in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, name))])
        BATCH_SIZE = 32

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )

        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )

        val_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=True
        )

        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )

        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(
            train_generator,
            epochs=10,
            validation_data=val_generator,
            steps_per_epoch=500,
            validation_steps=val_generator.samples // BATCH_SIZE
        )

        test_loss, test_accuracy = model.evaluate(test_generator)

        # Save the model to absolute path
        model.save(model_path)

        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        plot_filename = f'training_plot_{timestamp}.png'
        plot_path = os.path.join(settings.MEDIA_ROOT, plot_filename)

        plt.figure()
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(plot_path)
        plt.close()

        return render(request, 'users/training.html', {
            'test_accuracy': f"{test_accuracy*100:.2f}%",
            'plot_path': f"/media/{plot_filename}"
        })

    except Exception as e:
        return HttpResponse(f"Training failed with error: {str(e)}")

def prediction(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        file_name = uploaded_file.name
        file_path = os.path.join('uploads', file_name)
        full_file_path = default_storage.save(file_path, uploaded_file)
        full_path = os.path.join(settings.MEDIA_ROOT, full_file_path)

        try:
            # Validate uploaded file as an image
            img = Image.open(full_path)
            img.close()

            # Preprocess image
            img = image.load_img(full_path, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction (direct label without entropy threshold)
            prediction_probs = model.predict(img_array)[0]
            predicted_index = np.argmax(prediction_probs)
            predicted_label = class_labels[predicted_index]
            context['prediction'] = predicted_label

            context['image_url'] = os.path.join(settings.MEDIA_URL, full_file_path)

        except UnidentifiedImageError:
            context['message'] = "The uploaded file is not a valid image. Please upload a JPG or PNG."
        except Exception as e:
            context['message'] = f"An error occurred: {str(e)}"

    return render(request, 'users/prediction.html', context)

from pyexpat.errors import messages
from django.shortcuts import render

from users.forms import UserRegistrationForm
from users.models import UserRegistrationModel

# Create your views here.

def base(request):
    return render(request,'base.html')

from django.shortcuts import render
from django.contrib import messages
from .forms import UserRegistrationForm

def base(request):
    return render(request, 'base.html')  # Or 'users/home.html' if using a homepage template

from django.shortcuts import render
from django.contrib import messages
from .forms import UserRegistrationForm

def base(request):
    return render(request, 'base.html')

def index(request):
    return render(request,'index.html')

def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'You have been successfully registered!')
            return render(request, 'UserRegistration.html', {'form': UserRegistrationForm()})
        else:
            if form.errors.get('emailid') or form.errors.get('mobileno'):
                messages.error(request, 'Email or Mobile Number already exists.')
            else:
                messages.error(request, 'Please correct the errors below.')
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistration.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')  # Corrected to 'loginid'
        pswd = request.POST.get('pswd')        # Corrected to 'pswd'
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account is not activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})