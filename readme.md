SpeakerCounter!


Установка среды (для win, anaconda):

conda install virtualenv
conda install virtualenvwrapper-win

virtualenv <имя> –  создание среды в нужной папке (перейти до этого)
<имя>\scripts\activate – активация среды 
deactivate – деактивация среды

pip3 install --upgrage <lib> – установка нужных библиотек
или
pip3 install -r requirements.txt


Драйвера для обучения на видеокарте:

Tensorflow устанавливается для последней версии драйверов: CUDA Toolkit 11.2
Дополнительно скачать последнюю версию CUDNN (нужна регистрация), добавить все файлы в соответствующие папки драйверов
Путь: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2


Запуск модели (мой вариант):

cd PythonCodeProjects
env\env\scripts\activate
cd SpeakerCounter (cd CountNet-my)

python train_model.py – запуск обучения
python predict.py - предсказания для модели
