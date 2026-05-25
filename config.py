EMOTIONS = {
    'neutral': 0, 'happy': 1, 'sad': 2,
    'angry': 3, 'fearful': 4, 'disgust': 5, 'surprised': 6
}
LABEL_NAMES = list(EMOTIONS.keys())

SAMPLE_RATE = 22050
DURATION    = 3.0
N_MELS      = 128
MAX_LEN     = 128
INPUT_SHAPE = (128, 128, 1)
NUM_CLASSES = len(EMOTIONS)

DATA_PATHS = {
    'ravdess': 'data/ravdess',
    'crema':   'data/crema',
}