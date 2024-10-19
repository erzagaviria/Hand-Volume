import cv2
import mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Inisialisasi MediaPipe Hands
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils

# Class Deteksi Tangan
class HandDetection:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.hands = mpHands.Hands(max_num_hands=max_num_hands, 
                                   min_detection_confidence=min_detection_confidence,
                                   min_tracking_confidence=min_tracking_confidence)

    def findHandLandMarks(self, image, handNumber=0, draw=False):
        originalImage = image.copy()
        # Ubah ke RGB sesuai requirement mediapipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)
        landMarkList = []

        # Jika tangan terdeteksi
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[handNumber]
            for id, landMark in enumerate(hand.landmark):
                imgH, imgW, imgC = originalImage.shape
                xPos, yPos = int(landMark.x * imgW), int(landMark.y * imgH)
                landMarkList.append([id, xPos, yPos])
            # Gambar landmark dan koneksi tangan jika 'draw=True'
            if draw:
                mpDraw.draw_landmarks(originalImage, hand, mpHands.HAND_CONNECTIONS)

        return landMarkList, originalImage

# Fungsi untuk menghitung jarak antara dua titik (Euclidean Distance)
def calculateDistance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# Fungsi untuk mengatur volume dari 0 ke 100
def setVolume(volumeLevel):
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volumeRange = volume.GetVolumeRange()
    minVolume = volumeRange[0]
    maxVolume = volumeRange[1]
    
    # Skala volume level dari 0 ke 100 menjadi minVolume ke maxVolume
    scaledVolume = (volumeLevel / 100) * (maxVolume - minVolume) + minVolume
    volume.SetMasterVolumeLevel(scaledVolume, None)

# Inisialisasi Kamera
cap = cv2.VideoCapture(0)

# Inisialisasi HandDetection
handDetection = HandDetection()

# Inisialisasi pycaw untuk mengatur volume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Mendapatkan rentang volume (min/max volume)
volumeRange = volume.GetVolumeRange()
minVolume = volumeRange[0]
maxVolume = volumeRange[1]

while True:
    success, frame = cap.read()
    if not success:
        break

    # Deteksi dan tampilkan landmark tangan
    landMarks, annotatedImage = handDetection.findHandLandMarks(image=frame, draw=True)

    if len(landMarks) != 0:
        # Landmark jempol (id 4) dan telunjuk (id 8)
        thumbTip = landMarks[4][1:]
        indexTip = landMarks[8][1:]

        # Hitung jarak antara jempol dan telunjuk
        distance = calculateDistance(thumbTip, indexTip)

        # Skala jarak menjadi rentang volume
        # Sesuaikan jarak minimal dan maksimal sesuai dengan preferensi
        minDistance = 80   # Jarak minimal (tangan hampir tertutup)
        maxDistance = 250  # Jarak maksimal (tangan terbuka penuh)
        
        # Konversi jarak ke skala volume (0 hingga 100)
        vol = ((distance - minDistance) / (maxDistance - minDistance)) * 100
        vol = max(0, min(100, vol))  # Clamping agar berada di rentang 0 hingga 100
        
        # Setel volume sesuai dengan skala 0-100
        setVolume(vol)
        
        # Tampilkan jarak dan volume pada frame
        cv2.putText(annotatedImage, f'Distance: {int(distance)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(annotatedImage, f'Volume: {int(vol)}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Tampilkan gambar dengan deteksi tangan
    cv2.imshow("Hand Detection", annotatedImage)

    # Keluar dari loop jika 'x' ditekan
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
