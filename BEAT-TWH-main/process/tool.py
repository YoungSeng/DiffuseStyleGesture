"""
This script contains supporting function for the data processing.
It is used in several other scripts:
for generating bvh files, aligning sequences and calculation of speech features

@author: Taras Kucherenko
"""
import pdb

import librosa
import librosa.display
from essentia.standard import *
from pydub import AudioSegment
import parselmouth as pm
import os
import numpy as np
import scipy

NFFT = 4096
MFCC_INPUTS = 40 # How many features we will store for each MFCC vector
HOP_LENGTH = 1/30
DIM = 64


def derivative(x, f):
    """ Calculate numerical derivative (by FDM) of a 1d array
    Args:
        x: input space x
        f: Function of x
    Returns:
        der:  numerical derivative of f wrt x
    """

    x = 1000 * x  # from seconds to milliseconds

    # Normalization:
    dx = (x[1] - x[0])

    cf = np.convolve(f, [1, -1]) / dx

    # Remove unstable values
    der = cf[:-1].copy()
    der[0] = 0

    return der


def create_bvh(filename, prediction, frame_time):
    """
    Create BVH File
    Args:
        filename:    file, in which motion in bvh format should be written
        prediction:  motion sequences, to be written into file
        frame_time:  frame rate of the motion
    Returns:
        nothing, writes motion to the file
    """
    with open('hformat.txt', 'r') as ftemp:
        hformat = ftemp.readlines()

    with open(filename, 'w') as fo:
        prediction = np.squeeze(prediction)
        print("output vector shape: " + str(prediction.shape))
        offset = [0, 60, 0]
        offset_line = "\tOFFSET " + " ".join("{:.6f}".format(x) for x in offset) + '\n'
        fo.write("HIERARCHY\n")
        fo.write("ROOT Hips\n")
        fo.write("{\n")
        fo.write(offset_line)
        fo.writelines(hformat)
        fo.write("MOTION\n")
        fo.write("Frames: " + str(len(prediction)) + '\n')
        fo.write("Frame Time: " + frame_time + "\n")
        for row in prediction:
            row[0:3] = 0
            legs = np.zeros(24)
            row = np.concatenate((row, legs))
            label_line = " ".join("{:.6f}".format(x) for x in row) + " "
            fo.write(label_line + '\n')
        print("bvh generated")


def shorten(arr1, arr2, min_len=0):

    if min_len == 0:
        min_len = min(len(arr1), len(arr2))

    arr1 = arr1[:min_len]
    arr2 = arr2[:min_len]

    return arr1, arr2


def average(arr, n):
    """ Replace every "n" values by their average
    Args:
        arr: input array
        n:   number of elements to average on
    Returns:
        resulting array
    """
    end = n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)


def calculate_spectrogram(audio, sr):
    """ Calculate spectrogram for the audio file
    Args:
        audio_filename: audio file name
        duration: the duration (in seconds) that should be read from the file (can be used to load just a part of the audio file)
    Returns:
        log spectrogram values
    """


    # Make stereo audio being mono
    if len(audio.shape) == 2:
        audio = (audio[:, 0] + audio[:, 1]) / 2

    spectr = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=NFFT,
                                            hop_length=int(HOP_LENGTH * sr),
                                            n_mels=DIM)

    # Shift into the log scale
    eps = 1e-10
    log_spectr = np.log(abs(spectr)+eps)

    return np.transpose(log_spectr)


def calculate_mfcc(audio, sr):
    """
    Calculate MFCC features for the audio in a given file
    Args:
        audio_filename: file name of the audio
    Returns:
        feature_vectors: MFCC feature vector for the given audio file
    """

    # Make stereo audio being mono
    if len(audio.shape) == 2:
        audio = (audio[:, 0] + audio[:, 1]) / 2

    # Calculate MFCC feature with the window frame it was designed for
    input_vectors = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=MFCC_INPUTS, n_fft=NFFT, hop_length=int(HOP_LENGTH * sr),
                                            n_mels=DIM)

    return input_vectors.transpose()


def extract_prosodic_features(audio_filename):
    """
    Extract all 5 prosodic features
    Args:
        audio_filename:   file name for the audio to be used
    Returns:
        pros_feature:     energy, energy_der, pitch, pitch_der, pitch_ind
    """

    # Read audio from file
    sound = AudioSegment.from_file(audio_filename, format="wav")

    # Alternative prosodic features
    pitch, energy = compute_prosody(audio_filename, HOP_LENGTH / 10)

    duration = len(sound) / 1000
    t = np.arange(0, duration, HOP_LENGTH / 10)

    energy_der = derivative(t, energy)
    pitch_der = derivative(t, pitch)

    # Average everything in order to match the frequency
    energy = average(energy, 10)
    energy_der = average(energy_der, 10)
    pitch = average(pitch, 10)
    pitch_der = average(pitch_der, 10)

    # Cut them to the same size
    min_size = min(len(energy), len(energy_der), len(pitch_der), len(pitch_der))
    energy = energy[:min_size]
    energy_der = energy_der[:min_size]
    pitch = pitch[:min_size]
    pitch_der = pitch_der[:min_size]

    # Stack them all together
    pros_feature = np.stack((energy, energy_der, pitch, pitch_der))#, pitch_ind))

    # And reshape
    pros_feature = np.transpose(pros_feature)

    return pros_feature


def compute_prosody(audio_filename, time_step=0.05):
    audio = pm.Sound(audio_filename)

    # Extract pitch and intensity
    pitch = audio.to_pitch(time_step=time_step)
    intensity = audio.to_intensity(time_step=time_step)

    # Evenly spaced time steps
    times = np.arange(0, audio.get_total_duration() - time_step, time_step)

    # Compute prosodic features at each time step
    pitch_values = np.nan_to_num(
        np.asarray([pitch.get_value_at_time(t) for t in times]))
    intensity_values = np.nan_to_num(
        np.asarray([intensity.get_value(t) for t in times]))

    intensity_values = np.clip(
        intensity_values, np.finfo(intensity_values.dtype).eps, None)

    # Normalize features [Chiu '11]
    pitch_norm = np.clip(np.log(pitch_values + 1) - 4, 0, None)
    intensity_norm = np.clip(np.log(intensity_values) - 3, 0, None)

    return pitch_norm, intensity_norm

def extract_onsets(wav_path):

    # Load audio file.
    audio = MonoLoader(filename=wav_path, sampleRate=16000)()

    # 1. Compute the onset detection function (ODF).

    # The OnsetDetection algorithm provides various ODFs.
    od_hfc = OnsetDetection(method='hfc', sampleRate=16000)
    # od_complex = OnsetDetection(method='complex', sampleRate=16000)

    # We need the auxilary algorithms to compute magnitude and phase.
    w = Windowing(type='hann')
    fft = FFT()  # Outputs a complex FFT vector.
    c2p = CartesianToPolar()  # Converts it into a pair of magnitude and phase vectors.

    # Compute both ODF frame by frame. Store results to a Pool.
    pool = essentia.Pool()
    for frame in FrameGenerator(audio, frameSize=1024, hopSize=512):
        magnitude, phase = c2p(fft(w(frame)))
        pool.add('odf.hfc', od_hfc(magnitude, phase))
        # pool.add('odf.complex', od_complex(magnitude, phase))

    # 2. Detect onset locations.
    onsets = Onsets(frameRate=16000.0/512.0, silenceThreshold=0.04)     # default frameRate 44100/512=86.1328, silenceThreshold=0.02

    # This algorithm expects a matrix, not a vector.
    onsets_hfc = onsets(essentia.array([pool['odf.hfc']]), [1])        # frameRate 44100/512=86.1328
    # You need to specify weights, but if we use only one ODF
    # it doesn't actually matter which weight to give it

    # onsets_complex = onsets(essentia.array([pool['odf.complex']]), [1])

    # Add onset markers to the audio and save it to a file.
    # We use beeps instead of white noise and stereo signal as it's more distinctive.

    # We want to keep beeps in a separate audio channel.
    # Add them to a silent audio and use the original audio as another channel. Mux both into a stereo signal.
    silence = [0.] * len(audio)

    beeps_hfc = AudioOnsetsMarker(onsets=onsets_hfc, type='beep', sampleRate=16000)(silence)
    # beeps_complex = AudioOnsetsMarker(onsets=onsets_complex, type='beep')(silence)

    # audio_hfc = StereoMuxer()(audio, beeps_hfc)
    # audio_complex = StereoMuxer()(audio, beeps_complex)

    # Write audio to files in a temporary directory.
    # temp_dir = './tmp'
    # if not os.path.exists(temp_dir):
    #     os.makedirs(temp_dir)
    # AudioWriter(filename=temp_dir+ '/hiphop_onsets_hfc_stereo.mp3', format='mp3', sampleRate=16000)(audio_hfc)
    # AudioWriter(filename=temp_dir + '/hiphop_onsets_complex_stereo.mp3', format='mp3')(audio_complex)
    #
    # n_frames = len(pool['odf.hfc'])
    # frames_position_samples = np.array(range(n_frames)) * 512
    #
    # fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(4, 1, sharex=True, sharey=False, figsize=(15, 16))
    #
    # ax1.set_title('HFC ODF')
    # ax1.plot(frames_position_samples, pool['odf.hfc'], color='magenta')
    #
    # ax2.set_title('Complex ODF')
    # ax2.plot(frames_position_samples, pool['odf.complex'], color='red')
    #
    # ax3.set_title('Audio waveform and the estimated onset positions (HFC ODF)')
    # ax3.plot(audio)
    # for onset in onsets_hfc:
    #     ax3.axvline(x=onset * 44100, color='magenta')
    #
    # ax4.set_title('Audio waveform and the estimated onset positions (complex ODF)')
    # ax4.plot(audio)
    # for onset in onsets_complex:
    #     ax4.axvline(x=onset * 44100, color='red')
    #
    # plt.savefig(temp_dir + '/onset_detection.png')

    audio_hfc = StereoMuxer()(silence, beeps_hfc)       # y = audio_hfc[:, 1]

    # AudioWriter(filename=temp_dir+ '/hiphop_onsets_hfc_stereo_.mp3', format='mp3', sampleRate=16000)(audio_hfc)

    return onsets_hfc, audio_hfc
