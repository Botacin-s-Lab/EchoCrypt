import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import argparse

keystroke_length = 0.33
begin_offset = 0.1

def separate_keystrokes(energy, treshold, recording):
    keystrokes = []
    keystroke_begin = None
    i = 0
    while i < len(energy):
        if energy[i] > treshold and keystroke_begin is None:
            keystroke_begin = i
        elif keystroke_begin is not None:
            begin = keystroke_begin - int(rate * begin_offset)
            end = max(keystroke_begin + int(keystroke_length * rate), i)
            keystrokes.append(recording[begin:end])
            keystroke_begin = None
            i = end
            continue
        i += 1

    return keystrokes

def find_treshold(energy, init_treshold, recording, step, target_keystrokes):
    cur_treshold = init_treshold
    keystrokes = separate_keystrokes(energy, init_treshold, recording)
    while len(keystrokes) != target_keystrokes:
        if len(keystrokes) > target_keystrokes:
            cur_treshold += step
        else:
            cur_treshold -= step
        step = step * 0.99
        keystrokes = separate_keystrokes(energy, cur_treshold, recording)
        print('Keystroke count:', len(keystrokes), 'Treshold:', cur_treshold, 'Step:', step)

    return (keystrokes, cur_treshold)

parser = argparse.ArgumentParser(
                    prog='Keystroke extractor',
                    description='Extracts keystrokes from wav file')

parser.add_argument('filename')
parser.add_argument('prefix', type=str)
parser.add_argument('--number_of_keystrokes', type=int, default=25)
parser.add_argument('--init_treshold', type=float, default = 0.06)
parser.add_argument('--step', type=float, default = 0.05)

args = parser.parse_args()

rate, aud_data = scipy.io.wavfile.read(args.filename)
print(aud_data.shape, rate)

channel_1 = aud_data[:]

fourier = np.fft.fft(channel_1)

# plt.figure(1)
# plt.plot(fourier)
# plt.xlabel('n')
# plt.ylabel('amplitude')
# plt.show()

energy = np.absolute(np.sum(fourier, axis=1))
print(energy.shape)

# plt.figure(2)
# plt.plot(energy)
# plt.xlabel('time')
# plt.ylabel('energy')
# plt.show()

treshold = args.init_treshold
step = args.step
keystrokes, actual_treshold = find_treshold(energy, treshold, aud_data, step, args.number_of_keystrokes)

for i in range(len(keystrokes)):
    keystroke = keystrokes[i]
    scipy.io.wavfile.write(args.prefix + str(i) +'.wav', rate, keystroke)

print('DONE')
