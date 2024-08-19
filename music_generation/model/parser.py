import pretty_midi
import numpy as np

path = 'C:/Users/saraa/Desktop/music_generation/jazz_files/'
out_path = 'C:/Users/saraa/Desktop/music_generation/generated_music/'
_SAMPLING_RATE = 16000

def get_notes(index, max_len=100):
    midi = pretty_midi.PrettyMIDI(path+f'{index}.mid')
    instrument = midi.instruments[0]
    notes = sorted(instrument.notes, key=lambda note : note.start)
    # gonna order the notes by pitch, then by duration
    data = [[]]
    count = 0
    prev = 0 # incorporate rests in this way to have somewhat reasonable data values...
    for note in notes:
        if(count >= max_len):
            data.append([])
            count = 0
        data[-1].append([note.pitch, note.start-prev, note.end-note.start])
        prev = note.start
        count += 1
    if(count < max_len):
        # the bad way to pad the array, i wish i could do += [0]*(max_len-count)...
        for i in range(max_len-count):
            data[-1].append([0, 0, 0])
    return data

def to_midi(notes: list, out_file: str, velocity=100):
    time = 0.0
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(1)
    for notes_i in notes:
        for x in notes_i:
            note = pretty_midi.Note(velocity=velocity, pitch=int(x[0]), start=time+x[1], end=time+x[1]+x[2])
            instrument.notes.append(note)
            time += x[1]
    pm.instruments.append(instrument)
    pm.write(out_path+out_file)
    return pm

# data shape of (2239, 100, 3)
def get_training_data(max_len=100):
    raw = []
    for i in range(456):
        raw.extend(get_notes(i))
    return np.array(raw, dtype=np.float32)

pm = to_midi(get_notes(0), 'test0.mid')
