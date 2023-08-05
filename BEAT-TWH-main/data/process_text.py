import pdb


aligned_path = 'tts_align.txt'


with open(aligned_path.replace('.txt', '_process.tsv'), 'w') as output:
    with open(aligned_path, 'r') as f:
        FLAG_end_time = FLAG_start_time = FLAG_word = False
        end_time = start_time = word = ''
        for line in f.readlines():
            if line.startswith('      "end"'):
                line = line.split(':')
                end_time = str(float(line[1].strip().strip(',')))
                FLAG_end_time = True
            elif line.startswith('      "start"'):
                line = line.split(':')
                start_time = str(float(line[1].strip().strip(',')))
                FLAG_start_time = True
            elif line.startswith('      "word":'):
                line = line.split(':')
                word = line[1].strip().strip('"')
                FLAG_word = True
            else:
                continue
            if FLAG_end_time and FLAG_start_time and FLAG_word:
                FLAG_end_time = FLAG_start_time = FLAG_word = False
                output.write(' '.join([start_time, end_time, word]) + '\n')

