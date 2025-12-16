from fastai.vision.all import *
import ffmpeg


def make_hard_copy(input_file_path,
                  crf='18',
                  vcodec='libx264'):
    input_file_path = input_file_path
    output_file_path = f'{str(input_file_path).split(".")[0]}.mp4'
    (
        ffmpeg
        .input(input_file_path)
        .output(output_file_path, 
                vcodec=vcodec,
                crf=crf, 
                acodec='copy',
                movflags='+faststart')  # 2. Output flags belong inside .output()
        .run(overwrite_output=True)     # 3. Correct way to pass "-y"
    )


if __name__ == "__main__":
    base_path = '.'
    full_video_files = [file for file in get_files(base_path, extensions='.MOV')]
    for x in range(0, len(full_video_files)):
        make_hard_copy(full_video_files[x])