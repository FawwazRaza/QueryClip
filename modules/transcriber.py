import os
import whisper
import json
from moviepy.editor import VideoFileClip

def process_videos(video_directory, output_json_path):
    model = whisper.load_model("base")
    transcriptions = []

    for filename in os.listdir(video_directory):
        if filename.endswith(".mp4"):
            video_path = os.path.join(video_directory, filename)
            audio_path = os.path.join(video_directory, f"{os.path.splitext(filename)[0]}.wav")
            text_file_path = os.path.join(video_directory, f"{os.path.splitext(filename)[0]}_transcription.txt")

            # Extract audio
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path)

            # Transcribe audio
            result = model.transcribe(audio_path)
            transcription = [
                {"text": seg["text"], "start_time": seg["start"], "end_time": seg["end"]}
                for seg in result["segments"]
            ]

            with open(text_file_path, "w", encoding="utf-8") as text_file:
                for seg in transcription:
                    text_file.write(f"{seg['text']}\n")

            transcriptions.append({"video": filename, "transcription": transcription})

            os.remove(audio_path)

    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(transcriptions, json_file, indent=4)

if __name__ == "__main__":
    video_directory = "../data/videos"
    output_json_path = "../data/transcriptions.json"

    if not os.path.exists(video_directory):
        print(f"Video directory '{video_directory}' does not exist.")
    else:
        process_videos(video_directory, output_json_path)
