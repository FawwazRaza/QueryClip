
import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_transcription(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def split_transcription_with_timestamps(transcription_data, chunk_size=500, chunk_overlap=100):
    full_text = ""
    segment_offsets = []
    for seg in transcription_data:
        start = len(full_text)
        full_text += seg["text"] + " "
        end = len(full_text)
        segment_offsets.append({
            "start": start,
            "end": end,
            "start_time": seg["start_time"],
            "end_time": seg["end_time"]
        })

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    chunks = splitter.split_text(full_text.strip())

    chunks_with_timestamps = []
    for chunk in chunks:
        chunk_start_char = full_text.find(chunk)
        chunk_end_char = chunk_start_char + len(chunk)

        seg_start_idx = next(
            (i for i, seg in enumerate(segment_offsets) if seg["end"] > chunk_start_char), 0
        )
        # seg_start_idx = 0
        # for i, seg in enumerate(segment_offsets):
        #     if seg["end"] > chunk_start_char:
        #         seg_start_idx = i
        #         break
        seg_end_idx = next(
            (i for i, seg in reversed(list(enumerate(segment_offsets))) if seg["start"] < chunk_end_char),
            len(segment_offsets) - 1
        )
        # seg_end_idx = 0
        # for i in range(len(segment_offsets) - 1, -1, -1):
        #     if segment_offsets[i]["start"] < chunk_end_char:
        #         seg_end_idx = i
        #         break
        # else:
        #     seg_end_idx = len(segment_offsets) - 1
        chunk_start_time = segment_offsets[seg_start_idx]["start_time"]
        chunk_end_time = segment_offsets[seg_end_idx]["end_time"]
        chunks_with_timestamps.append({
            "text": chunk.strip(),
            "start_time": chunk_start_time,
            "end_time": chunk_end_time
        })
    return chunks_with_timestamps

def save_chunks_to_file(chunks, output_dir, video_title):
    
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f"{video_title}_chunks.json")
    with open(output_file_path, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=4, ensure_ascii=False)
    print(f"Chunks saved to {output_file_path}")

if __name__ == "__main__":
    transcription_file_path = "../data/transcriptions.json"
    output_directory = "../data/chunks"

    transcription_data = load_transcription(transcription_file_path)

    for video in transcription_data:
        video_title = video["video"].replace(".mp4", "")
        transcription = video["transcription"]
        chunks = split_transcription_with_timestamps(transcription)

        save_chunks_to_file(chunks, output_directory, video_title)
