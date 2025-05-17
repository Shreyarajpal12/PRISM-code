##########################################################
### 1) Imports & Install Instructions
##########################################################

"""
You likely need to install these (or confirm they're in your environment):

pip install --upgrade open-clip-torch timm
pip install opencv-python torch transformers openai-whisper ruptures
pip install langchain_ollama langchain_community
pip install git+https://github.com/openai/whisper.git  (for latest whisper)
pip install pytube  (only if you need to download from YouTube)
pip install ffmpeg-python (optional if you want pythonic ffmpeg control)

Also ensure you have ffmpeg installed system-wide (e.g. `apt-get install ffmpeg` on Ubuntu).
"""

import os
import json
import random
import subprocess
import numpy as np
import torch
import cv2
import ruptures as rpt

from PIL import Image
from io import BytesIO
from openai import OpenAI

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
# Initialize OpenAI client
client = OpenAI(api_key="ENTER YOUR API KEY HERE")

# open_clip / BiomedCLIP
from open_clip import create_model_from_pretrained, get_tokenizer

# LLM: LLaVA (for image-based Q&A)
from langchain_ollama import OllamaLLM   # usage: llm_llava = OllamaLLM(model="minicpm-v")

# LLM: LLaMA (for text-based summarization)
from langchain_community.llms import Ollama   # usage: llm_llama = Ollama(model="llama3.1")

##########################################################
### 2) Setup CLIP Model
##########################################################

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()


context_length = 256  # For BiomedCLIP, default is 256

##########################################################
### 3) Audio Extraction & Full-Video Whisper Transcription
###    (Processed Separately & Merged at Final)
##########################################################

def extract_audio_ffmpeg(video_path, output_audio="temp_audio.wav"):
    """
    Extract the audio track from a video using ffmpeg.
    Saves to 'temp_audio.wav' by default.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-q:a", "0",
        "-ac", "1",
        "-ar", "16000",
        "-map", "a",
        output_audio
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not os.path.exists(output_audio):
        raise RuntimeError("Failed to extract audio with ffmpeg")

    return output_audio

def transcribe_audio_with_whisper(audio_path, model_size="base"):
    """
    Use OpenAI Whisper to transcribe an audio file fully.
    Returns a single text string with the entire transcript.
    """
    import whisper  # from openai-whisper
    whisper_model = whisper.load_model(model_size)
    result = whisper_model.transcribe(audio_path, verbose=False)
    # Combine entire transcribed text:
    return result["text"].strip()

def get_full_video_transcript(video_path, whisper_model_size="base"):
    """
    Extracts audio from the video, runs Whisper, and returns the entire transcript text.
    """
    audio_file = extract_audio_ffmpeg(video_path, "temp_audio.wav")
    full_transcript = transcribe_audio_with_whisper(audio_file, model_size=whisper_model_size)
    # (Optionally delete temp_audio.wav if no longer needed)
    # os.remove(audio_file)
    return full_transcript

##########################################################
### 4) Frame Extraction & Basic Feature Extraction
###    (Your Existing Pipeline Logic)
##########################################################

# As an example, you used ResNet (from "microsoft/resnet-18"),
# but below is the reference to that step.
# If you want to switch to CLIP or BiomedCLIP features for the same logic,
# you can adapt. We'll keep your original approach for the threshold/PELT selection.

from transformers import AutoImageProcessor, AutoModelForImageClassification
def extract_frames_per_second(video_path, output_folder, fps_extract=1):
    """
    Extract frames at a specified rate (fps_extract frames per second).
    Returns: (frames, frame_indices, extracted_frame_numbers, fps)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_capture = cv2.VideoCapture(video_path)
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, fps // fps_extract)

    frames = []
    frame_indices = []
    extracted_frame_numbers = []
    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
            frame_indices.append(frame_count)
            extracted_frame_numbers.append(extracted_count + 1)
            frame_filename = os.path.join(output_folder, f"frame_{extracted_count+1:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1
        frame_count += 1

    video_capture.release()
    return frames, frame_indices, extracted_frame_numbers, fps


def extract_resnet_features(frames):
    """
    Use ResNet-18 from Hugging Face for feature extraction.
    (You can adapt this to any model for feature-based analysis.)
    """
    image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
    model_rn = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18").to(device)
    model_rn.eval()

    frame_features = []
    for idx, frame in enumerate(frames):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = image_processor(frame_rgb, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model_rn(**inputs).logits
            feat_vec = logits.squeeze().cpu().numpy()
            frame_features.append(feat_vec)
    return frame_features

def compute_feature_differences(features):
    """
    Consecutive L2-norm differences for change detection.
    """
    diffs = [np.linalg.norm(features[i] - features[i+1]) for i in range(len(features)-1)]
    return diffs

def select_frames(diffs, method, threshold=None):
    """
    Basic frame selection by either threshold or PELT (ruptures).
    Return a 1-based list of indices to keep.
    """
    total_possible = len(diffs) + 1
    selected = []
    if method == "threshold":
        selected = [i + 1 for i, diff in enumerate(diffs) if diff > threshold]
    elif method == "pelt":
        algo = rpt.Pelt(model="l2").fit(np.array(diffs).reshape(-1, 1))
        selected = algo.predict(pen=1)[:-1]  # last index is len(diffs)+1

    # Make sure we don't exceed total frames
    selected = [idx for idx in selected if idx < total_possible]
    return selected

def map_frames_bidirectional(threshold_set, pelt_set, neighbor_threshold=3, take_both=False):
    """
    Combine frames from threshold-based selection & PELT-based selection
    with some neighbor logic.
    """
    final_set = set()
    common = threshold_set.intersection(pelt_set)
    final_set.update(common)

    threshold_only = threshold_set - common
    pelt_only = pelt_set - common

    for pelt_frame in pelt_only:
        neighbors = [frame for frame in threshold_only if abs(frame - pelt_frame) <= neighbor_threshold]
        if neighbors:
            closest_neighbor = min(neighbors, key=lambda x: abs(x - pelt_frame))
            if take_both:
                final_set.add(pelt_frame)
                final_set.add(closest_neighbor)
            else:
                final_set.add(closest_neighbor)
        else:
            final_set.add(pelt_frame)

    print(f"Total frames selected after mapping: {len(final_set)}")
    return sorted(final_set)

##########################################################
### 5) LLaVA: Frame-by-Frame (or segment) Image QA and Adaptive sampling
##########################################################

def convert_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    return buffered.getvalue()
def adaptive_select_from_segments(final_frames, final_extracted_numbers, features, segment_size=10):
    selected_frames = []
    selected_numbers = []
    segment_avg_diffs = []  # To store average diffs of each segment

    for i in range(0, len(final_frames), segment_size):
        segment = final_frames[i:i+segment_size]
        segment_numbers = final_extracted_numbers[i:i+segment_size]
        segment_features = features[i:i+segment_size]

        print(f"\nProcessing segment {i//segment_size + 1}: Frames {segment_numbers}")

        if len(segment) < 2:
            print(f"  - Segment too small (only {len(segment)} frame(s)), picking all frames.")
            selected_frames.extend(segment)
            selected_numbers.extend(segment_numbers)
            continue

        # Compute differences between consecutive frames in the segment
        diffs = [float(np.linalg.norm(segment_features[j] - segment_features[j+1])) for j in range(len(segment_features)-1)]
        avg_diff = np.mean(diffs)
        segment_avg_diffs.append(avg_diff)

        print(f"  - Average difference in segment: {avg_diff:.2f}")
        print(f"  - Differences between consecutive frames: {[round(d, 2) for d in diffs]}")

    # After processing all segments, calculate global threshold
    global_median_diff = np.median(segment_avg_diffs)
    print(f"\n=== Global median difference across all segments: {global_median_diff:.2f} ===")
    diff_threshold = global_median_diff
    print(f"=== Adaptive threshold set to: {diff_threshold:.2f} ===\n")

    # Now do the actual selection based on new diff_threshold
    for i in range(0, len(final_frames), segment_size):
        segment = final_frames[i:i+segment_size]
        segment_numbers = final_extracted_numbers[i:i+segment_size]
        segment_features = features[i:i+segment_size]

        if len(segment) < 2:
            selected_frames.extend(segment)
            selected_numbers.extend(segment_numbers)
            continue

        diffs = [float(np.linalg.norm(segment_features[j] - segment_features[j+1])) for j in range(len(segment_features)-1)]
        avg_diff = np.mean(diffs)

        if avg_diff > diff_threshold:
            # High motion detected
            sorted_diff_indices = np.argsort(diffs)[-2:]  # Safely get top-2
            sorted_diff_indices = sorted(sorted_diff_indices)  # Keep order
            print(f"  - High motion segment detected (Avg: {avg_diff:.2f}). Picking top-2 frames: {sorted_diff_indices}")

            for idx in sorted_diff_indices:
                if idx < len(segment):
                    selected_frames.append(segment[idx])
                    selected_numbers.append(segment_numbers[idx])
                else:
                    print(f"Warning: Index {idx} out of segment range. Skipping.")
        else:
            # Low motion detected
            rand_idx = random.randint(0, len(segment) - 1)
            print(f"  - Low motion segment (Avg: {avg_diff:.2f}). Picking 1 random frame at index {rand_idx} (Frame {segment_numbers[rand_idx]}).")
            selected_frames.append(segment[rand_idx])
            selected_numbers.append(segment_numbers[rand_idx])

    print(f"\nTotal selected frames after adaptive sampling: {len(selected_frames)}")
    return selected_frames, selected_numbers


def process_frame_with_llava(frame):
    """
    Call the LLaVA model with a prompt describing the frame.
    """
    # LLaVA instance (model="minicpm-v")
    llm_llava = OllamaLLM(model="minicpm-v")

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_base64 = convert_to_base64(img)

    prompt = (
        "Explain the what is happening in the image. This is a frame from a video of an activity spanning daily activities likecooking, playing , dancing, sports. etc.."

    )
    response = llm_llava.invoke(prompt, images=[image_base64])
    return response

##########################################################
### 6) LLaMA for Text Summaries
##########################################################

def call_llama_model(prompt):
    """
    Calls a LLaMA model (model="llama3.1") to get text completions.
    """
    llm_llama = Ollama(model="llama3.1")
    return llm_llama.invoke([prompt])

##########################################################
### 7) GPT Label Generation for LLaVA Outputs
###    (Placeholder for "gpt-4o" or your custom GPT usage)
##########################################################

def call_gpt_text(prompt):
    try:
        chat_completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT text call failed: {e}")
        return ""

def update_json_labels(analysis_folder="./"):
    """
    Iterates over JSON files named '*_analysis.json', calls GPT to refine each label.
    Appends the 'gpt_label' to each entry.
    """
    json_files = [f for f in os.listdir(analysis_folder) if f.endswith("_analysis.json")]
    for json_file in json_files:
        json_path = os.path.join(analysis_folder, json_file)
        print(f"Processing {json_file}")

        with open(json_path, "r") as file:
            data = json.load(file)

        for entry in data:
            label = entry.get("llava_output", "")
            if label:
                prompt = f"""Extract info from: {label}
tell me in few words (5-6) by giving a label for the most important details that you find from the text description of the activity happening in the image i give you,
this is an action and i want to know what action is currently happening,
Don't make the label vague like only the heading of the main activity, tell me exactly what is going on in the image and only give the label as in simple words."""
                gpt_label = call_gpt_text(prompt)
                gpt_valid= call_gpt_text(f"imagine you are an expert on validating labels, given this label: {gpt_label}, do you think it is valid to check an image in an important video? like is it useless? like black screen or too general like this is a surgery video, or this is a girl standnig . so based on your judgement, i dont want labels that are useless say nothing about the image or toooo general, return 0 if you think this is unimportant or too general, return 1 if you think it is important label, return only a number nothing else ")
                if int(gpt_valid)==0:
                    pass
                else:                     
                    entry["gpt_label"] = gpt_label

        # Save updated JSON
        with open(json_path, "w") as file:
            json.dump(data, file, indent=4)

        print(f"Updated GPT labels for {json_file}")


##########################################################
### 8) After LLaVA + GPT Labeling:  We use BiomedCLIP to
###    score frames vs. these labels, threshold them,
###    store results in JSON
##########################################################

def load_labels(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    # It's a list of frame entries with a "gpt_label"
    return [entry["gpt_label"] for entry in data if "gpt_label" in entry]



def process_frames_with_labels_clip(
    filtered_folder,
    json_file,
    threshold=0.9,
    output_json="frames_with_high_scores_clip.json",
    unused_labels_json="unused_labels_clip.json"
):

    labels = load_labels(json_file)
    label_occurrences = {label: 0 for label in labels}

    frame_files = sorted(os.listdir(filtered_folder))
    segment_size = 10

    frames_with_labels = []
    frames_without_labels = []
    best_labels_set = set()
    top_five_labels_set = set()
    total_no_label_count = 0

    for segment_index in range(0, len(frame_files), segment_size):
        segment_frames = frame_files[segment_index:segment_index + segment_size]
        images = []

        for img_name in segment_frames:
            im_path = os.path.join(filtered_folder, img_name)
            try:
                image = Image.open(im_path).convert("RGB")
                images.append(image)
            except Exception as e:
                print(f"❌ Skipping {img_name}: {e}")

        if not images:
            continue

        text_prompts = [f"a photo of {label}" for label in labels]
        inputs = clip_processor(text=text_prompts, images=images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits = outputs.logits_per_image  # [batch, num_labels]
            min_val = logits.min(dim=1, keepdim=True).values
            max_val = logits.max(dim=1, keepdim=True).values
            logits = (logits - min_val) / (max_val - min_val + 1e-8)
            logits = logits.cpu().numpy()

        print(f"Processing segment {(segment_index // segment_size) + 1} with all labels...")

        for i, img_name in enumerate(segment_frames):
            row = logits[i]
            best_score = 0.0
            best_label = None
            sorted_indices = row.argsort()[::-1][:5]
            top_five_for_frame = [labels[idx] for idx in sorted_indices]
            top_five_labels_set.update(top_five_for_frame)

            valid_labels = []
            for j, label in enumerate(labels):
                score = float(row[j])
                if score > 0.0:
                    valid_labels.append({"label": label, "score": round(score, 2)})
                    label_occurrences[label] += 1
                if score > best_score:
                    best_score = score
                    best_label = label

            if best_score >= threshold:
                frames_with_labels.append({
                    "frame_name": img_name,
                    "label_count": len(valid_labels),
                    "labels": valid_labels,
                    "best_label": best_label
                })
                best_labels_set.add(best_label)
            else:
                frames_without_labels.append(img_name)
                total_no_label_count += 1

    # Save high-confidence frames
    with open(output_json, "w") as f:
        json.dump(frames_with_labels, f, indent=4)

    # Save unused label info
    unused_labels = list(set(labels) - best_labels_set)
    unused_but_recognized = [lb for lb in unused_labels if label_occurrences[lb] > 0]
    with open(unused_labels_json, "w") as f:
        json.dump({
            "unused_labels": unused_labels,
            "best_labels": list(best_labels_set),
            "unused_but_recognized": unused_but_recognized,
            "top_five_labels": list(top_five_labels_set)
        }, f, indent=4)

    print(f"\nTotal frames without any label: {total_no_label_count}")
    print(f"Frames with no valid labels: {frames_without_labels}")
    print(f"Unused labels (never best): {unused_labels}")
    print(f"Unused but recognized: {unused_but_recognized}")
    print(f"Labels in top-5 of any frame: {top_five_labels_set}")

    return total_no_label_count, frames_without_labels, frames_with_labels, unused_labels, unused_but_recognized
##########################################################
### 9) Combine Frames (2×2) + Summaries with LLaVA
##########################################################

def combine_frames(frame_paths, output_path):
    """
    Creates a 2×2 grid from four frames, saves to disk.
    """
    frames = [cv2.imread(fp) for fp in frame_paths]
    frames = [cv2.resize(f, (256, 256)) for f in frames]
    top_row = np.hstack((frames[0], frames[1]))
    bottom_row = np.hstack((frames[2], frames[3]))
    combined = np.vstack((top_row, bottom_row))
    cv2.imwrite(output_path, combined)
    return output_path

def process_combined_frame(combined_frame_path, majority_label):
    """
    Pass the combined 2×2 image into LLaVA with a more descriptive prompt.
    """
    llm_llava = OllamaLLM(model="minicpm-v")
    frame = cv2.imread(combined_frame_path)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_base64 = convert_to_base64(img)

    prompt = f"""
**Context:** This is a combination of 4 images from  a video of an activity spanning daily activities likecooking, playing , dancing, sports. etc..
Each small picture represents a step in the sequence:
   The current major label is '{majority_label}'.
- Top-right: Step 2
- Top-left: Step 1
- Bottom-right: Step 4
- Bottom-left: Step 3


Provide a detailed description for each of the 4 images."""

    response = llm_llava.invoke(prompt, images=[image_base64])
    return response

def generate_combined_summaries(
    filtered_folder,
    high_score_json,
    combined_folder="combined_frames_resec",
    output_json="combined_summary_minicpm-v.json"
):
    """
    Takes the frames that have best_label >= threshold, groups them in 4s,
    merges them into a single image, calls LLaVA for a multi-frame summary,
    and saves the result in JSON.
    """
    # Load frames with high scores
    with open(high_score_json, "r") as file:
        frames_data = json.load(file)

    # Map frame name -> all info
    frame_info_map = {entry["frame_name"]: entry for entry in frames_data}

    os.makedirs(combined_folder, exist_ok=True)
    summaries = []

    # Process in groups of 4
    for i in range(0, len(frames_data), 4):
        group = frames_data[i:i+4]
        if len(group) < 4:
            break  # skip if not enough frames for a full 2x2 grid

        frame_paths = [
            os.path.join(filtered_folder, item["frame_name"])
            for item in group
        ]
        combined_name = f"combined_{i//4}.jpg"
        combined_path = os.path.join(combined_folder, combined_name)

        # Create the combined image
        combined_path = combine_frames(frame_paths, combined_path)

        # Majority label
        best_labels = [entry["best_label"] for entry in group]
        counts = {}
        for lb in best_labels:
            counts[lb] = counts.get(lb, 0) + 1
        max_count = max(counts.values())
        majority_labels = [lb for lb, c in counts.items() if c == max_count]
        if len(majority_labels) == 1:
            majority_label = majority_labels[0]
        else:
            majority_label = ", ".join(majority_labels)

        print(f"Processing combined frame {combined_name} with majority label: {majority_label}")
        summary = process_combined_frame(combined_path, majority_label)

        # store each frame's best score for reference
        frames_this_group = []
        for entry in group:
            fname = entry["frame_name"]
            blabel = entry["best_label"]
            bscore = max(d["score"] for d in entry["labels"] if d["label"] == blabel)
            frames_this_group.append({
                "frame_name": fname,
                "best_label": blabel,
                "best_score": bscore
            })

        summaries.append({
            "combined_frame_name": combined_name,
            "summary": summary,
            "frames": frames_this_group,
            "majority_label": majority_label
        })

    # Save
    with open(output_json, "w") as file:
        json.dump(summaries, file, indent=4)

    print(f"Summaries saved to {output_json}")

##########################################################
### 10) Final Summaries (Recursive Summarization) +
###     Merging Entire Audio Transcript at the End
##########################################################

def call_model_llama(prompt):
    """
    Just re-aliasing to keep consistent naming;
    or you can directly call call_llama_model.
    """
    return call_llama_model(prompt)

def chunk_text(texts, chunk_size=2040):
    """
    Splits a list of text segments into chunks
    so each chunk is under ~chunk_size words.
    """
    chunks = []
    current_chunk = []
    current_length = 0

    for txt in texts:
        # approximate word count by splitting on whitespace
        text_length = len(txt.split())
        if current_length + text_length <= chunk_size:
            current_chunk.append(txt)
            current_length += text_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [txt]
            current_length = text_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def recursive_summarization(chunks, level=1, all_chunks=None):
    """
    Summarize large amounts of text recursively until it's short enough.
    all_chunks will store intermediate steps for debugging if needed.
    """
    if all_chunks is None:
        all_chunks = {}

    summaries = []
    chunk_data = {}

    for i, chunk in enumerate(chunks):
        chunk_name = f"Stage Group {i+1}"
        chunk_data[chunk_name] = chunk

        prompt = f"""
You are summarizing partial text from a video of an activity spanning daily activities likecooking, playing , dancing, sports. etc.
try to give a name to this activity like a lady doing movements could be dancing, try to draw correlation like if there is a cat and some food then it could be abotu that 
try to correlate thedifferent activity to speak for one main act, like what does the bunch of activities infer? what could be the main activity or theme happening? begin with a title to the video and describe on those directions
Rewrite the text in a frame-wise, narrative format with transitions between steps:

{chunk}

---
"""
        summary = call_model_llama(prompt)  # or call_gpt_text
        summaries.append(summary)
        chunk_data[f"Summary of {chunk_name}"] = summary

    all_chunks[f"Level {level} Stages"] = chunk_data
    merged_summary = " ".join(summaries)

    # If merged_summary is still huge, chunk & re-summarize
    if len(merged_summary.split()) > 10000:  # arbitrary cutoff
        new_chunks = chunk_text([merged_summary])
        return recursive_summarization(new_chunks, level + 1, all_chunks)

    return merged_summary, all_chunks

def process_all_combined_summaries(
    input_dir="./",
    output_dir="./",
    final_prefix="report_lap_"
):
    """
    Reads all files that match 'combined_summary_*.json',
    merges them, does multi-level summarization,
    and writes final structured JSON.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect all combined summary JSONs
    combined_files = [
        f for f in os.listdir(input_dir)
        if f.endswith(".json") and f.startswith("combined_summary_")
    ]

    for file in combined_files:
        input_path = os.path.join(input_dir, file)
        method_name = file.replace("combined_summary_", "").replace(".json", "")
        output_path = os.path.join(output_dir, f"{final_prefix}{method_name}.json")

        with open(input_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        # Build a list of structured text for each "combined" block
        structured_summaries = []
        for i, entry in enumerate(json_data):
            if "summary" in entry:
                structured_summaries.append(f"Stage {i+1}: {entry['summary']}")

        # Summaries can still be huge. We'll chunk them:
        chunks = chunk_text(structured_summaries, chunk_size=2000)
        final_summary_text, chunk_details = recursive_summarization(chunks)

        # Now we have a "final_summary_text" for all combined frames
        # We'll store it in JSON.
        result_data = {
            "final_summary": final_summary_text,
            "chunk_details": chunk_details
        }

        # Save
        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(result_data, out_file, indent=4)

        print(f"Saved structured report for '{method_name}' to {output_path}")

##########################################################
### 11) Merge the Entire Whisper Transcript at the End
##########################################################

def merge_audio_transcript_with_summary(
    final_summary_json,
    transcript_text,
    output_json="complete_summary_report.json"
):
    """
    Takes the 'final_summary' from your summary JSON,
    merges with the full audio transcript at the end,
    and calls an LLM to produce a final, integrated summary.
    """
    with open(final_summary_json, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    final_summary_text = summary_data.get("final_summary", "")
    chunk_details = summary_data.get("chunk_details", {})

    # Now unify them into one single doc
    prompt = f"""
We have a final summary of video frames:

{final_summary_text}

We also have the raw transcript from the entire  audio:

{transcript_text}

Please produce a unified, cohesive summary of the entire steps in the activity happneing in the video which could be related to one of these: daily activities likecooking, playing , dancing, sports. etc..,
incorporating relevant information from the audio transcript. If the transcript
provides additional details or clarifications, weave them into the final summary.
If the transcript includes extraneous content, omit it.
Focus on a  coherent storyline of the entire action or activity of the video.
"""

    integrated_summary = call_model_llama(prompt)

    # Save
    final_report = {
        "frames_summary": final_summary_text,
        "audio_transcript": transcript_text,
        "integrated_summary": integrated_summary,
        "chunk_details": chunk_details
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4)

    print(f"Integrated final report saved to {output_json}")


def normalize(arr):
    a = np.array(arr, dtype=float)
    mn, mx = a.min(), a.max()
    return ((a - mn) / (mx - mn + 1e-8)).tolist()

def full_pipeline_example(video_path, output_base, extracted_folder, filtered_folder, combined_folder):
    """
    Runs the full pipeline AND computes s1–s5 for each adaptively selected frame,
    then writes out `frame_scores.json` at the end.
    """
    os.makedirs(output_base, exist_ok=True)

    # 1) Transcribe Audio (skipped)
    entire_transcript = get_full_video_transcript(video_path, whisper_model_size="base")
    with open(os.path.join(output_base, "transcript.txt"), "w") as f:
        f.write(entire_transcript)

    # 2) Extract Frames
    frames, frame_indices, extracted_numbers, fps = extract_frames_per_second(video_path, extracted_folder, fps_extract=1)

    # 3) Extract Features & raw diffs
    features = extract_resnet_features(frames)
    diffs = compute_feature_differences(features)

    # 4) Frame Selection: threshold + PELT
    THRESH = 30.0
    thr_set = set(i+1 for i,d in enumerate(diffs) if d > THRESH)
    pelt_frames = set(select_frames(diffs, method="pelt"))
    final_nums = map_frames_bidirectional(thr_set, pelt_frames, neighbor_threshold=3)

    # Copy filtered frames
    os.makedirs(filtered_folder, exist_ok=True)
    for num in final_nums:
        fname = f"frame_{num:04d}.jpg"
        subprocess.run(["cp",
            os.path.join(extracted_folder, fname),
            os.path.join(filtered_folder, fname)
        ])

    # 5) Prepare for adaptive sampling
    sel_feats = [features[num-1] for num in final_nums]
    sel_s1_raw = [diffs[num-1] if diffs[num-1] > THRESH else 0.0 for num in final_nums]
    sel_s1 = normalize(sel_s1_raw)        # s1 for each final_nums
    sel_s2_raw = [1.0 if num in pelt_frames else 0.0 for num in final_nums]
    sel_s2 = normalize(sel_s2_raw)        # s2 for each final_nums
    sel_frames = [frames[num-1] for num in final_nums]

    # 6) Adaptive Sampling -> returns adaptively_selected_frames, numbers, and optionally raw_s3
    try:
        adapt_frames, adapt_nums, raw_s3 = adaptive_select_from_segments(
            sel_frames, final_nums, sel_feats
        )
    except ValueError:
        # adaptive_select_from_segments returned only frames & nums
        adapt_frames, adapt_nums = adaptive_select_from_segments(
            sel_frames, final_nums, sel_feats
        )
        # default raw_s3 = 0 for each selected frame
        raw_s3 = [0.0] * len(adapt_nums)

    # Normalize s3 only over adaptive set
    s3 = normalize(raw_s3)
    s3_map = {adapt_nums[i]: s3[i] for i in range(len(adapt_nums))}

    # 7) LLaVA + GPT labeling (unchanged)
    analysis_data = []
    for num in adapt_nums:
        frame = frames[num-1]
        llava_output = process_frame_with_llava(frame)
        analysis_data.append({"frame_number": num, "llava_output": llava_output})
    analysis_path = os.path.join(output_base, "frame_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis_data, f, indent=4)
    update_json_labels(analysis_folder=output_base)


    _, _, frames_with_labels, _, _ = process_frames_with_labels_clip(
        filtered_folder,
        analysis_path,
        threshold=0.9,
        output_json=os.path.join(output_base, "frames_with_high_lap_scores.json"),
        unused_labels_json=os.path.join(output_base, "unused_labels.json")
    )
    clip_map = {}

    for entry in frames_with_labels:
        num = int(entry["frame_name"].split('_')[1].split('.')[0])
        best_score = max(lbl['score'] for lbl in entry['labels'])
        clip_map[num] = best_score
    # align s4 to adapt_nums
    s4 = normalize([clip_map.get(num, 0.0) for num in adapt_nums])
    s4_map = {adapt_nums[i]: s4[i] for i in range(len(adapt_nums))}

    # 9) 2x2 Combination labeling -> s5
    combo_map = {}
    group_size = 4
    sorted_adapt = sorted(adapt_nums)
    for i in range(0, len(sorted_adapt), group_size):
        grp = sorted_adapt[i:i+group_size]
        if len(grp) < group_size: break
        counts = {}
        for num in grp:
            label = next((e['best_label'] for e in frames_with_labels if int(e['frame_name'].split('_')[1].split('.')[0])==num), None)
            counts[label] = counts.get(label, 0) + 1
        for num in grp:
            combo_map[num] = counts.get(next((e['best_label'] for e in frames_with_labels if int(e['frame_name'].split('_')[1].split('.')[0])==num), None), 0) / group_size
    s5 = normalize([combo_map.get(num,0.0) for num in adapt_nums])
    s5_map = {adapt_nums[i]: s5[i] for i in range(len(adapt_nums))}

      # 10) Compile and normalize across ALL extracted frames
    # Build raw per-stage scores for every extracted frame
    total_frames = len(frames)
    # s1: raw ResNet diffs normalized across all frames
    raw_diffs = [float(np.linalg.norm(features[i] - features[i+1])) for i in range(len(frames)-1)]
    if raw_diffs:
        raw_diffs.append(raw_diffs[-1])
    else:
        raw_diffs = [0.0]
    s1 = normalize(raw_diffs)

    # s2: adaptive sampling score per group of 10, normalized per group; 0 if not adaptively selected
    s2 = [0.0] * total_frames
    group_size = 10
    for start in range(0, total_frames, group_size):
        end = min(start + group_size, total_frames)
        grp_idxs = list(range(start, end))
        grp_diffs = [raw_diffs[i] for i in grp_idxs]
        mn, mx = min(grp_diffs), max(grp_diffs)
        for i in grp_idxs:
            if (i+1) in adapt_nums:
                if mx - mn < 1e-8:
                    s2[i] = 0.0
                else:
                    s2[i] = (raw_diffs[i] - mn) / (mx - mn)
            else:
                s2[i] = 0.0

    # s3: CLIP best-label score, 0 if missing
    s3 = [clip_map.get(i+1, 0.0) for i in range(total_frames)]

    # s4: 4-frame combination consistency score, 0 if missing
    s4 = [combo_map.get(i+1, 0.0) for i in range(total_frames)]

    # Sum and average per frame over 4 stages
    final_scores = []
    for i in range(total_frames):
        total = s1[i] + s2[i] + s3[i] + s4[i]
        avg_score = total / 4.0
        final_scores.append({"frame": i+1, "score": avg_score})

    # Write JSON for all frames
    with open(os.path.join(output_base, "frame_scores.json"), 'w') as f:
        json.dump(final_scores, f, indent=2)

    print(f"✅ Completed pipeline and wrote scores for all {total_frames} frames")
    generate_combined_summaries(
        filtered_folder=filtered_folder,
        high_score_json=os.path.join(output_base, "frames_with_high_lap_scores.json"),
        combined_folder=combined_folder,
        output_json=os.path.join(output_base, "combined_summary_minicpm-v.json")
    )

    process_all_combined_summaries(
        input_dir=output_base,
        output_dir=output_base,
        final_prefix="report_lap_"
    )

    merge_audio_transcript_with_summary(
        final_summary_json=os.path.join(output_base, "report_lap_minicpm-v.json"),
        transcript_text=entire_transcript,
        output_json=os.path.join(output_base, "complete_surgical_report.json")
    )
    print(f"✅ Completed pipeline for: {os.path.basename(video_path)}")
    print(f"✅ Completed and saved scores for {os.path.basename(video_path)}")


def run_pipeline_on_all_videos(base_video_folder="./videos", base_output_folder="./test/minicpm_act"):
    """
    Runs the pipeline on all videos in a folder and saves outputs in separate folders.
    Skips videos if their output folder already exists.
    """
    os.makedirs(base_output_folder, exist_ok=True)
    video_files = [f for f in os.listdir(base_video_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    for video_file in video_files:
        video_path = os.path.join(base_video_folder, video_file)
        video_name = os.path.splitext(video_file)[0]
        output_base = os.path.join(base_output_folder, video_name)

        # Skip if folder already exists
        if os.path.exists(output_base):
            print(f"⏩ Skipping {video_name} — output already exists.")
            continue

        extracted_folder = os.path.join(output_base, "extracted_frames")
        filtered_folder = os.path.join(output_base, "filtered_frames")
        combined_folder = os.path.join(output_base, "combined_frames_resec")

        os.makedirs(extracted_folder, exist_ok=True)
        os.makedirs(filtered_folder, exist_ok=True)
        os.makedirs(combined_folder, exist_ok=True)

        print(f"\n✅ Starting pipeline for: {video_name}")
        full_pipeline_example(video_path, output_base, extracted_folder, filtered_folder, combined_folder)

    print("\n🎉 All videos processed (skipping existing ones).")

# Optional: run directly
if __name__ == "__main__":
    run_pipeline_on_all_videos()

