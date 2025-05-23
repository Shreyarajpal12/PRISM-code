# PRISM: Procedural Summarization for Instructional Videos

PRISM is a semantic summarization pipeline that processes instructional videos (like cooking or daily activities) using frame selection, CLIP-based visual understanding, Whisper transcription, and LLM-based captioning. It is evaluated using standard metrics like BLEU, METEOR, ROUGE, and BERTScore.

---

## 📦 Datasets

Download dataset videos via:
```bash
python ycdown.py
````

* [YouCook2 Dataset (lmms-lab/YouCook2)](https://huggingface.co/datasets/lmms-lab/YouCook2)
* [ActivityNet Captions (HuggingFaceM4/ActivitiyNet\_Captions)](https://huggingface.co/datasets/Leyo/ActivityNet_Captions)

---

## 🚀 Running the Pipeline

To run PRISM on a folder of videos:

```bash
python main_act.py
```

All intermediate steps including frame extraction, adaptive sampling, semantic filtering, label generation, and report summarization will run end-to-end.

---

## 🧪 Evaluation

Evaluation is based on:

* **BLEU**
* **METEOR**
* **ROUGE**
* **BERTScore**

These are computed using Hugging Face's `evaluate` and `bertscore` libraries.
The complete script and scores are available in the eval.py

---

## 📁 Key Files

* `main_act.py` – Main pipeline for summarization
* `ycdown.py` – Downloads and organizes YouCook2 and ActivityNet Captions datasets
* `requirements.txt` – List of required dependencies

---

## 📥 Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

If Whisper is used for transcription, install:

```bash
pip install git+https://github.com/openai/whisper.git
```

Also ensure `ffmpeg` is installed system-wide (e.g. `sudo apt install ffmpeg` on Linux).

---



## 📄 License

This project is licensed under the MIT License.

