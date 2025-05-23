import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import evaluate
from openai import OpenAI

# --- Data ---
#Activity Net Captions
# ground_truth = [
#     "A lady is combing another lady's hair as the title is shown.",
#     "We see the supplies the lady will be using as she talks to the camera.",
#     "The stylist combs the customer's hair, and adds a roller.",
#     "The stylist removes the roller and adds another.",
#     "The stylist removes that one and adds a vertical one.",
#     "The stylist speaks and the video ends."
# ]
# predicted = [
#     "In this instructional tutorial, Amelia Smith shares her professional insights on achieving voluminous curls using hot roller pins.",
#     "The video provides a step-by-step approach, beginning with a demonstration of necessary tools—styling comb, rollers, and pins—emphasizing the superior functionality of alligator-type clips.",
#     "Amelia guides viewers through the process of preparing the hair: combing, sectioning, and properly aligning rollers for optimal curl definition.",
#     "She offers practical tips, such as adjusting the pin's angle to ensure a secure hold, adapting the pin placement depending on the roller orientation, and modifying pins over time to maintain their snugness.",
#     "The tutorial concludes with essential final adjustments, showcasing Amelia's dedication to detail in setting the perfect hairstyle.",
#     "Her approachable explanations and expert advice empower viewers to confidently craft stunning curls, fostering both beginner and seasoned stylists to enhance their hairstyling skills from the comfort of home."
# ]
#Youcook2
ground_truth =[
    "add miso paste soy sauce frozen veggies and the mushrooms to the pot of water. Mix and boil the ingredients.",
    "add some udon noodles to the broth.",
    "add some leaves of chard and tofu to the broth."
]
predicted =[
    "Begin by boiling water in a large pot. Once boiling, add miso paste, soy sauce, frozen veggies, and mushrooms, then let the mixture sit for a few minutes.",
    "Once the broth starts boiling again, add the pre-cooked udon noodles and let them cook for a couple of minutes.",
    "Finally, incorporate shredded green onions and cubed tofu into the pot. Throughout the process, monitor the cooking to ensure everything is perfectly done. Serve the soup hot, enjoying the blend of flavors."
]


# --- Metric Setup ---
smoothie = SmoothingFunction().method4
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")
semncg_metric = evaluate.load("nbansal/semncg", model_name="all-MiniLM-L6-v2")

# --- LLM Judge Setup (as per your code) ---
client = OpenAI(api_key="<YOUR OPENAI KEY HERE>")


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

def prompt_factual_accuracy(summary, reference):
    return f"""Rate from 1 to 5 how factually accurate and relevant the summary is compared to the reference.
The focus should be on the fact that does the summary suggest the steps in the reference or mention them
Summary, if yes score higher:
\"\"\"{summary}\"\"\"
Reference:
\"\"\"{reference}\"\"\"
Only return a number from 1 to 5."""

def prompt_completeness(summary, reference):
    return f"""Rate from 1 to 5 how completely the summary covers the key content in the reference.
It could even give a better and complete description than the reference text.
Summary:
\"\"\"{summary}\"\"\"
Reference:
\"\"\"{reference}\"\"\"
Only return a number from 1 to 5."""

def prompt_detail_specificity(summary, reference):
    return f"""Rate from 1 to 5 how specific and detailed the summary is, compared to the reference.

Summary:
\"\"\"{summary}\"\"\"
Reference:
\"\"\"{reference}\"\"\"
Only return a number from 1 to 5."""

def prompt_relevance(summary, reference):
    return f"""Rate from 1 to 5 how well the summary captures key actions or events in the reference.

Summary:
\"\"\"{summary}\"\"\"
Reference:
\"\"\"{reference}\"\"\"
Only return a number from 1 to 5."""

def prompt_redundancy(summary, reference):
    return f"""Rate from 1 to 5 how well the summary avoids repetition.
Summary:
\"\"\"{summary}\"\"\"
Reference:
\"\"\"{reference}\"\"\"
Only return a number from 1 to 5."""

def get_llm_judge_score(summary, reference):
    prompts = [
        prompt_factual_accuracy(summary, reference),
        prompt_completeness(summary, reference),
        prompt_detail_specificity(summary, reference),
        prompt_relevance(summary, reference),
        prompt_redundancy(summary, reference)
    ]
    scores = []
    for prompt in prompts:
        try:
            score = int(call_gpt_text(prompt).strip())
            scores.append(min(max(score, 1), 5))
        except:
            scores.append(0)
    factual = scores[0]
    composite = round((2 * factual + sum(scores[1:])) / 6, 2)
    return {
        "Factual Accuracy": factual,
        "Completeness": scores[1],
        "Detail/Specificity": scores[2],
        "Relevance": scores[3],
        "Redundancy": scores[4],
        "LLM_Judge_Score": composite
    }

# --- Compute Metrics Per Segment ---
bleu_scores, rougeL_scores, meteor_scores = [], [], []
bert_precision, bert_recall, bert_f1, semncg_scores, llm_scores = [], [], [], [], []

for ref, pred in zip(ground_truth, predicted):
    # BLEU
    bleu = sentence_bleu([nltk.word_tokenize(ref)], nltk.word_tokenize(pred), smoothing_function=smoothie)
    bleu_scores.append(bleu)
    # ROUGE-L
    rouge_result = scorer.score(ref, pred)['rougeL']
    rougeL_scores.append(rouge_result.fmeasure)
    # METEOR
    meteor_result = meteor.compute(predictions=[pred], references=[ref])
    meteor_scores.append(meteor_result['meteor'])
    # BERTScore
    bert_result = bertscore.compute(predictions=[pred], references=[ref], lang="en")
    bert_precision.append(bert_result['precision'][0])
    bert_recall.append(bert_result['recall'][0])
    bert_f1.append(bert_result['f1'][0])
    # SemNCG
    semncg_mean, _ = semncg_metric.compute(predictions=[pred], references=[ref], documents=[ref])
    semncg_scores.append(semncg_mean)
    # LLM Judge 
    llm = get_llm_judge_score(pred, ref)
    llm_scores.append(llm["LLM_Judge_Score"])

for idx, (ref, pred) in enumerate(zip(ground_truth, predicted)):
    print(f"\n--- Segment {idx} ---")
    print(f"Predicted:   {pred}")
    print(f"Ground Truth:{ref}")
    print(f"BLEU:        {bleu_scores[idx]}")
    print(f"ROUGE-L:     {rougeL_scores[idx]}")
    print(f"METEOR:      {meteor_scores[idx]}")
    print(f"BERTScore P: {bert_precision[idx]}")
    print(f"BERTScore R: {bert_recall[idx]}")
    print(f"BERTScore F1:{bert_f1[idx]}")
    print(f"SemNCG:      {semncg_scores[idx]}")
    print(f"LLM Judge: {llm_scores[idx]/5}")

# --- Print Total Averages ---
print("BLEU:", sum(bleu_scores) / len(bleu_scores))
print("ROUGE-L:", sum(rougeL_scores) / len(rougeL_scores))
print("METEOR:", sum(meteor_scores) / len(meteor_scores))
print("BERTScore Precision:", sum(bert_precision) / len(bert_precision))
print("BERTScore Recall:", sum(bert_recall) / len(bert_recall))
print("BERTScore F1:", sum(bert_f1) / len(bert_f1))
print("SemNCG:", sum(semncg_scores) / len(semncg_scores))
print("LLM Judge (Composite, avg):", sum(llm_scores) / len(llm_scores))
