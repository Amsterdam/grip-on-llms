"""Implementation of summarization, simplification, translation, etc metrics"""

import logging

import bert_score.utils
import evaluate

hf_rouge = evaluate.load("rouge")
hf_sari = evaluate.load("sari")
hf_bleu = evaluate.load("bleu")
hf_meteor = evaluate.load("meteor")
# FYI: bertscore for dutch supposedly uses bert-base-multilingual-cased
hf_bertscore = evaluate.load("bertscore")

_bert_score_sent_encode = bert_score.utils.sent_encode


def _sent_encode(tokenizer, sent):
    """Encode empty sentences without build_inputs_with_special_tokens (gone in transformers 5).

    bert_score 0.3.13 calls tokenizer.build_inputs_with_special_tokens([]) for empty strings;
    encoding "" with special tokens gives the same ids ([CLS] [SEP]), so scores are unchanged.
    """
    if sent.strip() == "":
        return tokenizer("", add_special_tokens=True)["input_ids"]
    return _bert_score_sent_encode(tokenizer, sent)


bert_score.utils.sent_encode = _sent_encode


def rouge(predictions, references):
    # references = list(map(list, references))
    logging.info("Calculating ROUGE")
    try:
        score = hf_rouge.compute(predictions=predictions, references=references)
    except Exception as e:
        logging.warning(f"ROUGE failed: {e}")
        score = None
    return score


def sari(sources, predictions, references):
    logging.info("Calculating SARI")
    try:
        references = list(map(list, references))
        score = hf_sari.compute(sources=sources, predictions=predictions, references=references)
    except Exception as e:
        logging.warning(f"SARI failed: {e}")
        score = None
    return score


def bleu(predictions, references):
    logging.info("Calculating BLEU")
    try:
        score = hf_bleu.compute(predictions=predictions, references=references)
    except Exception as e:
        logging.warning(f"BLEU failed: {e}")
        score = None
    return score


def meteor(predictions, references):
    logging.info("Calculating METEOR")
    try:
        score = hf_meteor.compute(predictions=predictions, references=references)
    except Exception as e:
        logging.warning(f"METEOR failed: {e}")
        score = None
    return score


def bertscore(predictions, references, lang="nl"):
    logging.info("Calculating BERTScore")
    try:
        score = hf_bertscore.compute(predictions=predictions, references=references, lang=lang)
    except Exception as e:
        logging.warning(f"BERTScore failed: {e}")
        score = None
    return score


if __name__ == "__main__":
    # Example data from the INT Duidelijke Taal dataset
    sources = [
        "Thans gaat ongeveer 55% van de kinderen in de schoolgaande leeftijd naar "
        "school, waarvan 48% uit meisjes bestaat.",
        "Hoewel 47% van de beroepsbevolking in Mauritanië werkzaam is in de landbouw, "
        "is het land niet zelfvoorzienend.",
    ]
    predictions = [
        "Nu gaat ongeveer 55% van alle kinderen die naar school kunnen, ook echt naar "
        "school. Van deze groep is 48% meisje.",
        "In Mauritanië werkt 47% van de mensen in de landbouw. Toch kan het land niet "
        "genoeg voedsel voor zichzelf maken.",
    ]
    references = [
        "Op dit moment gaat iets meer dan de helft van de kinderen die naar school "
        "kunnen, namelijk 55%, ook daadwerkelijk naar school. Van deze groep zijn 48% "
        "meisjes.",
        "In Mauritanië werkt bijna de helft van de mensen in de landbouw. Toch kan "
        "het land niet al het voedsel produceren dat het nodig heeft.",
    ]

    sari_score = sari(sources=sources, predictions=predictions, references=references)
    bleu_score = bleu(predictions=predictions, references=references)
    meteor_score = meteor(predictions=predictions, references=references)
    bert_score = bertscore(predictions=predictions, references=references, lang="nl")

    print(f"Sari: {sari_score}")
    print(f"BLEU: {bleu_score}")
    print(f"Meteor: {meteor_score}")
    print(f"bertscore: {bert_score}")
